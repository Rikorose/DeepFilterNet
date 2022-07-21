import atexit
import os
import queue
import threading
import time
import warnings
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch._utils import ExceptionWrapper
from torch.utils.data._utils.pin_memory import _pin_memory_loop

from libdfdata import _FdDataLoader


class Batch:
    def __init__(self, b: Tuple[np.ndarray, ...]):
        # Pytorch complains that the returned numpy arrays are not writable. Since they were
        # allocated within the python GIL and their content is only used for this batch, it is
        # safe to assume writable.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert len(b) == 10
            speech, noisy, erb, spec, lengths, max_freq, snr, gain, timings = b
            if erb.size <= 1:
                self.feat_erb = None
            if spec.size <= 1:
                self.feat_spec = None
            else:
                self.feat_erb = torch.from_numpy(erb)
                self.feat_spec = torch.from_numpy(spec)
            self.speech = torch.from_numpy(speech)
            self.noisy = torch.from_numpy(noisy)
            self.lengths = torch.from_numpy(lengths.astype(np.int64)).long()
            self.snr = torch.from_numpy(snr)
            self.gain = torch.from_numpy(gain)
            self.max_freq = torch.from_numpy(max_freq.astype(np.int32)).long()
            self.timings = torch.from_numpy(timings.astype(np.float32))

    def pin_memory(self):
        self.speech = self.speech.pin_memory()
        self.noisy = self.noisy.pin_memory()
        if self.feat_erb is not None:
            self.feat_erb = self.feat_erb.pin_memory()
        if self.feat_spec is not None:
            self.feat_spec = self.feat_spec.pin_memory()
        if self.max_freq is not None:
            self.max_freq = self.max_freq.pin_memory()
        return self

    def __repr__(self):
        bs = len(self.lengths)
        s = f"Batch of size {bs}:\n"
        snrs = "".join(f"{s}," for s in self.snr)[:-1]
        gains = "".join(f"{g}," for g in self.gain)[:-1]
        s += f"    SNRs: {snrs}\n"
        s += f"    Gains: {gains}\n"
        return s


class PytorchDataLoader:
    def __init__(
        self,
        ds_dir: str,
        ds_config: str,
        sr: int,
        batch_size: int,
        max_len_s: Optional[float] = 10.0,
        prefetch: int = 8,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        drop_last: bool = False,  # Drop the last batch if it contains fewer samples then batch_size
        fft_size: Optional[int] = None,  # FFT size for stft calcualtion
        hop_size: Optional[int] = None,  # Hop size for stft calcualtion
        nb_erb: Optional[int] = None,  # Number of ERB bands
        nb_spec: Optional[int] = None,  # Number of complex spectrogram bins
        norm_alpha: Optional[float] = None,  # Exponential normalization decay for erb/spec_feat
        batch_size_eval: Optional[int] = None,  # Different batch size for evaluation
        p_reverb: Optional[float] = None,  # Percentage of reverberant speech/noise samples
        p_bw_ext: Optional[float] = None,  # Percentage of bandwidth limited signal for extension
        overfit: bool = False,  # Overfit on one epoch
        cache_valid: bool = False,  # Cache validiation dataset
        seed: int = 0,
        min_nb_erb_freqs: Optional[int] = None,  # Minimum number of frequency bins per ERB band
        log_timings: bool = False,
        global_sampling_factor: Optional[float] = None,  # Additional over/undersampling of all ds
        snrs=None,  # Signal to noise ratios (SNRs) to generate. Defaults to [-5,0,5,10,20,40] dB
        gains=None,  # Additional gains applied to speech. Defaults to [-6,0,6] dB
        log_level: Optional[str] = None,  # Log level for dataloader logging
    ):
        self.fft_size = fft_size
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        prefetch_loader = batch_size * (num_workers or 1)
        logger.info(f"Initializing dataloader with data directory {ds_dir}")
        assert self.fft_size is not None, "No fft_size provided"
        self.loader = _FdDataLoader(
            ds_dir,
            ds_config,
            sr,
            max_len_s=max_len_s,
            batch_size=batch_size,
            batch_size_eval=batch_size_eval,
            num_threads=num_workers,
            fft_size=self.fft_size,
            hop_size=hop_size,
            nb_erb=nb_erb,
            nb_spec=nb_spec,
            norm_alpha=norm_alpha,
            p_reverb=p_reverb,
            p_bw_ext=p_bw_ext,
            prefetch=prefetch_loader,
            drop_last=drop_last,
            overfit=overfit,
            cache_valid=cache_valid,
            seed=seed,
            min_nb_erb_freqs=min_nb_erb_freqs,
            global_sampling_factor=global_sampling_factor,
            snrs=snrs,
            gains=gains,
            log_level=log_level,
        )
        self.log_dataloader_msgs()
        self.prefetch = prefetch
        self.pin_memory = pin_memory if torch.cuda.is_available() else False
        self.idx = 0
        self.worker_out_queue = self._get_worker_queue_dummy()
        self.pin_memory_thread_done_event: Optional[threading.Event] = None
        self.pin_memory_thread: Optional[threading.Thread] = None
        self.data_queue = None
        self.log_timings = log_timings
        self.timings_py: List[float] = []
        self.timings_rs: List[torch.Tensor] = []
        atexit.register(self.loader.cleanup)

    def set_batch_size(self, batch_size: int, split: str):
        self.loader.set_batch_size(batch_size, split)

    def cleanup_pin_memory_thread(self):
        if self.pin_memory:
            # Check if still running from previous epoch
            if self.pin_memory_thread is not None and self.pin_memory_thread.is_alive():
                self.pin_memory_thread_done_event.set()
                while True:
                    try:  # Empty queue
                        _ = self.data_queue.get(timeout=2)
                    except queue.Empty:
                        break
                self.pin_memory_thread.join()
                self.data_queue.join()
                print("Pinmemory cleanup done")

    def setup_data_queue(self):
        if self.pin_memory:
            self.cleanup_pin_memory_thread()
            self.pin_memory_thread_done_event = threading.Event()
            self.data_queue = queue.Queue(maxsize=self.prefetch)
            pin_memory_thread = threading.Thread(
                target=_pin_memory_loop,
                args=(
                    self.worker_out_queue,
                    self.data_queue,
                    torch.cuda.current_device(),
                    self.pin_memory_thread_done_event,
                ),
                name="PinMemoryLoop",
            )
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            self.pin_memory_thread = pin_memory_thread
        else:
            self.data_queue = self.worker_out_queue

    def len(self, split: str) -> int:
        return self.loader.dataloader_len(split)

    def __len__(self) -> int:
        """Return training length."""
        return self.len("train")

    def dataset_len(self, split: str) -> int:
        return self.loader.dataset_len(split)

    def cleanup(self):
        self.loader.cleanup()

    def _get_worker_queue_dummy(self):
        class _Queue:
            def get(*__args, **__kwargs) -> Tuple[int, Batch]:
                idx = 0
                try:
                    idx, batch = self.idx, self.loader.get_batch()
                    batch = Batch(batch)
                except RuntimeError as e:
                    self.loader.cleanup()
                    if str(e) == "DF dataloader error: TimeoutError":
                        logger.error("{}. Stopping epoch.".format(str(e)))
                        self.cleanup_pin_memory_thread()
                        exit(1)
                    logger.error("Error during get_batch(): {}".format(str(e)))
                    raise e
                except StopIteration:
                    batch = ExceptionWrapper(where="in pin memory worker queue")
                self.idx += 1
                return idx, batch

        q = _Queue()
        return q

    def _get_batch(self) -> Batch:
        if self.log_timings:
            t0 = time.time()
        _, batch = self.data_queue.get()
        if isinstance(batch, ExceptionWrapper):
            batch.reraise()
        if self.log_timings:
            self.timings_py.append(time.time() - t0)
            self.timings_rs.append(batch.timings)
        self.log_dataloader_msgs()
        return batch

    def log_dataloader_msgs(self):
        # message has type (level: str, message: str, module: Optional[str], lineno: Optional[int])
        for (level, msg, module, lineno) in self.loader.get_log_messages():
            # with logger.contextualize(module: "Dataloader", file=file, lineno=lineno):
            def patch(r):
                r["file"] = {"file": os.path.basename(module) + ".rs", "path": "pyDF-data/src/"}
                r["module"] = module
                r["function"] = module
                r["line"] = lineno

            logger.patch(patch).log(level, msg)

    def iter_epoch(self, split: str, seed: int) -> Iterator[Batch]:
        self.idx = 0
        # Initializes workers. This needs to be done before pin_memory thread is
        # started via setup_data_queue().
        self.loader.start_epoch(split, seed)
        self.log_dataloader_msgs()
        if self.log_timings:
            self.timings_py = []
            self.timings_rs = []
        # Initialize data out queue (maybe incl. pin memory thread)
        self.setup_data_queue()
        try:
            if self.pin_memory:
                if self.pin_memory_thread is None:
                    raise ValueError("pin memory thread is none.")
                while self.pin_memory_thread.is_alive():
                    batch = self._get_batch()
                    yield batch
                else:
                    raise RuntimeError("Pin memory thread exited unexpectedly")
            else:
                while True:
                    batch = self._get_batch()
                    yield batch
        except StopIteration:
            if self.pin_memory_thread_done_event is not None:
                self.pin_memory_thread_done_event.set()
            if self.log_timings:
                logger.info("Avg batch loading time (py): {:.4f}s", np.mean(self.timings_py))
                # Last batch may be shorter, just skip it
                if len(self.timings_rs[-1]) != len(self.timings_rs[0]):
                    self.timings_rs = self.timings_rs[:-1]
                timings_rs = torch.stack(self.timings_rs)
                timings_samples = timings_rs[:, :-1]
                timings_batch = timings_rs[:, -1]  # last if for whole batch
                logger.info("Avg batch loading time (rs): {:.4f}s", timings_batch.mean().item())
                logger.info(
                    "Min/Avg/Max sample loading time (rs): {:.4f}/{:.4f}/{:.4f}s",
                    timings_samples.min().item(),
                    timings_samples.mean().item(),
                    timings_samples.max().item(),
                )
            return
