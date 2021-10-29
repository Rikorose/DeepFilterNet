import atexit
import queue
import threading
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch._utils import ExceptionWrapper
from torch.utils.data._utils.pin_memory import _pin_memory_loop

from libdfdata import _FdDataLoader, _TdDataLoader


class Batch:
    def __init__(self, b: Tuple[np.ndarray, ...]):
        # Pytorch complains that the returned numpy arrays are not writable. Since they were
        # allocated within the python GIL and their content is only used for this batch, it is
        # safe to assume writable.
        if len(b) == 10:  # FftDataloader
            speech, noise, noisy, erb, spec, lengths, max_freq, snr, gain, atten = b
            if erb.size <= 1:
                self.feat_erb = None
            if spec.size <= 1:
                self.feat_spec = None
            else:
                self.feat_erb = torch.from_numpy(erb)
                self.feat_spec = torch.from_numpy(spec)
        else:
            speech, noise, noisy, lengths, max_freq, snr, gain, atten = b
            self.feat_erb = None
            self.feat_spec = None
        self.speech = torch.from_numpy(speech)
        self.noise = torch.from_numpy(noise)
        self.noisy = torch.from_numpy(noisy)
        self.lengths = torch.from_numpy(lengths.astype(np.int64)).long()
        self.snr = torch.from_numpy(snr)
        self.gain = torch.from_numpy(gain)
        self.atten = torch.from_numpy(atten).long()
        self.atten[self.atten == 0] = 1000
        self.max_freq = torch.from_numpy(max_freq.astype(np.int32)).long()

    def pin_memory(self):
        self.speech = self.speech.pin_memory()
        self.noisy = self.noisy.pin_memory()
        if self.feat_erb is not None:
            self.feat_erb = self.feat_erb.pin_memory()
        if self.feat_spec is not None:
            self.feat_spec = self.feat_spec.pin_memory()
        if self.max_freq is not None:
            self.max_freq = self.max_freq.pin_memory()
        self.atten.pin_memory()
        return self

    def __repr__(self):
        bs = len(self.lengths)
        s = f"Batch of size {bs}:\n"
        for i in range(bs):
            s += f"    length: {self.lengths[i]}\n"
            s += f"    snr: {self.snr[i]}\n"
            s += f"    gain: {self.gain[i]}\n"
        return s


class PytorchDataLoader:
    def __init__(
        self,
        ds_dir: str,
        ds_config: str,
        sr: int,
        batch_size: int,
        max_len_s: Optional[float] = 10.0,
        prefetch=4,
        num_workers=None,
        pin_memory=True,
        fft_dataloader=False,  # Following parameters are only used if fft_dataloader == True
        fft_size: int = None,  # FFT size for stft calcualtion
        hop_size: int = None,  # Hop size for stft calcualtion
        nb_erb: int = None,  # Number of ERB bands
        nb_spec: int = None,  # Number of complex spectrogram bins
        norm_alpha: float = None,  # Exponential normalization decay for erb_feat and spec_feat
        batch_size_eval: Optional[int] = None,  # Different batch size for evaluation
        p_atten_lim: Optional[float] = None,  # Limit attenuation by providing a noisy target
        p_reverb: Optional[float] = None,  # Percentage of reverberant speech/noise samples
        overfit=False,  # Overfit on one epoch
        seed=0,
        min_nb_erb_freqs: int = None,  # Minimum number of frequency bins per ERB band
    ):
        self.fft_size = fft_size
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        prefetch_loader = prefetch * batch_size * num_workers
        logger.info(f"Initializing dataloader with data directory {ds_dir}")
        if fft_dataloader:
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
                p_atten_lim=p_atten_lim,
                p_reverb=p_reverb,
                prefetch=prefetch_loader,
                overfit=overfit,
                seed=seed,
                min_nb_erb_freqs=min_nb_erb_freqs,
            )
        else:
            self.loader = _TdDataLoader(
                ds_dir,
                ds_config,
                sr,
                max_len_s=max_len_s,
                batch_size=batch_size,
                batch_size_eval=batch_size_eval,
                num_threads=num_workers,
                p_atten_lim=p_atten_lim,
                p_reverb=p_reverb,
                prefetch=prefetch_loader,
                overfit=overfit,
                seed=seed,
                min_nb_erb_freqs=min_nb_erb_freqs,
            )
        self.prefetch = prefetch
        self.pin_memory = pin_memory if torch.cuda.is_available() else False
        self.idx = 0
        self.worker_out_queue = self._get_worker_queue_dummy()
        self.pin_memory_thread_done_event: Optional[threading.Event] = None
        self.pin_memory_thread: Optional[threading.Thread] = None
        self.data_queue = None
        atexit.register(self.loader.cleanup)

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
        return self.loader.len_of(split)

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
                    if str(e) == "DF dataset error: TimeoutError":
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
        _, batch = self.data_queue.get()
        if isinstance(batch, ExceptionWrapper):
            batch.reraise()
        return batch

    def iter_epoch(self, split: str, seed: int) -> Iterator[Batch]:
        self.idx = 0
        # Initializes workers. This needs to be done before pin_memory thread is
        # started via setup_data_queue().
        self.loader.start_epoch(split, seed)
        # Initialize data out queue (maybe incl. pin memory thread)
        self.setup_data_queue()
        try:
            if self.pin_memory:
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
            return
