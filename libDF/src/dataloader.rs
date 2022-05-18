use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::fmt;
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::time::Instant;

use crossbeam_channel::unbounded;
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::SliceRandom;
use rayon;
use rayon::{current_num_threads, prelude::*};
use thiserror::Error;

type Result<T> = std::result::Result<T, DfDataloaderError>;

use crate::{augmentations::*, dataset::*, util::*, Complex32};

#[derive(Error, Debug)]
pub enum DfDataloaderError {
    #[error("Dataloading Timeout")]
    TimeoutError,
    #[error("Channels not initialized. Have you already called start_epoch()?")]
    ChannelsNotInitializedError,
    #[error(
        "Dataset {split} size ({dataset_size}) smaller than batch size ({batch_size}). Try increasing the dataset sampling factor or decreasing the batch size."
    )]
    DatasetTooSmall {
        split: Split,
        dataset_size: usize,
        batch_size: usize,
    },
    #[error("Dataset Drained")]
    DatasetDrained,
    #[error("Multithreading Send Error: {0:?}")]
    SendError(String),
    #[error("Thread Join Error: {0:?}")]
    ThreadJoinError(String),
    #[error("Threadpool Builder Error")]
    ThreadPoolBuildError(#[from] rayon::ThreadPoolBuildError),
    #[error("DF Transforms Error")]
    TransformError(#[from] crate::transforms::TransformError),
    #[error("DF Augmentation Error")]
    AugmentationError(#[from] crate::augmentations::AugmentationError),
    #[error("DF Utils Error")]
    UtilsError(#[from] crate::util::UtilsError),
    #[error("DF Dataset Error")]
    DatasetError(#[from] crate::dataset::DfDatasetError),
    #[error("Ndarray Shape Error")]
    NdarrayShapeError(#[from] ndarray::ShapeError),
}

impl<T> From<std::sync::mpsc::SendError<T>> for DfDataloaderError {
    fn from(error: std::sync::mpsc::SendError<T>) -> Self {
        DfDataloaderError::SendError(error.to_string())
    }
}

pub struct DataLoader {
    ds_train: Option<Arc<FftDataset>>, // Option is needed to retake ownership via option.take()
    ds_valid: Option<Arc<FftDataset>>,
    ds_test: Option<Arc<FftDataset>>,
    batch_size_train: usize,
    batch_size_eval: usize,
    num_workers: usize,
    num_prefech: usize,
    idcs: Arc<Mutex<VecDeque<(usize, isize)>>>,
    current_split: Split,
    fill_thread: Option<thread::JoinHandle<Result<()>>>,
    out_receiver: Option<Receiver<(usize, Result<Sample<Complex32>>)>>,
    out_buf: BTreeMap<usize, Sample<Complex32>>,
    cur_out_idx: usize,
    drop_last: bool,
    drained: bool,
    overfit: bool,
}

#[derive(Default)]
pub struct DataLoaderBuilder {
    _ds: Option<Datasets>,
    _batch_size: Option<usize>,
    _batch_size_eval: Option<usize>,
    _prefetch: Option<usize>,
    _num_threads: Option<usize>,
    _drop_last: bool,
    _overfit: bool,
}

impl DataLoaderBuilder {
    pub fn new(ds: Datasets) -> Self {
        DataLoaderBuilder {
            _ds: Some(ds),
            _batch_size: None,
            _batch_size_eval: None,
            _prefetch: None,
            _num_threads: None,
            _drop_last: false,
            _overfit: false,
        }
    }
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self._batch_size = Some(batch_size);
        self
    }
    pub fn batch_size_eval(mut self, batch_size: usize) -> Self {
        self._batch_size_eval = Some(batch_size);
        self
    }
    pub fn prefetch(mut self, prefetch: usize) -> Self {
        self._prefetch = Some(prefetch);
        self
    }
    pub fn num_threads(mut self, num_threads: usize) -> Self {
        self._num_threads = Some(num_threads);
        self
    }
    pub fn overfit(mut self) -> Self {
        self._overfit = true;
        self
    }
    pub fn drop_last(mut self) -> Self {
        self._drop_last = true;
        self
    }
    pub fn build(self) -> Result<DataLoader> {
        let bs_train = self._batch_size.unwrap_or(1);
        let prefetch = self._prefetch.unwrap_or(bs_train * self._num_threads.unwrap_or(4));
        let mut loader = DataLoader::new(
            self._ds.unwrap(),
            bs_train,
            self._batch_size_eval,
            prefetch,
            self._num_threads,
            self._drop_last,
        )?;
        loader.overfit = self._overfit;
        Ok(loader)
    }
}

impl DataLoader {
    pub fn builder(ds: Datasets) -> DataLoaderBuilder {
        DataLoaderBuilder::new(ds)
    }
    pub fn new(
        datasets: Datasets,
        batch_size_train: usize,
        batch_size_eval: Option<usize>,
        num_prefech: usize,
        num_threads: Option<usize>,
        drop_last: bool,
    ) -> Result<Self> {
        // Register global rayon threadpool. It will only be used for data loader workers.
        hdf5::sync::sync(|| {});
        let num_workers = num_threads.unwrap_or_else(current_num_threads);
        hdf5::sync::sync(|| {});
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .thread_name(|idx| format!("DataLoader Worker {}", idx))
            .start_handler(|_| hdf5::sync::sync(|| {}))
            .build_global()
            .unwrap_or(());
        let batch_size_eval = batch_size_eval.unwrap_or(batch_size_train);
        Ok(DataLoader {
            ds_train: Some(Arc::new(datasets.train)),
            ds_valid: Some(Arc::new(datasets.valid)),
            ds_test: Some(Arc::new(datasets.test)),
            batch_size_train,
            batch_size_eval,
            num_workers,
            num_prefech,
            idcs: Arc::new(Mutex::new(VecDeque::new())),
            current_split: Split::Train,
            fill_thread: None,
            out_receiver: None,
            out_buf: BTreeMap::new(),
            cur_out_idx: 0,
            drop_last,
            drained: false,
            overfit: false,
        })
    }

    pub fn get_ds_arc<S: Into<Split>>(&self, split: S) -> Arc<FftDataset> {
        match split.into() {
            Split::Train => self.ds_train.as_ref().unwrap().clone(),
            Split::Valid => self.ds_valid.as_ref().unwrap().clone(),
            Split::Test => self.ds_test.as_ref().unwrap().clone(),
        }
    }

    pub fn set_ds<S: Into<Split>>(&mut self, split: S, ds: FftDataset) {
        match split.into() {
            Split::Train => self.ds_train.replace(Arc::new(ds)),
            Split::Valid => self.ds_valid.replace(Arc::new(ds)),
            Split::Test => self.ds_test.replace(Arc::new(ds)),
        };
    }

    pub fn dataset_len<S: Into<Split>>(&self, split: S) -> usize {
        let split = split.into();
        let len = self.get_ds_arc(split).len();
        if self.overfit && split != Split::Train {
            // During valid/test only return one batch for each epoch.
            // All batches will be the same and result in same metrics/loss anyways.
            return len.min(self.batch_size_eval);
        }
        len
    }

    pub fn dataloader_len<S: Into<Split> + Copy>(&self, split: S) -> usize {
        let bs = self.batch_size(split);
        if self.drop_last {
            self.dataset_len(split) / bs
        } else {
            (self.dataset_len(split) as f32 / bs as f32).ceil() as usize
        }
    }

    pub fn batch_size<S: Into<Split>>(&self, split: S) -> usize {
        if split.into() == Split::Train {
            self.batch_size_train
        } else {
            self.batch_size_eval
        }
    }

    pub fn set_batch_size<S: Into<Split>>(&mut self, batch_size: usize, split: S) {
        if split.into() == Split::Train {
            self.batch_size_train = batch_size;
        } else {
            self.batch_size_eval = batch_size;
        }
    }

    pub fn start_idx_worker(
        &mut self,
        split: Split,
        epoch_seed: u64,
    ) -> Result<thread::JoinHandle<Result<()>>> {
        let bs = self.batch_size(split);
        if self.num_prefech < bs {
            eprintln!(
                "Warning: Prefetch size ({}) is smaller then batch size ({}).",
                self.num_prefech, bs
            )
        }
        let (out_sender, out_receiver) = sync_channel(self.num_prefech);
        self.out_receiver = Some(out_receiver);
        let ds = self.get_ds_arc(split);
        let (in_sender, in_receiver) = unbounded();
        for idx in self.idcs.lock().unwrap().drain(..) {
            in_sender.send(idx).expect("Could not send index");
        }
        in_sender.send((0, -1)).expect("Could not send index");

        let worker_recievers: Vec<_> = (0..self.num_workers).map(|_| in_receiver.clone()).collect();
        let overfit = self.overfit;
        let handle = thread::spawn(move || -> Result<()> {
            worker_recievers.par_iter().try_for_each(|r| {
                while let Ok((sample_idx, ordering_idx)) = r.recv() {
                    if ordering_idx == -1 {
                        out_sender.send((0, Err(DfDataloaderError::DatasetDrained)))?;
                        return Ok(());
                    }
                    assert!(ordering_idx >= 0);
                    let seed = Some(if overfit {
                        epoch_seed
                    } else {
                        epoch_seed + sample_idx as u64
                    });
                    let sample = match ds.get_sample(sample_idx, seed) {
                        Ok(s) => Ok(s),
                        Err(e) => {
                            eprintln!("Error during get_sample(): {:?}", e);
                            Err(e.into())
                        }
                    };
                    out_sender.send((ordering_idx as usize, sample))?;
                }
                Ok(())
            })
        });
        Ok(handle)
    }

    pub fn start_epoch<S: Into<Split>>(&mut self, split: S, mut epoch_seed: usize) -> Result<()> {
        let split: Split = split.into();
        // Drop fill thread if exits
        if self.fill_thread.is_some() {
            self.join_fill_thread()?;
        }
        // Check whether we need to regenerate. Typically only required for a custom sampling factor.
        for split in Split::iter() {
            if self.get_ds_arc(split).need_generate_keys() {
                let mut ds = match Arc::try_unwrap(
                    match split {
                        Split::Train => self.ds_train.take(),
                        Split::Valid => self.ds_valid.take(),
                        Split::Test => self.ds_test.take(),
                    }
                    .unwrap(),
                ) {
                    Ok(ds) => ds,
                    Err(_) => panic!("Could not regain ownership over dataset"),
                };
                ds.generate_keys()?;
                self.set_ds(split, ds);
            }
        }
        // Output buffers for ordering analogue to self.idcs
        self.out_buf = BTreeMap::new();
        self.cur_out_idx = 0;
        // Prepare for new epoch
        self.current_split = split;
        if self.overfit {
            epoch_seed = 0;
        }
        seed_from_u64(epoch_seed as u64);
        {
            // Recreate indices to index into the dataset and shuffle them
            let n_samples = self.dataset_len(split);
            let sample_idcs: Vec<usize> = if self.overfit {
                println!("Overfitting on one batch.");
                (0..n_samples).cycle().take(n_samples).collect()
            } else {
                let mut tmp = (0..n_samples).collect::<Vec<usize>>();
                tmp.shuffle(&mut thread_rng()?);
                tmp
            };
            // Concatenate an ordering index
            let idcs: VecDeque<(usize, isize)> =
                sample_idcs.into_iter().zip(0..self.dataset_len(split) as isize).collect();
            self.idcs.lock().unwrap().clone_from(&idcs);
        }
        // Start thread to submit dataset jobs for the pool workers
        self.fill_thread = Some(self.start_idx_worker(split, epoch_seed as u64)?);
        self.drained = false;
        Ok(())
    }

    pub fn get_batch<C>(&mut self) -> Result<Option<DsBatch<Complex32>>>
    where
        C: Collate<Complex32>,
    {
        #[cfg(feature = "dataset_timings")]
        let t0 = Instant::now();
        let bs = self.batch_size(self.current_split);
        let mut timings = Vec::with_capacity(bs);
        let mut samples = Vec::with_capacity(bs);
        let target_idx = self.dataset_len(self.current_split).min(self.cur_out_idx + bs);
        if self.cur_out_idx >= self.dataset_len(self.current_split) {
            self.drained = true;
        }
        let mut tries = 0;
        let mut ids = Vec::with_capacity(self.batch_size(self.current_split));
        let reciever = match self.out_receiver.as_ref() {
            None => {
                return Err(DfDataloaderError::ChannelsNotInitializedError);
            }
            Some(r) => r,
        };
        let mut ts0 = Instant::now();
        'outer: while self.cur_out_idx < target_idx {
            // Check if we have some buffered samples
            if let Some(s) = self.out_buf.remove(&self.cur_out_idx) {
                ids.push(self.cur_out_idx);
                samples.push(s);
                let ts1 = Instant::now();
                timings.push((ts1 - ts0).as_secs_f32());
                ts0 = ts1;
                self.cur_out_idx += 1;
            } else {
                // Or check worker threads
                match reciever.recv_timeout(Duration::from_millis(100)) {
                    Err(_e) => {
                        if tries > 1000 {
                            return Err(DfDataloaderError::TimeoutError);
                        }
                        tries += 1;
                        continue 'outer;
                    }
                    Ok((_, Err(DfDataloaderError::DatasetDrained))) => {
                        self.drained = true;
                    }
                    Ok((_, Err(e))) => {
                        return Err(e);
                    }
                    Ok((o_idx, Ok(s))) => {
                        if o_idx == self.cur_out_idx {
                            samples.push(s);
                            let ts1 = Instant::now();
                            timings.push((ts1 - ts0).as_secs_f32());
                            ts0 = ts1;
                            ids.push(o_idx);
                            self.cur_out_idx += 1;
                        } else {
                            assert!(self.out_buf.insert(o_idx, s).is_none());
                        }
                    }
                }
            }
            tries = 0;
        }
        #[cfg(feature = "dataset_timings")]
        let t1 = Instant::now();

        let out = if self.drained && (self.drop_last || samples.is_empty()) {
            assert!(self.cur_out_idx >= target_idx);
            assert!(self.out_buf.is_empty());
            self.join_fill_thread()?;
            None
        } else {
            let mut batch = C::collate(
                samples.as_mut_slice(),
                self.get_ds_arc(self.current_split).max_sample_len(),
            )?;
            batch.ids.extend(ids);
            debug_assert!(batch.batch_size() <= self.batch_size(self.current_split));
            if !self.drained && self.cur_out_idx < target_idx {
                debug_assert_eq!(batch.batch_size(), self.batch_size(self.current_split));
            }
            batch.timings = timings;
            Some(batch)
        };
        #[cfg(feature = "dataset_timings")]
        if log::log_enabled!(log::Level::Trace) {
            let t2 = Instant::now();
            log::trace!(
                "Returning batch in {} ms, (got samples in {} ms)",
                (t2 - t0).as_millis(),
                (t1 - t0).as_millis()
            );
        }
        Ok(out)
    }

    pub fn join_fill_thread(&mut self) -> Result<()> {
        // Drop out_receiver so that parallel iter in fill thread will return
        drop(self.out_receiver.take());
        if let Some(thread) = self.fill_thread.take() {
            if let Err(e) = thread
                .join()
                .map_err(|e| DfDataloaderError::ThreadJoinError(format!("{:?}", e)))?
            {
                match e {
                    DfDataloaderError::SendError(_) => (),
                    // Not expected send error due to out_channel closing
                    e => {
                        eprint!("Error during worker shutdown: {:?}", e);
                        return Err(e);
                    }
                }
            }
        }
        Ok(())
    }
}

pub trait Collate<T: Data> {
    fn collate(samples: &mut [Sample<T>], len: usize) -> Result<DsBatch<T>>;
}
impl Collate<f32> for f32 {
    fn collate(samples: &mut [Sample<f32>], len: usize) -> Result<DsBatch<f32>> {
        let lengths = samples.iter().map(|s| s.speech.len_of(Axis(1))).collect();
        let speech = unpack_pad(|s: &mut Sample<f32>| &mut s.speech, samples, len)?;
        let noisy = unpack_pad(|s: &mut Sample<f32>| &mut s.noisy, samples, len)?;
        let max_freq = samples.iter().map(|s| s.max_freq).collect();
        let snr = samples.iter().map(|s| s.snr).collect();
        let gain = samples.iter().map(|s| s.gain).collect();
        Ok(DsBatch {
            speech,
            noisy,
            feat_erb: None,
            feat_spec: None,
            lengths,
            max_freq,
            snr,
            gain,
            ids: Vec::new(),
            timings: Vec::new(),
        })
    }
}
impl Collate<Complex32> for Complex32 {
    fn collate(samples: &mut [Sample<Complex32>], len: usize) -> Result<DsBatch<Complex32>> {
        let lengths = samples.iter().map(|s| s.speech.len_of(Axis(1))).collect();
        let speech = unpack_pad(|s: &mut Sample<Complex32>| &mut s.speech, samples, len)?;
        let noisy = unpack_pad(|s: &mut Sample<Complex32>| &mut s.noisy, samples, len)?;
        let feat_erb = if samples.first().unwrap().feat_erb.is_some() {
            Some(unpack_pad(
                |s: &mut Sample<Complex32>| s.feat_erb.as_mut().unwrap(),
                samples,
                len,
            )?)
        } else {
            None
        };
        let feat_spec = if samples.first().unwrap().feat_spec.is_some() {
            Some(unpack_pad(
                |s: &mut Sample<Complex32>| s.feat_spec.as_mut().unwrap(),
                samples,
                len,
            )?)
        } else {
            None
        };
        let max_freq = samples.iter().map(|s| s.max_freq).collect();
        let snr = samples.iter().map(|s| s.snr).collect();
        let gain = samples.iter().map(|s| s.gain).collect();
        Ok(DsBatch {
            speech,
            noisy,
            feat_erb,
            feat_spec,
            lengths,
            max_freq,
            snr,
            gain,
            ids: Vec::new(),
            timings: Vec::new(),
        })
    }
}

impl Drop for DataLoader {
    fn drop(&mut self) {
        self.join_fill_thread().unwrap(); // Stop out_receiver and join fill thread
        for split in Split::iter() {
            let ds = match Arc::try_unwrap(
                match split {
                    Split::Train => self.ds_train.take(),
                    Split::Valid => self.ds_valid.take(),
                    Split::Test => self.ds_test.take(),
                }
                .unwrap(),
            ) {
                Ok(ds) => ds,
                Err(_) => panic!("Could not regain ownership over dataset"),
            };
            self.set_ds(split, ds);
        }
    }
}

pub struct DsBatch<T>
where
    T: Data,
{
    pub speech: ArrayD<T>,
    pub noisy: ArrayD<T>,
    pub feat_erb: Option<ArrayD<f32>>,
    pub feat_spec: Option<ArrayD<Complex32>>,
    pub lengths: Array1<usize>,
    pub max_freq: Array1<usize>,
    pub snr: Vec<i8>,
    pub gain: Vec<i8>,
    pub ids: Vec<usize>,
    pub timings: Vec<f32>,
}
impl<T> DsBatch<T>
where
    T: Data,
{
    pub fn batch_size(&self) -> usize {
        self.speech.len_of(Axis(0))
    }
    pub fn sample_len(&self) -> usize {
        self.speech.len_of(Axis(2))
    }
}
impl<T> fmt::Debug for DsBatch<T>
where
    T: Data,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "Dataset Batch with batch_size: '{}, len: '{}', snrs: '{:?}', gain: '{:?}')",
            self.batch_size(),
            self.sample_len(),
            self.snr,
            self.gain
        ))
    }
}

fn unpack_pad<Ts, To, F>(mut f: F, samples: &mut [Sample<Ts>], len: usize) -> Result<ArrayD<To>>
where
    Ts: Data,
    To: Data,
    F: FnMut(&mut Sample<Ts>) -> &mut ArrayD<To>,
{
    let mut out: Vec<ArrayViewMutD<To>> = Vec::with_capacity(samples.len());
    for sample in samples.iter_mut() {
        let x: &mut ArrayD<To> = f(sample);

        let missing = len.saturating_sub(x.len_of(Axis(1)));
        if missing > 0 {
            let mut shape: Vec<usize> = x.shape().into();
            shape[1] = missing;
            let tmp: ArrayD<To> = ArrayD::<To>::zeros(shape);
            x.append(Axis(1), tmp.into_dimensionality()?.view())?;
        }
        out.push(x.view_mut());
    }
    let out: Vec<ArrayViewD<To>> = out.iter().map(|s| s.view()).collect();
    if !out.windows(2).all(|w| w[0].shape() == w[1].shape()) {
        eprintln!("Shapes do not match!");
        for outs in out.iter() {
            eprintln!("  shape: {:?}", outs.shape());
        }
    }
    Ok(ndarray::stack(Axis(0), out.as_slice())?.into_dyn())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::DatasetConfigJson;
    use crate::util::seed_from_u64;

    #[test]
    pub fn test_fft_dataset() -> Result<()> {
        println!("******** Start test_data_loader() ********");
        seed_from_u64(42);
        let fft_size = 960;
        let hop_size = Some(480);
        let nb_erb = Some(32);
        let nb_spec = None;
        let norm_alpha = None;
        let sr = 48000;
        let ds_dir = "../assets/";
        let max_len_s = 1.0;
        let mut cfg = DatasetConfigJson::open("../assets/dataset.cfg")?;
        let builder = DatasetBuilder::new(ds_dir, sr)
            .df_params(fft_size, hop_size, nb_erb, nb_spec, norm_alpha)
            .max_len(max_len_s);
        for dataset_size in [1, 2, 4, 17] {
            for c in cfg.train.iter_mut() {
                c.1 = dataset_size as f32; // Set sampling factor
                assert_eq!(c.sampling_factor(), dataset_size as f32);
            }
            for c in cfg.valid.iter_mut() {
                c.1 = dataset_size as f32; // Set sampling factor
                assert_eq!(c.sampling_factor(), dataset_size as f32);
            }
            for c in cfg.test.iter_mut() {
                c.1 = dataset_size as f32; // Set sampling factor
                assert_eq!(c.sampling_factor(), dataset_size as f32);
            }
            'inner: for batch_size in [1, 2, 16] {
                let ds = Datasets {
                    train: builder
                        .clone()
                        .dataset(cfg.split_config(Split::Train))
                        .build_fft_dataset()?,
                    valid: builder
                        .clone()
                        .dataset(cfg.split_config(Split::Valid))
                        .build_fft_dataset()?,
                    test: builder
                        .clone()
                        .dataset(cfg.split_config(Split::Valid))
                        .build_fft_dataset()?,
                };
                let mut loader = match DataLoader::builder(ds)
                    .num_threads(1)
                    .batch_size(batch_size)
                    .batch_size_eval(1)
                    .build()
                {
                    Ok(loader) => loader,
                    Err(e) => match e {
                        DfDataloaderError::DatasetTooSmall {
                            split: s_,
                            dataset_size: ds_,
                            batch_size: bs_,
                        } => {
                            if dataset_size < batch_size {
                                continue 'inner; // This is expected
                            }
                            return Err(DfDataloaderError::DatasetTooSmall {
                                split: s_,
                                dataset_size: ds_,
                                batch_size: bs_,
                            });
                        }
                        e => return Err(e),
                    },
                };
                for split in Split::iter() {
                    for epoch in 0..2 {
                        println!(
                        "***** Test: Loader with dataset_size {}, batch_size {}, epoch {} ******",
                        dataset_size, batch_size, epoch
                    );
                        loader.start_epoch(split, epoch)?;
                        let mut n_samples = 0;
                        dbg!(dataset_size);
                        while let Some(batch) = loader.get_batch::<Complex32>().unwrap() {
                            n_samples += batch.batch_size();
                            dbg!(n_samples, batch.speech.shape());
                            debug_assert_eq!(
                                batch.speech.len_of(Axis(2)),
                                (max_len_s * (sr / hop_size.unwrap()) as f32).round() as usize
                            );
                            assert!(n_samples <= dataset_size);
                        }
                        dbg!(n_samples, dataset_size);
                    }
                }
            }
        }
        Ok(())
    }
}
