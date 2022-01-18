use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::fmt;
use std::fs;
use std::io::{BufReader, Cursor};
use std::ops::Range;
use std::path::Path;
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crossbeam_channel::unbounded;
use hdf5::{types::VarLenUnicode, File};
use lewton::inside_ogg::OggStreamReader;
use ndarray::{prelude::*, Slice};
use ogg::reading::PacketReader as OggPacketReader;
use rand::prelude::{IteratorRandom, SliceRandom};
use rand::Rng;
use rayon;
use rayon::{current_num_threads, prelude::*};
use realfft::num_traits::Zero;
use serde::Deserialize;
use thiserror::Error;

use crate::{augmentations::*, transforms::*, util::*, Complex32, DFState};

type Result<T> = std::result::Result<T, DfDatasetError>;

#[derive(Error, Debug)]
pub enum DfDatasetError {
    #[error("Dataloading Timeout")]
    TimeoutError,
    #[error("No Hdf5 datasets found")]
    NoDatasetFoundError,
    #[error("No Hdf5 dataset type found")]
    Hdf5DsTypeNotFoundError,
    #[error("{codec:?} codec not supported for file {file:?}")]
    CodecNotSupportedError { codec: Codec, file: String },
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
    #[error("Unsupported during PCM decode: {0}")]
    PcmUnspportedDimension(usize),
    #[error("Wav Reader Error")]
    WarReadError(#[from] crate::wav_utils::WavUtilsError),
    #[error("Input Range ({range:?}) larger than dataset size ({size:?})")]
    PcmRangeToLarge {
        range: Range<usize>,
        size: Vec<usize>,
    },
    #[error("Data Processing Error: {0:?}")]
    DataProcessingError(String),
    #[error("Multithreading Send Error: {0:?}")]
    SendError(String),
    #[error("DF Transforms Error")]
    TransformError(#[from] crate::transforms::TransformError),
    #[error("DF Augmentation Error")]
    AugmentationError(#[from] crate::augmentations::AugmentationError),
    #[error("DF Utils Error")]
    UtilsError(#[from] crate::util::UtilsError),
    #[error("Ndarray Shape Error")]
    NdarrayShapeError(#[from] ndarray::ShapeError),
    #[error("Hdf5 Error")]
    Hdf5Error(#[from] hdf5::Error),
    #[error("Hdf5 Error Detail")]
    Hdf5ErrorDetail { source: hdf5::Error, msg: String },
    #[error("IO Error")]
    IoError(#[from] std::io::Error),
    #[error("Json Decoding Error")]
    JsonDecode(#[from] serde_json::Error),
    #[error("Threadpool Builder Error")]
    ThreadPoolBuildError(#[from] rayon::ThreadPoolBuildError),
    #[error("Vorbis Decode Error")]
    VorbisError(#[from] lewton::VorbisError),
    #[error("Ogg Decode Error")]
    OggReadError(#[from] ogg::reading::OggReadError),
    #[error("Thread Join Error: {0:?}")]
    ThreadJoinError(String),
}

impl<T> From<std::sync::mpsc::SendError<T>> for DfDatasetError {
    fn from(error: std::sync::mpsc::SendError<T>) -> Self {
        DfDatasetError::SendError(error.to_string())
    }
}

type Signal = Array2<f32>;

fn one() -> f32 {
    1.
}
#[derive(Deserialize, Debug, Clone)]
pub struct Hdf5Cfg(
    pub String,                                              // file name
    #[serde(default = "one")] pub f32,                       // dataset sampling factor
    #[serde(default = "Option::default")] pub Option<usize>, // fallback sampling rate
    #[serde(default = "Option::default")] pub Option<usize>, // fallback max freq
);
impl Hdf5Cfg {
    pub fn filename(&self) -> &str {
        self.0.as_str()
    }
    pub fn sampling_factor(&self) -> f32 {
        self.1
    }
    pub fn fallback_sr(&self) -> Option<usize> {
        self.2
    }
    pub fn fallback_max_freq(&self) -> Option<usize> {
        self.3
    }
}
#[derive(Deserialize, Debug)]
pub struct DatasetConfigJson {
    pub train: Vec<Hdf5Cfg>,
    pub valid: Vec<Hdf5Cfg>,
    pub test: Vec<Hdf5Cfg>,
}
impl DatasetConfigJson {
    pub fn open(path: &str) -> Result<Self> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        let cfg = serde_json::from_reader(reader)?;
        Ok(cfg)
    }
    pub fn split_config(&self, split: Split) -> DatasetSplitConfig {
        match split {
            Split::Train => DatasetSplitConfig {
                hdf5s: self.train.clone(),
                split: Split::Train,
            },
            Split::Valid => DatasetSplitConfig {
                hdf5s: self.valid.clone(),
                split: Split::Valid,
            },
            Split::Test => DatasetSplitConfig {
                hdf5s: self.test.clone(),
                split: Split::Test,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct DatasetSplitConfig {
    pub hdf5s: Vec<Hdf5Cfg>,
    split: Split,
}

impl DatasetSplitConfig {
    pub fn extend(&mut self, other: DatasetSplitConfig) {
        assert_eq!(self.split, other.split);
        self.hdf5s.extend(other.hdf5s);
    }
    pub fn is_empty(&self) -> bool {
        self.hdf5s.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = &Hdf5Cfg> {
        self.hdf5s.iter()
    }
}

pub struct Datasets {
    pub train: FftDataset,
    pub valid: FftDataset,
    pub test: FftDataset,
}

impl Datasets {
    fn get<S: Into<Split>>(&self, split: S) -> &FftDataset {
        match split.into() {
            Split::Train => &self.train,
            Split::Valid => &self.valid,
            Split::Test => &self.test,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum Split {
    Train = 0,
    Valid = 1,
    Test = 2,
}

impl Split {
    pub fn iter() -> impl Iterator<Item = Split> {
        [Split::Train, Split::Valid, Split::Test].iter().cloned()
    }
}

impl fmt::Display for Split {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Train => write!(f, "train"),
            Self::Valid => write!(f, "valid"),
            Self::Test => write!(f, "test"),
        }
    }
}

impl From<&str> for Split {
    fn from(split: &str) -> Self {
        match split {
            "train" => Split::Train,
            "valid" => Split::Valid,
            "test" => Split::Test,
            s => panic!("Split '{}' does not exist.", s),
        }
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
    _drop_last: Option<bool>,
    _overfit: Option<bool>,
}

impl DataLoaderBuilder {
    pub fn new(ds: Datasets) -> Self {
        DataLoaderBuilder {
            _ds: Some(ds),
            _batch_size: None,
            _batch_size_eval: None,
            _prefetch: None,
            _num_threads: None,
            _drop_last: None,
            _overfit: None,
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
    pub fn overfit(mut self, overfit: bool) -> Self {
        self._overfit = Some(overfit);
        self
    }
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self._drop_last = Some(drop_last);
        self
    }
    fn check_dataset_size(&self, bs_train: usize) -> Result<()> {
        for split in [Split::Train, Split::Valid, Split::Test] {
            let batch_size = match split {
                Split::Train => bs_train,
                _ => self._batch_size_eval.unwrap_or(bs_train),
            };
            let dataset_size = self._ds.as_ref().unwrap().get(split).len();
            if dataset_size < batch_size {
                return Err(DfDatasetError::DatasetTooSmall {
                    split,
                    dataset_size,
                    batch_size,
                });
            }
        }
        Ok(())
    }
    pub fn build(self) -> Result<DataLoader> {
        let bs_train = self._batch_size.unwrap_or(1);
        self.check_dataset_size(bs_train)?;
        let prefetch = self._prefetch.unwrap_or(bs_train * self._num_threads.unwrap_or(4) * 2);
        let mut loader = DataLoader::new(
            self._ds.unwrap(),
            bs_train,
            self._batch_size_eval,
            prefetch,
            self._num_threads,
            self._drop_last.unwrap_or(false),
        )?;
        loader.overfit = self._overfit.unwrap_or(false);
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
        let mut poolbuilder = rayon::ThreadPoolBuilder::new();
        let num_workers = num_threads.unwrap_or_else(current_num_threads);
        poolbuilder = poolbuilder.num_threads(num_workers);
        match poolbuilder
            .thread_name(|idx| format!("DataLoader Worker {}", idx))
            .build_global()
        {
            Ok(()) => (),
            Err(e) => {
                if e.to_string() != "The global thread pool has already been initialized." {
                    return Err(e.into());
                }
                // else: already initialized, do not complain.
            }
        };
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
        self.get_ds_arc(split).len()
    }

    pub fn len_of<S: Into<Split>>(&self, split: S) -> usize {
        let split = split.into();
        let bs = self.batch_size(&split);
        if self.drop_last {
            self.dataset_len(split) / bs
        } else {
            (self.dataset_len(split) as f32 / bs as f32).ceil() as usize
        }
    }

    pub fn batch_size(&self, split: &Split) -> usize {
        if split == &Split::Train {
            self.batch_size_train
        } else {
            self.batch_size_eval
        }
    }

    pub fn start_idx_worker(
        &mut self,
        split: Split,
        epoch_seed: u64,
    ) -> Result<thread::JoinHandle<Result<()>>> {
        let bs = self.batch_size(&split);
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
        let handle = thread::spawn(move || -> Result<()> {
            worker_recievers.par_iter().try_for_each(|r| {
                while let Ok((sample_idx, ordering_idx)) = r.recv() {
                    if ordering_idx == -1 {
                        out_sender.send((0, Err(DfDatasetError::DatasetDrained)))?;
                        return Ok(());
                    }
                    assert!(ordering_idx >= 0);
                    let sample = ds.get_sample(sample_idx, Some(epoch_seed));
                    out_sender.send((ordering_idx as usize, sample))?;
                }
                Ok(())
            })
        });
        Ok(handle)
    }

    pub fn start_epoch<S: Into<Split>>(&mut self, split: S, seed: usize) -> Result<()> {
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
        seed_from_u64(seed as u64);
        {
            // Recreate indices to index into the dataset and shuffle them
            let sample_idcs: Vec<usize> = if self.overfit {
                println!("Overfitting on one batch.");
                let bs = self.batch_size(&split);
                (0..bs).cycle().take(self.dataset_len(split)).collect()
            } else {
                let mut tmp = (0..self.dataset_len(split)).collect::<Vec<usize>>();
                tmp.shuffle(&mut thread_rng()?);
                tmp
            };
            // Concatenate an ordering index
            let idcs: VecDeque<(usize, isize)> =
                sample_idcs.into_iter().zip(0..self.dataset_len(split) as isize).collect();
            self.idcs.lock().unwrap().clone_from(&idcs);
        }
        // Start thread to submit dataset jobs for the pool workers
        self.fill_thread = Some(self.start_idx_worker(split, seed as u64)?);
        self.drained = false;
        Ok(())
    }

    pub fn get_batch<C>(&mut self) -> Result<Option<DsBatch<Complex32>>>
    where
        C: Collate<Complex32>,
    {
        let bs = self.batch_size(&self.current_split);
        let mut samples = Vec::with_capacity(bs);
        let target_idx = self.dataset_len(self.current_split).min(self.cur_out_idx + bs);
        if self.cur_out_idx >= self.dataset_len(self.current_split) {
            self.drained = true;
        }
        let mut tries = 0;
        let mut ids = Vec::with_capacity(self.batch_size(&self.current_split));
        let reciever = match self.out_receiver.as_ref() {
            None => {
                return Err(DfDatasetError::ChannelsNotInitializedError);
            }
            Some(r) => r,
        };
        'outer: while self.cur_out_idx < target_idx {
            // Check if we have some buffered samples
            if let Some(s) = self.out_buf.remove(&self.cur_out_idx) {
                ids.push(self.cur_out_idx);
                samples.push(s);
                self.cur_out_idx += 1;
            } else {
                // Or check worker threads
                match reciever.recv_timeout(Duration::from_millis(100)) {
                    Err(_e) => {
                        if tries > 1000 {
                            return Err(DfDatasetError::TimeoutError);
                        }
                        tries += 1;
                        continue 'outer;
                    }
                    Ok((_, Err(DfDatasetError::DatasetDrained))) => {
                        self.drained = true;
                    }
                    Ok((_, Err(e))) => {
                        return Err(e);
                    }
                    Ok((o_idx, Ok(s))) => {
                        if o_idx == self.cur_out_idx {
                            samples.push(s);
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
            debug_assert!(batch.batch_size() <= self.batch_size(&self.current_split));
            if !self.drained && self.cur_out_idx < target_idx {
                debug_assert_eq!(batch.batch_size(), self.batch_size(&self.current_split));
            }
            Some(batch)
        };
        Ok(out)
    }

    pub fn join_fill_thread(&mut self) -> Result<()> {
        // Drop out_receiver so that parallel iter in fill thread will return
        drop(self.out_receiver.take());
        if let Some(thread) = self.fill_thread.take() {
            if let Err(e) =
                thread.join().map_err(|e| DfDatasetError::ThreadJoinError(format!("{:?}", e)))?
            {
                match e {
                    DfDatasetError::SendError(_) => (),
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
        let noise = unpack_pad(|s: &mut Sample<f32>| &mut s.noise, samples, len)?;
        let noisy = unpack_pad(|s: &mut Sample<f32>| &mut s.noisy, samples, len)?;
        let max_freq = samples.iter().map(|s| s.max_freq).collect();
        let snr = samples.iter().map(|s| s.snr).collect();
        let gain = samples.iter().map(|s| s.gain).collect();
        let atten = samples.iter().map(|s| s.attenuation.unwrap_or(0)).collect();
        Ok(DsBatch {
            speech,
            noise,
            noisy,
            feat_erb: None,
            feat_spec: None,
            lengths,
            max_freq,
            snr,
            gain,
            atten,
            ids: Vec::new(),
        })
    }
}
impl Collate<Complex32> for Complex32 {
    fn collate(samples: &mut [Sample<Complex32>], len: usize) -> Result<DsBatch<Complex32>> {
        let lengths = samples.iter().map(|s| s.speech.len_of(Axis(1))).collect();
        let speech = unpack_pad(|s: &mut Sample<Complex32>| &mut s.speech, samples, len)?;
        let noise = unpack_pad(|s: &mut Sample<Complex32>| &mut s.noise, samples, len)?;
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
        let atten = samples.iter().map(|s| s.attenuation.unwrap_or(0)).collect();
        Ok(DsBatch {
            speech,
            noise,
            noisy,
            feat_erb,
            feat_spec,
            lengths,
            max_freq,
            snr,
            gain,
            atten,
            ids: Vec::new(),
        })
    }
}

impl Drop for DataLoader {
    fn drop(&mut self) {
        self.join_fill_thread().unwrap(); // Stop out_receiver and join fill thread
    }
}

pub struct DsBatch<T>
where
    T: Data,
{
    pub speech: ArrayD<T>,
    pub noise: ArrayD<T>,
    pub noisy: ArrayD<T>,
    pub feat_erb: Option<ArrayD<f32>>,
    pub feat_spec: Option<ArrayD<Complex32>>,
    pub lengths: Array1<usize>,
    pub max_freq: Array1<usize>,
    pub snr: Vec<i8>,
    pub gain: Vec<i8>,
    pub atten: Vec<u8>, // attenuation limit in dB; 0 stands for no limit
    pub ids: Vec<usize>,
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

pub trait Data: Sized + Clone + Default + Send + Sync + Zero + 'static {}
impl Data for f32 {}
impl Data for Complex32 {}

pub enum SampleType {
    TimeDomain,
    FreqDomain,
}
pub struct Sample<T>
where
    T: Data,
{
    pub speech: ArrayD<T>,
    pub noise: ArrayD<T>,
    pub noisy: ArrayD<T>,
    pub feat_erb: Option<ArrayD<f32>>,
    pub feat_spec: Option<ArrayD<Complex32>>,
    pub max_freq: usize,
    pub snr: i8,
    pub gain: i8,
    pub attenuation: Option<u8>,
    pub idx: usize,
}
impl Sample<f32> {
    fn sample_type(&self) -> SampleType {
        SampleType::TimeDomain
    }
    fn get_speech_view(&self) -> Result<ArrayView2<f32>> {
        Ok(self.speech.view().into_dimensionality()?)
    }
    fn get_noise_view(&self) -> Result<ArrayView2<f32>> {
        Ok(self.noise.view().into_dimensionality()?)
    }
    fn get_noisy_view(&self) -> Result<ArrayView2<f32>> {
        Ok(self.noisy.view().into_dimensionality()?)
    }
    fn dim(&self) -> usize {
        2
    }
}
impl Sample<Complex32> {
    fn sample_type(&self) -> SampleType {
        SampleType::FreqDomain
    }
    fn get_speech_view(&self) -> Result<ArrayView3<Complex32>> {
        Ok(self.speech.view().into_dimensionality()?)
    }
    fn get_noise_view(&self) -> Result<ArrayView3<Complex32>> {
        Ok(self.noise.view().into_dimensionality()?)
    }
    fn get_noisy_view(&self) -> Result<ArrayView3<Complex32>> {
        Ok(self.noisy.view().into_dimensionality()?)
    }
    fn dim(&self) -> usize {
        3
    }
}

impl<T> fmt::Debug for Sample<T>
where
    T: Data,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "Dataset Sample {} with len: '{}', snr: '{}', gain: '{}')",
            self.idx,
            self.speech.shape().last().unwrap(),
            self.snr,
            self.gain
        ))
    }
}

pub trait Dataset<T>
where
    T: Data,
{
    fn get_sample(&self, idx: usize, seed: Option<u64>) -> Result<Sample<T>>;
    fn sr(&self) -> usize;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn max_sample_len(&self) -> usize;
    fn set_seed(&mut self, seed: u64);
    fn need_generate_keys(&self) -> bool;
    fn generate_keys(&mut self) -> Result<()>;
}

#[derive(Clone)]
pub struct DatasetBuilder<'a> {
    ds_dir: &'a str,
    sr: usize,
    fft_size: Option<usize>,
    datasets: Option<DatasetSplitConfig>,
    max_len_s: Option<f32>,
    hop_size: Option<usize>,
    nb_erb: Option<usize>,
    nb_spec: Option<usize>,
    norm_alpha: Option<f32>,
    p_atten_lim: Option<f32>,
    p_reverb: Option<f32>,
    p_fill_speech: Option<f32>,
    seed: Option<u64>,
    min_nb_freqs: Option<usize>,
}
impl<'a> DatasetBuilder<'a> {
    pub fn new(ds_dir: &'a str, sr: usize) -> Self {
        DatasetBuilder {
            ds_dir,
            sr,
            datasets: None,
            max_len_s: None,
            fft_size: None,
            hop_size: None,
            nb_erb: None,
            nb_spec: None,
            norm_alpha: None,
            p_atten_lim: None,
            p_reverb: None,
            p_fill_speech: None,
            seed: None,
            min_nb_freqs: None,
        }
    }
    pub fn build_fft_dataset(self) -> Result<FftDataset> {
        if self.datasets.is_none() {
            panic!("No datasets provided")
        }
        let ds = self.clone().build_td_dataset()?;
        if self.fft_size.is_none() {
            panic!("No fft size provided when building FFT dataset.")
        }
        let fft_size = self.fft_size.unwrap();
        let hop_size = self.hop_size.unwrap_or(fft_size / 2);
        let nb_erb = self.nb_erb.unwrap_or(32);
        if let Some(b) = self.nb_spec {
            let nfreqs = fft_size / 2 + 1;
            if b > nfreqs {
                let msg = format!("Number of spectrogram bins ({}) is larger then number of available frequency bins ({})", b, nfreqs);
                return Err(DfDatasetError::DataProcessingError(msg));
            }
        }
        Ok(FftDataset {
            ds,
            fft_size,
            hop_size,
            nb_erb: Some(nb_erb),
            nb_spec: self.nb_spec,
            norm_alpha: self.norm_alpha,
            min_nb_freqs: self.min_nb_freqs,
        })
    }
    pub fn build_td_dataset(self) -> Result<TdDataset> {
        let mut datasets = match self.datasets {
            None => panic!("No datasets provided"),
            Some(ds) => ds,
        };
        // TODO: Return all sample by default and not only 10 seconds
        let max_samples: usize = (self.max_len_s.unwrap_or(10.) * self.sr as f32).round() as usize;
        // Get dataset handles and keys. Each key is a unique String.
        let mut hdf5_handles = Vec::new();
        let mut ds_keys = Vec::new();
        let mut config: Vec<Hdf5Cfg> = Vec::new();
        let mut has_rirs = false;
        for (i, cfg) in datasets.hdf5s.drain(..).enumerate() {
            let name = cfg.filename();
            let path = Path::new(self.ds_dir).join(name);
            if (!path.is_file()) && path.read_link().is_err() {
                eprintln!("Dataset {:?} not found. Skipping.", path);
                continue;
            }
            let ds = Hdf5Dataset::new(path.to_str().unwrap())?;
            if ds.dstype == DsType::RIR {
                has_rirs = true
            }
            ds_keys.push((ds.dstype, i, ds.keys()?));
            hdf5_handles.push(ds);
            config.push(cfg);
        }
        if hdf5_handles.is_empty() {
            return Err(DfDatasetError::NoDatasetFoundError);
        }
        let snrs = vec![-5, 0, 5, 10, 20, 40];
        let gains = vec![-6, 0, 6];
        let attenuation_range = (6, 40);
        let p_atten_lim = self.p_atten_lim.unwrap_or(0.);
        let p_fill_speech = self.p_fill_speech.unwrap_or(0.);
        let sp_transforms = Compose::new(vec![
            Box::new(RandRemoveDc::default_with_prob(0.25)),
            Box::new(RandLFilt::default_with_prob(0.25)),
            Box::new(RandEQ::default_with_prob(0.25).with_sr(self.sr)),
            Box::new(RandResample::default_with_prob(0.1).with_sr(self.sr)),
        ]);
        let ns_transforms = sp_transforms.clone();
        let p_reverb = self.p_reverb.unwrap_or(0.);
        if p_reverb > 0. && !has_rirs {
            eprintln!("Warning: Reverb augmentation enabled but no RIRs provided!");
        }
        let reverb = RandReverbSim::new(p_reverb, self.sr);
        let seed = self.seed.unwrap_or(0);
        let mut ds = TdDataset {
            config,
            hdf5_handles,
            max_samples,
            sr: self.sr,
            ds_keys,
            ds_split: datasets.split,
            sp_keys: Vec::new(),
            ns_keys: Vec::new(),
            rir_keys: Vec::new(),
            snrs,
            gains,
            attenuation_range,
            p_fill_speech,
            p_atten_lim,
            sp_transforms,
            ns_transforms,
            reverb,
            seed,
        };
        // Generate inital speech/noise/rir dataset keys. May be changed at the start of each epoch.
        ds.generate_keys()?;
        Ok(ds)
    }
    pub fn dataset(mut self, datasets: DatasetSplitConfig) -> Self {
        let has_ds = self.datasets.is_some();
        if has_ds {
            self.datasets.as_mut().unwrap().extend(datasets)
        } else {
            self.datasets = Some(datasets)
        }
        self
    }
    pub fn max_len(mut self, max_len_s: f32) -> Self {
        self.max_len_s = Some(max_len_s);
        self
    }
    pub fn df_params(
        mut self,
        fft_size: usize,
        hop_size: Option<usize>,
        nb_erb: Option<usize>,
        nb_spec: Option<usize>,
        norm_alpha: Option<f32>,
    ) -> Self {
        self.fft_size = Some(fft_size);
        self.hop_size = hop_size;
        self.nb_erb = nb_erb;
        self.nb_spec = nb_spec;
        self.norm_alpha = norm_alpha;
        self
    }
    pub fn prob_atten_lim(mut self, p_atten_lim: f32) -> Self {
        assert!((0. ..=1.).contains(&p_atten_lim));
        self.p_atten_lim = Some(p_atten_lim);
        self
    }
    pub fn prob_reverberation(mut self, p_reverb: f32) -> Self {
        assert!((0. ..=1.).contains(&p_reverb));
        self.p_reverb = Some(p_reverb);
        self
    }
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    pub fn p_sample_full_speech(mut self, p_full: f32) -> Self {
        self.p_fill_speech = Some(p_full);
        self
    }
    pub fn min_nb_erb_freqs(mut self, n: usize) -> Self {
        self.min_nb_freqs = Some(n);
        self
    }
}

pub struct FftDataset {
    ds: TdDataset,
    fft_size: usize,
    hop_size: usize,
    nb_erb: Option<usize>,
    nb_spec: Option<usize>,
    norm_alpha: Option<f32>,
    min_nb_freqs: Option<usize>,
}
impl Dataset<Complex32> for FftDataset {
    fn get_sample(&self, idx: usize, seed: Option<u64>) -> Result<Sample<Complex32>> {
        let sample: Sample<f32> = self.ds.get_sample(idx, seed)?;
        let nb_erb = self.nb_erb.unwrap_or(1);
        let mut state = DFState::new(self.sr(), self.fft_size, self.hop_size, nb_erb, 1);
        let speech = stft(sample.get_speech_view()?, &mut state, false);
        let noise = stft(sample.get_noise_view()?, &mut state, true);
        let noisy = stft(sample.get_noisy_view()?, &mut state, true);
        let erb = if let Some(_b) = self.nb_erb {
            let mut erb = erb(&noisy.view(), true, &state.erb)?;
            if let Some(alpha) = self.norm_alpha {
                erb_norm(&mut erb.view_mut(), None, alpha)?;
            }
            Some(erb.into_dyn())
        } else {
            None
        };
        let spec = if let Some(b) = self.nb_spec {
            let mut spec = noisy.slice_axis(Axis(2), Slice::from(..b)).into_owned();
            if let Some(alpha) = self.norm_alpha {
                unit_norm(&mut spec.view_mut(), None, alpha)?;
            }
            Some(spec.into_dyn())
        } else {
            None
        };
        Ok(Sample {
            speech: speech.into_dyn(),
            noise: noise.into_dyn(),
            noisy: noisy.into_dyn(),
            feat_erb: erb,
            feat_spec: spec,
            max_freq: sample.max_freq,
            gain: sample.gain,
            snr: sample.snr,
            attenuation: sample.attenuation,
            idx: sample.idx,
        })
    }

    fn len(&self) -> usize {
        self.ds.sp_keys.len()
    }

    fn sr(&self) -> usize {
        self.ds.sr
    }

    fn max_sample_len(&self) -> usize {
        self.ds.max_samples / self.hop_size
    }

    fn set_seed(&mut self, seed: u64) {
        self.ds.set_seed(seed)
    }

    fn need_generate_keys(&self) -> bool {
        self.ds.need_generate_keys()
    }

    fn generate_keys(&mut self) -> Result<()> {
        self.ds.generate_keys()
    }
}

pub struct TdDataset {
    config: Vec<Hdf5Cfg>,                       // config
    hdf5_handles: Vec<Hdf5Dataset>,             // Handles to access opened Hdf5 datasets
    max_samples: usize,                         // Number of samples in time domain
    sr: usize,                                  // Sampling rate
    ds_keys: Vec<(DsType, usize, Vec<String>)>, // Dataset keys as a vector of [DS Type, hdf5 index, Vec<str keys>].
    ds_split: Split,                            // Train/Valid/Test
    sp_keys: Vec<(usize, String)>, // Pair of hdf5 index and dataset keys. Will be generated at each epoch start
    ns_keys: Vec<(usize, String)>,
    rir_keys: Vec<(usize, String)>,
    snrs: Vec<i8>,               // in dB; SNR to sample from
    gains: Vec<i8>,              // in dB; Speech (loudness) to sample from
    attenuation_range: (u8, u8), // in dB; Return a target sample containing noise for attenuation limited algorithm
    p_atten_lim: f32,            // Probability for containing noise in target
    p_fill_speech: f32, // Probability to completely fill the speech signal to `max_samples` with a different speech sample
    sp_transforms: Compose, // Transforms to augment speech samples
    ns_transforms: Compose, // Transforms to augment noise samples
    reverb: RandReverbSim, // Separate reverb transform that may be applied to both speech and noise
    seed: u64,
}

impl TdDataset {
    fn _read_from_hdf5(
        &self,
        key: &str,
        idx: usize,
        max_len: Option<usize>,
    ) -> Result<Array2<f32>> {
        let h = &self.hdf5_handles[idx];
        let sr = h.sr.unwrap_or_else(|| self.config[idx].fallback_sr().unwrap_or(self.sr));
        let slc = if let Some(l) = max_len {
            let l_sr = l * sr / self.sr;
            let sample_len = h.sample_len(key)?;
            let max_len = sample_len.min(l_sr);
            let s = sample_len as i64 - max_len as i64;
            if s > 0 {
                let s = thread_rng()?.gen_range(0..(s as usize));
                Some(s..s + l_sr)
            } else {
                None
            }
        } else {
            None
        };
        let mut x = if let Some(slc) = slc {
            h.read_slc(key, slc)?
        } else {
            h.read(key)?
        };
        if sr != self.sr {
            x = resample(&x, sr, self.sr, None)?;
            if let Some(l) = max_len {
                if x.len_of(Axis(1)) > l {
                    x.slice_axis_inplace(Axis(1), Slice::from(0..l))
                }
            }
            return Ok(x);
        }
        Ok(x)
    }

    fn read(&self, idx: usize, key: &str) -> Result<Array2<f32>> {
        let x = self._read_from_hdf5(key, idx, None)?;
        Ok(x)
    }

    fn read_max_len(&self, idx: usize, key: &str) -> Result<Array2<f32>> {
        let x = match self._read_from_hdf5(key, idx, Some(self.max_samples)) {
            Err(e) => {
                eprintln!("Error during speech reading get_data(): {:?}", e);
                if e.to_string().contains("inflate") {
                    // Get a different speech then
                    let idx = thread_rng()?.gen_range(0..self.len());
                    let (sp_idx, sp_key) = &self.sp_keys[idx];
                    eprintln!(
                        "Returning a different speech sample from {}",
                        self.ds_name(*sp_idx)
                    );
                    self.read_max_len(*sp_idx, sp_key)?
                } else {
                    return Err(e);
                }
            }
            Ok(s) => s,
        };
        debug_assert!(x.len_of(Axis(1)) <= self.max_samples);
        Ok(x)
    }

    fn max_freq(&self, idx: usize) -> Result<usize> {
        let ds = &self.hdf5_handles[idx];
        let max_freq = match ds.max_freq {
            Some(x) if x > 0 => x,
            _ => self.config[idx].fallback_max_freq().unwrap_or_else(|| {
                ds.sr.unwrap_or_else(|| self.config[idx].fallback_sr().unwrap_or(self.sr)) / 2
            }),
        };
        Ok(max_freq)
    }

    fn ds_name(&self, idx: usize) -> String {
        self.hdf5_handles[idx].name()
    }
}

impl Dataset<f32> for TdDataset {
    fn get_sample(&self, idx: usize, seed: Option<u64>) -> Result<Sample<f32>> {
        seed_from_u64(idx as u64 + self.seed + seed.unwrap_or(0));
        let mut rng = thread_rng()?;
        let (sp_idx, sp_key) = &self.sp_keys[idx];
        let mut speech = self.read_max_len(*sp_idx, sp_key)?;
        self.sp_transforms.transform(&mut speech)?;
        let mut max_freq = self.max_freq(*sp_idx)?;
        while speech.len_of(Axis(1)) < self.max_sample_len()
            && self.p_fill_speech > 0.0
            && self.p_fill_speech > rng.gen_range(0f32..1f32)
        {
            // If too short, maybe sample another speech sample
            let (sp_idx, sp_key) = &self.sp_keys.choose(&mut rng).unwrap();
            let mut another_speech = self.read_max_len(*sp_idx, sp_key)?;
            self.sp_transforms.transform(&mut another_speech)?;
            speech.append(Axis(1), another_speech.view())?;
            max_freq = max_freq.min(self.max_freq(*sp_idx)?);
        }
        if speech.len_of(Axis(1)) > self.max_sample_len() {
            speech.slice_axis_inplace(Axis(1), Slice::from(..self.max_samples));
        }
        // Apply low pass to the noise as well
        let noise_low_pass = if max_freq < self.sr / 2 {
            Some(LpParam {
                cut_off: max_freq,
                sr: self.sr,
            })
        } else {
            None
        };
        let mut ch = speech.len_of(Axis(0));
        let mut len = speech.len_of(Axis(1));
        if len > self.max_samples {
            speech.slice_axis_inplace(Axis(1), Slice::from(..self.max_samples));
            len = speech.len_of(Axis(1));
        }
        if ch > 1 {
            speech.slice_axis_inplace(Axis(0), Slice::from(..1));
            ch = 1;
        }
        // Sample 2-5 noises and augment each
        let n_noises = rng.gen_range(2..6);
        let ns_ids = self.ns_keys.iter().choose_multiple(&mut rng, n_noises);
        let mut noises = Vec::with_capacity(n_noises);
        let mut noise_gains = Vec::with_capacity(n_noises);
        for (ns_idx, ns_key) in &ns_ids {
            let mut ns = match self.read_max_len(*ns_idx, ns_key) {
                Err(e) => {
                    eprintln!("Error during noise reading get_data(): {}", e);
                    continue;
                }
                Ok(n) => n,
            };
            if ns.len_of(Axis(1)) < 10 {
                continue;
            }
            self.ns_transforms.transform(&mut ns)?;
            if ns.len_of(Axis(1)) > self.max_samples {
                ns.slice_axis_inplace(Axis(1), Slice::from(..self.max_samples));
            }
            noises.push(ns);
            noise_gains.push(self.gains.choose(&mut rng).unwrap());
        }
        let noise_gains_f32: Vec<f32> = noise_gains.iter().map(|x| **x as f32).collect();
        // Sample SNR and gain
        let &snr = self.snrs.choose(&mut rng).unwrap();
        let &gain = self.gains.choose(&mut rng).unwrap();
        // Sample attenuation limiting during training
        let atten = if self.p_atten_lim > 0. && self.p_atten_lim > rng.gen_range(0f32..1f32) {
            Some(rng.gen_range(self.attenuation_range.0..self.attenuation_range.1))
        } else {
            None
        };
        // Truncate to speech len, combine noises and mix to noisy
        let mut noise = combine_noises(ch, len, &mut noises, Some(noise_gains_f32.as_slice()))?;
        // Apply reverberation using a randomly sampled RIR
        let speech_rev = if !self.rir_keys.is_empty() {
            self.reverb.transform(&mut speech, &mut noise, || {
                let (rir_idx, rir_key) = self.rir_keys.iter().choose(&mut rng).unwrap();
                let rir = self.read(*rir_idx, rir_key)?;
                Ok(rir)
            })?
        } else {
            None
        };
        let (speech, noise, noisy) = mix_audio_signal(
            speech,
            speech_rev,
            noise,
            snr as f32,
            gain as f32,
            atten.map(|a| a as f32),
            noise_low_pass,
        )?;
        Ok(Sample {
            speech: speech.into_dyn(),
            noise: noise.into_dyn(),
            noisy: noisy.into_dyn(),
            feat_erb: None,
            feat_spec: None,
            max_freq,
            snr,
            gain,
            attenuation: atten,
            idx,
        })
    }

    fn len(&self) -> usize {
        self.sp_keys.len()
    }

    fn sr(&self) -> usize {
        self.sr
    }

    fn max_sample_len(&self) -> usize {
        self.max_samples
    }

    fn set_seed(&mut self, seed: u64) {
        self.seed = seed
    }

    fn need_generate_keys(&self) -> bool {
        if self.sp_keys.is_empty() {
            return true;
        }
        if self.ds_split == Split::Train {
            for (_, hdf5_idx, _) in self.ds_keys.iter() {
                let f = self.config[*hdf5_idx].sampling_factor();
                dbg!(f);
                // if not a natural number, then we need to regenerate.
                if f != f.round() {
                    return true;
                }
            }
        }
        false
    }

    fn generate_keys(&mut self) -> Result<()> {
        self.sp_keys.clear();
        self.ns_keys.clear();
        self.rir_keys.clear();

        for (dstype, hdf5_idx, keys) in self.ds_keys.iter() {
            let len = self.hdf5_handles[*hdf5_idx].len();
            let n_samples =
                (self.config[*hdf5_idx].sampling_factor() * len as f32).round() as usize;
            let mut keys = keys.clone();
            if self.ds_split == Split::Train {
                keys.shuffle(&mut thread_rng()?)
            }
            let keys: Vec<(usize, String)> =
                keys.iter().cycle().take(n_samples).map(|k| (*hdf5_idx, k.clone())).collect();
            match dstype {
                DsType::Speech => self.sp_keys.extend(keys),
                DsType::Noise => self.ns_keys.extend(keys),
                DsType::RIR => self.rir_keys.extend(keys),
            }
        }
        dbg!(self.sp_keys.len());
        dbg!(self.ns_keys.len());
        dbg!(self.rir_keys.len());
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Clone, Copy)]
pub enum DsType {
    Speech = 0,
    Noise = 1,
    RIR = 2,
}
impl fmt::Display for DsType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
#[derive(Debug, Eq, PartialEq)]
pub enum Codec {
    PCM = 0,
    Vorbis = 1,
}
impl Default for &Codec {
    fn default() -> Self {
        &Codec::PCM
    }
}
impl Default for Codec {
    fn default() -> Self {
        Codec::PCM
    }
}
#[derive(Debug)]
pub enum DType {
    I16 = 0,
    F32 = 1,
}

#[derive(Debug)]
pub struct Hdf5Dataset {
    file: File,
    dstype: DsType,
    sr: Option<usize>,
    codec: Option<Codec>,
    max_freq: Option<usize>,
    dtype: Option<DType>,
}

fn get_dstype(file: &File) -> Option<DsType> {
    for g in file.member_names().unwrap_or_default() {
        match g.to_lowercase().as_str() {
            "speech" => return Some(DsType::Speech),
            "noise" => return Some(DsType::Noise),
            "rir" => return Some(DsType::RIR),
            _ => (),
        };
    }
    None
}

impl Hdf5Dataset {
    fn new(path: &str) -> Result<Self> {
        let file = File::open(path).map_err(move |e: hdf5::Error| -> DfDatasetError {
            DfDatasetError::Hdf5ErrorDetail {
                source: e,
                msg: format!("Error during File::open of dataset {}", path),
            }
        })?;
        match get_dstype(&file) {
            None => Err(DfDatasetError::Hdf5DsTypeNotFoundError),
            Some(dstype) => {
                let sr = match file.attr("sr") {
                    Err(_e) => None,
                    Ok(attr) => Some(attr.read_scalar::<usize>().unwrap()),
                };
                let max_freq = match file.attr("max_freq") {
                    Err(_e) => None,
                    Ok(attr) => Some(attr.read_scalar::<usize>().unwrap()),
                };
                let codec = match file.attr("codec") {
                    Err(_e) => None,
                    Ok(attr) => match attr.read_scalar::<VarLenUnicode>().unwrap().as_str() {
                        "pcm" => Some(Codec::PCM),
                        "vorbis" => Some(Codec::Vorbis),
                        _ => None,
                    },
                };
                let dtype = match file.attr("dtype") {
                    Err(_e) => None,
                    Ok(attr) => match attr.read_scalar::<VarLenUnicode>().unwrap().as_str() {
                        "float32" => Some(DType::F32),
                        "int16" => Some(DType::I16),
                        _ => None,
                    },
                };
                Ok(Hdf5Dataset {
                    file,
                    dstype,
                    sr,
                    max_freq,
                    codec,
                    dtype,
                })
            }
        }
    }
    fn name(&self) -> String {
        self.file.filename()
    }
    fn group(&self) -> Result<hdf5::Group> {
        Ok(self.file.group(&self.dstype.to_string().to_lowercase())?)
    }
    pub fn len(&self) -> usize {
        self.group().unwrap().len() as usize
    }
    pub fn is_empty(&self) -> bool {
        self.group().unwrap().is_empty()
    }
    pub fn keys(&self) -> Result<Vec<String>> {
        Ok(self.group()?.member_names()?)
    }
    pub fn attributes(&self) -> Result<Vec<String>> {
        Ok(self.file.attr_names()?)
    }
    fn fmt_err(
        &self,
        fn_: &'static str,
        key: &str,
    ) -> impl FnOnce(hdf5::Error) -> DfDatasetError + '_ {
        let key = key.to_string();
        let name = self.name();
        move |e: hdf5::Error| -> DfDatasetError {
            DfDatasetError::Hdf5ErrorDetail {
                source: e,
                msg: format!("Error during {} of dataset {} key {}", fn_, name, key),
            }
        }
    }
    fn sample_len(&self, key: &str) -> Result<usize> {
        let ds = self.group()?.dataset(key)?;
        if *self.codec.as_ref().unwrap_or(&Codec::PCM) == Codec::Vorbis {
            let s = *ds.shape().last().unwrap(); // length of raw buffer
            let lastpages = ds
                .read_slice_1d(s![s - 50 * 1024 / 8..])
                .map_err(self.fmt_err("sample_len", key))?; // seek to last 50 kB
            let mut rdr = OggPacketReader::new(Cursor::new(lastpages.as_slice().unwrap()));
            // Ensure that rdr is at the start of a ogg page
            rdr.seek_absgp(None, 0).unwrap();
            let mut absgp = 0;
            while let Some(pkg) = rdr.read_packet()? {
                absgp = pkg.absgp_page();
            }
            Ok(absgp as usize)
        } else {
            Ok(*ds.shape().last().unwrap_or(&0))
        }
    }
    fn sample_shape(&self, key: &str) -> Result<Vec<usize>> {
        let ds = self.group()?.dataset(key)?;
        match *self.codec.as_ref().unwrap_or(&Codec::PCM) {
            Codec::PCM => Ok(ds.shape()),
            Codec::Vorbis => {
                let firstpages =
                    ds.read_slice_1d(s![..512]).map_err(self.fmt_err("sample_shape", key))?;
                let ident_hdr =
                    lewton::header::read_header_ident(firstpages.as_slice().unwrap()).unwrap();
                Ok(vec![ident_hdr.audio_channels.into(), self.sample_len(key)?])
            }
        }
    }
    pub fn read_pcm(&self, key: &str, r: Option<Range<usize>>) -> Result<Array2<f32>> {
        let ds = self.group()?.dataset(key)?;
        let mut arr: ArrayD<f32> = if let Some(r) = r {
            if r.end > *ds.shape().last().unwrap_or(&0) {
                return Err(DfDatasetError::PcmRangeToLarge {
                    range: r,
                    size: ds.shape(),
                });
            }
            match ds.ndim() {
                1 => ds.read_slice(s![r]).map_err(self.fmt_err("read_pcm", key))?,
                2 => ds.read_slice(s![0, r]).map_err(self.fmt_err("read_pcm", key))?, // Just take the first channel for now
                n => return Err(DfDatasetError::PcmUnspportedDimension(n)),
            }
        } else {
            ds.read_dyn::<f32>().map_err(self.fmt_err("read_pcm", key))?
        };
        #[allow(clippy::branches_sharing_code)]
        let mut arr = if arr.ndim() == 1 {
            let len = arr.len_of(Axis(0));
            arr.into_shape((1, len))?
        } else {
            let ch = arr.len_of(Axis(0));
            if ch > 1 {
                let idx = thread_rng()?.gen_range(0..ch);
                arr.slice_axis_inplace(Axis(0), Slice::from(idx..idx + 1));
            }
            arr.into_dimensionality()?
        };
        match self.dtype {
            Some(DType::I16) => arr /= std::i16::MAX as f32,
            Some(DType::F32) => (),
            None => {
                if ds.dtype()?.is::<i16>() {
                    arr /= std::i16::MAX as f32
                }
            }
        }
        Ok(arr)
    }
    pub fn read_vorbis(&self, key: &str, r: Option<Range<usize>>) -> Result<Array2<f32>> {
        let ds = self.group()?.dataset(key)?;
        let encoded = Cursor::new(ds.read_raw::<u8>().map_err(self.fmt_err("read_vorbis", key))?);
        let mut srr = OggStreamReader::new(encoded)?;
        let ch = srr.ident_hdr.audio_channels as usize;
        let mut out: Vec<i16> = Vec::new();
        while let Some(mut pck) = srr.read_dec_packet_itl()? {
            out.append(&mut pck);
        }
        let mut out: Array2<i16> = Array2::from_shape_vec((out.len() / ch, ch), out)?;
        if ch > 1 {
            let idx = thread_rng()?.gen_range(0..ch);
            out.slice_axis_inplace(Axis(1), Slice::from(idx..idx + 1));
        }
        debug_assert_eq!(1, out.len_of(Axis(1)));
        if let Some(r) = r {
            out.slice_axis_inplace(Axis(0), Slice::from(r));
        }
        let out = out.mapv(|x| x as f32 / std::i16::MAX as f32);
        // Transpose to channels first
        let out_len = out.len_of(Axis(0));
        Ok(out.into_shape((1, out_len))?)
    }

    pub fn read(&self, key: &str) -> Result<Array2<f32>> {
        match *self.codec.as_ref().unwrap_or_default() {
            Codec::PCM => self.read_pcm(key, None),
            Codec::Vorbis => self.read_vorbis(key, None),
        }
    }
    pub fn read_slc(&self, key: &str, r: Range<usize>) -> Result<Array2<f32>> {
        match *self.codec.as_ref().unwrap_or_default() {
            Codec::PCM => self.read_pcm(key, Some(r)),
            Codec::Vorbis => self.read_vorbis(key, Some(r)),
        }
    }
}

struct LpParam {
    sr: usize,
    cut_off: usize,
}

fn combine_noises(
    ch: usize,
    len: usize,
    noises: &mut [Array2<f32>],
    noise_gains: Option<&[f32]>,
) -> Result<Signal> {
    let mut rng = thread_rng()?;
    // Adjust length of noises to clean length
    for ns in noises.iter_mut() {
        loop {
            if len.checked_sub(ns.len_of(Axis(1))).is_some() {
                // TODO: Remove this clone if ndarray supports repeat
                ns.append(Axis(1), ns.clone().view())?;
            } else {
                break;
            }
        }
        let too_large = ns.len_of(Axis(1)).checked_sub(len);
        if let Some(too_large) = too_large {
            let start: usize = rng.gen_range(0..too_large);
            ns.slice_collapse(s![.., start..start + len]);
        }
    }
    // Adjust number of noise channels to clean channels
    for ns in noises.iter_mut() {
        while ns.len_of(Axis(0)) > ch {
            ns.remove_index(Axis(0), rng.gen_range(0..ns.len_of(Axis(0))))
        }
        while ns.len_of(Axis(0)) < ch {
            let r = rng.gen_range(0..ns.len_of(Axis(0)));
            let slc = ns.slice(s![r..r + 1, ..]).to_owned();
            ns.append(Axis(0), slc.view())?;
        }
    }
    // Apply gain to noises
    if let Some(ns_gains) = noise_gains {
        for (ns, &g) in noises.iter_mut().zip(ns_gains) {
            *ns *= 10f32.powf(g / 20.);
        }
    }
    // Average noises
    let noise = Array2::zeros((ch, len));
    let noise = noises.iter().fold(noise, |acc, x| acc + x) / ch as f32;
    Ok(noise)
}

/// Mix a clean signal with noise signal at given SNR.
///
/// Arguments
///
/// * `clean` - A clean speech signal of shape `[C, N]`.
/// * `clean_rev` - A optional reverberant speech signal of shape `[C, N]`. If provided, this signal
///                 will be used for creating the noisy mixture. `clean` may be used as a training
///                 target and usually contains no or less reverberation. This can be used to learn
///                 some dereverberation.
/// * `noise` - A noise signal of shape `[C, N]`. Will be modified in place.
/// * `snr_db` - Signal to noise ratio in decibel used for mixing.
/// * `gain_db` - Gain to apply to the clean signal in decibel before mixing.
/// * `atten_db` - Target attenuation limit in decibel. The resulting clean target will contain
///                `atten_db` less noise compared to the noisy output.
/// * `noise_resample`: Optional resample parameters which will be used to apply a low-pass via
///                     resampling to the noise signal. This may be used to make sure a speech
///                     signal with a lower sampling rate will also be mixed with noise having the
///                     same sampling rate.
fn mix_audio_signal(
    clean: Array2<f32>,
    clean_rev: Option<Array2<f32>>,
    mut noise: Array2<f32>,
    snr_db: f32,
    gain_db: f32,
    atten_db: Option<f32>,
    noise_resample: Option<LpParam>,
) -> Result<(Signal, Signal, Signal)> {
    let len = clean.len_of(Axis(1));
    if let Some(re) = noise_resample {
        // Low pass filtering via resampling
        noise = low_pass_resample(&noise, re.cut_off, re.sr)?;
        noise.slice_axis_inplace(Axis(1), Slice::from(..len));
    }
    // Apply gain to speech
    let g = 10f32.powf(gain_db / 20.);
    let mut clean_out = &clean * g;
    // clean_mix may contain reverberant speech
    let clean_mix = clean_rev.map(|c| &c * g).unwrap_or_else(|| clean_out.clone());
    // For energy calculation use clean speech to also consider direct-to-reverberant ratio
    noise *= mix_f(clean_out.view(), noise.view(), snr_db);
    if let Some(atten_db) = atten_db {
        // Create a mixture with a higher SNR as target signal
        let k_target = 1. / mix_f(clean_out.view(), noise.view(), snr_db + atten_db);
        clean_out *= k_target;
        // clean_mix *= k_target;
        clean_out += &noise;
    }
    let mut mixture = clean_mix + &noise;
    // Guard against clipping
    let max = &([&clean_out, &noise, &mixture].iter().map(|x| find_max_abs(x.iter())))
        .collect::<std::result::Result<Vec<f32>, crate::util::UtilsError>>()?;
    let max = find_max(max)?;
    if (max - 1.) > 1e-10 {
        let f = 1. / (max + 1e-10);
        clean_out *= f;
        noise *= f;
        mixture *= f;
    }
    Ok((clean_out, noise, mixture))
}

fn calc_snr_mixture<'a, I>(y: I, v: I) -> f32
where
    I: IntoIterator<Item = &'a f32>,
{
    let mut e_clean = 0.;
    let mut e_noise = 0.;
    for (xy, xv) in y.into_iter().zip(v.into_iter()) {
        e_clean += (xy - xv).powi(2);
        e_noise += xv.powi(2);
    }
    10. * (e_clean / e_noise).log10()
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
    use crate::util::seed_from_u64;

    fn calc_rms(x: &[f32]) -> f32 {
        let n = x.len() as f32;
        (x.iter().map(|x| x.powi(2)).sum::<f32>() * (1. / n)).sqrt()
    }
    fn calc_snr<'a, I>(x: I, v: I) -> f32
    where
        I: IntoIterator<Item = &'a f32>,
    {
        let e_clean = x.into_iter().fold(0f32, |acc, x| acc + x.powi(2));
        let e_noise = v.into_iter().fold(0f32, |acc, x| acc + x.powi(2));
        10. * (e_clean / e_noise).log10()
    }
    fn calc_snr_mixture<'a, I>(y: I, v: I) -> f32
    where
        I: IntoIterator<Item = &'a f32>,
    {
        let mut e_clean = 0.;
        let mut e_noise = 0.;
        for (xy, xv) in y.into_iter().zip(v.into_iter()) {
            e_clean += (xy - xv).powi(2);
            e_noise += xv.powi(2);
        }
        10. * (e_clean / e_noise).log10()
    }

    #[test]
    pub fn test_mix_audio_signal() -> Result<()> {
        seed_from_u64(0);
        let sr = 48_000;
        let n = sr;
        let clean = arr1(rng_uniform(n, -0.1, 0.1)?.as_slice()).into_shape([1, n])?;
        let noise = arr1(rng_uniform(n, -0.1, 0.1)?.as_slice()).into_shape([1, n])?;
        let gains = [-6., 0., 6.];
        let snrs = [-10., -5., 0., 5., 10., 20., 40.];
        let atten_limits = [None, Some(20.), Some(10.), Some(3.)];
        let atol = 1e-4;
        for clean_rev in [None, Some(clean.clone())] {
            for gain in gains {
                for snr in snrs {
                    for attn in atten_limits {
                        let (c, n, m) = mix_audio_signal(
                            clean.clone(),
                            clean_rev.clone(),
                            noise.clone(),
                            snr,
                            gain,
                            attn,
                            None,
                        )?;
                        if attn.is_none() {
                            assert_eq!(&c + &n, m);
                        }
                        dbg!(clean_rev.is_some(), gain, snr, attn);
                        // Input SNR of mixture
                        let snr_inp_m = calc_snr_mixture(m.iter(), n.iter());
                        assert!(
                            (snr_inp_m - snr).abs() < atol,
                            "Input SNR does not match: {}, {}",
                            snr_inp_m,
                            snr
                        );
                        // Target SNR between noise and target (clean) speech. Clean speech may
                        // contain noise due to the attenuation limit, effectively resulting in an
                        // `attn` higher SNR.
                        let snr_target_c = if attn.is_none() {
                            calc_snr(c.iter(), n.iter())
                        } else {
                            // With enabled attenuation limiting, the target signal `c` contains
                            // some noise so that its SNR is `attn` higher then the input mixture.
                            calc_snr_mixture(c.iter(), n.iter())
                        };
                        assert!(
                            (snr_target_c - (snr + attn.unwrap_or(0.))).abs() < atol,
                            "Target SNR does not match: {}, {}",
                            snr_target_c,
                            snr + attn.unwrap_or(0.),
                        );
                        // Test the SNR difference between input and target
                        assert!((snr_inp_m + attn.unwrap_or(0.) - snr_target_c).abs() < atol);
                    }
                }
            }
        }
        Ok(())
    }

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
        let mut cfg = DatasetConfigJson::open("../assets/dataset.cfg")?;
        let split = Split::Train;
        let builder = DatasetBuilder::new(ds_dir, sr)
            .df_params(fft_size, hop_size, nb_erb, nb_spec, norm_alpha)
            .max_len(1.);
        for dataset_size in [1, 2, 4, 17] {
            for c in cfg.train.iter_mut() {
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
                dbg!(dataset_size, batch_size);
                let mut loader = match DataLoader::builder(ds)
                    .num_threads(1)
                    .batch_size(batch_size)
                    .batch_size_eval(1)
                    .build()
                {
                    Ok(loader) => loader,
                    Err(e) => match e {
                        DfDatasetError::DatasetTooSmall {
                            split: s_,
                            dataset_size: ds_,
                            batch_size: bs_,
                        } => {
                            if dataset_size < batch_size {
                                continue 'inner; // This is expected
                            }
                            return Err(DfDatasetError::DatasetTooSmall {
                                split: s_,
                                dataset_size: ds_,
                                batch_size: bs_,
                            });
                        }
                        e => return Err(e),
                    },
                };
                for epoch in 0..2 {
                    loader.start_epoch(split, epoch)?;
                    let mut n_samples = 0;
                    loop {
                        let batch = loader.get_batch::<Complex32>()?;
                        if let Some(batch) = batch {
                            n_samples += batch.batch_size();
                            dbg!(n_samples);
                        } else {
                            break;
                        }
                    }
                    assert_eq!(n_samples, dataset_size);
                }
            }
        }
        Ok(())
    }
}
