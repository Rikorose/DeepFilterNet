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

use hdf5::{types::VarLenUnicode, File};
use lewton::inside_ogg::OggStreamReader;
use ndarray::prelude::*;
use ndarray::Slice;
use ogg::reading::PacketReader as OggPacketReader;
use rand::prelude::{IteratorRandom, SliceRandom};
use rand::Rng;
use rayon::prelude::*;
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
pub struct DatasetConfig {
    pub train: Vec<Hdf5Cfg>,
    pub valid: Vec<Hdf5Cfg>,
    pub test: Vec<Hdf5Cfg>,
}
impl DatasetConfig {
    pub fn open(path: &str) -> Result<Self> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        let cfg = serde_json::from_reader(reader)?;
        Ok(cfg)
    }
}

pub struct Datasets<T> {
    train: Arc<dyn Dataset<T> + Sync + Send>,
    valid: Arc<dyn Dataset<T> + Sync + Send>,
    test: Arc<dyn Dataset<T> + Sync + Send>,
}

impl<T> Datasets<T> {
    pub fn new(
        train: Arc<dyn Dataset<T> + Sync + Send>,
        valid: Arc<dyn Dataset<T> + Sync + Send>,
        test: Arc<dyn Dataset<T> + Sync + Send>,
    ) -> Self {
        Datasets { train, valid, test }
    }
    fn get<S: Into<Split>>(&self, split: S) -> &Arc<dyn Dataset<T> + Sync + Send> {
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

pub struct DataLoader<T>
where
    T: Data,
{
    datasets: Datasets<T>,
    batch_size_train: usize,
    batch_size_eval: usize,
    num_prefech: usize,
    idcs: Arc<Mutex<VecDeque<usize>>>,
    current_split: Split,
    fill_thread: Option<thread::JoinHandle<Result<()>>>,
    out_receiver: Option<Receiver<Result<Sample<T>>>>,
    overfit: bool,
}

#[derive(Default)]
pub struct DataLoaderBuilder<T>
where
    T: Data,
{
    _ds: Option<Datasets<T>>,
    _batch_size: Option<usize>,
    _batch_size_eval: Option<usize>,
    _prefetch: Option<usize>,
    _num_threads: Option<usize>,
    _overfit: Option<bool>,
}

impl<T> DataLoaderBuilder<T>
where
    T: Data,
{
    pub fn new(ds: Datasets<T>) -> Self {
        DataLoaderBuilder::<T> {
            _ds: Some(ds),
            _batch_size: None,
            _batch_size_eval: None,
            _prefetch: None,
            _num_threads: None,
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
    pub fn build(self) -> Result<DataLoader<T>> {
        let bs = self._batch_size.unwrap_or(1);
        let prefetch = self._prefetch.unwrap_or(bs * self._num_threads.unwrap_or(4) * 2);
        let mut loader = DataLoader::new(
            self._ds.unwrap(),
            bs,
            self._batch_size_eval,
            prefetch,
            self._num_threads,
        )?;
        loader.overfit = self._overfit.unwrap_or(false);
        Ok(loader)
    }
}

impl<T> DataLoader<T>
where
    T: Data,
{
    pub fn builder(ds: Datasets<T>) -> DataLoaderBuilder<T> {
        DataLoaderBuilder::new(ds)
    }
    pub fn new(
        datasets: Datasets<T>,
        batch_size_train: usize,
        batch_size_eval: Option<usize>,
        num_prefech: usize,
        num_threads: Option<usize>,
    ) -> Result<Self> {
        // Register global rayon threadpool. It will only be used for data loader workers.
        let mut poolbuilder = rayon::ThreadPoolBuilder::new();
        if let Some(num_threads) = num_threads {
            poolbuilder = poolbuilder.num_threads(num_threads)
        }
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
            datasets,
            batch_size_train,
            batch_size_eval,
            num_prefech,
            idcs: Arc::new(Mutex::new(VecDeque::new())),
            current_split: Split::Train,
            fill_thread: None,
            out_receiver: None,
            overfit: false,
        })
    }

    pub fn dataset_len<S: Into<Split>>(&self, split: S) -> usize {
        self.datasets.get(split).len()
    }

    pub fn len_of<S: Into<Split>>(&self, split: S) -> usize {
        let split = split.into();
        let bs = self.batch_size(&split);
        self.dataset_len(split) / bs
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
        seed: u64,
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
        let ds = Arc::clone(self.datasets.get(split));
        let idcs = self.idcs.clone();
        let handle = thread::spawn(move || -> Result<()> {
            idcs.lock().unwrap().par_drain(..).try_for_each_init(
                || {
                    seed_from_u64(seed);
                },
                // TODO: This closure get's submitted to the thread pool in order. However,
                // get_sample may take different amounts of time resulting in a different return
                // order. This should be the last major thing that reduces reproducability a little.
                // To make sure, we get the samples in the correct order, we could add another
                // ordering index and some kind of cache to use in get_batch().
                |(), idx| -> Result<()> {
                    let sample = ds.get_sample(idx);
                    if let Err(e) = out_sender.send(sample) {
                        return Err(DfDatasetError::SendError(e.to_string()));
                    }
                    Ok(())
                },
            )?;
            Ok(())
        });
        Ok(handle)
    }

    pub fn start_epoch<S: Into<Split>>(&mut self, split: S, seed: usize) -> Result<()> {
        let split: Split = split.into();
        // Drop fill thread if exits
        if self.fill_thread.is_some() {
            self.join_fill_thread()?;
        }
        // Prepare for new epoch
        self.current_split = split;
        if self.batch_size(&split) > self.dataset_len(split) {
            return Err(DfDatasetError::DatasetTooSmall {
                split,
                dataset_size: self.dataset_len(split),
                batch_size: self.batch_size(&split),
            });
        }
        seed_from_u64(seed as u64);
        {
            // Recreate indices to index into the dataset and shuffle them
            let mut idcs = self.idcs.lock().unwrap();
            if self.overfit {
                println!("Overfitting on one batch.");
                let bs = self.batch_size(&split);
                idcs.clone_from(&(0..bs).cycle().take(self.dataset_len(split)).collect());
            } else {
                idcs.clone_from(&(0..self.dataset_len(split)).collect());
                idcs.make_contiguous().shuffle(&mut thread_rng()?);
            }
        }
        // Start thread to submit dataset jobs for the pool workers
        self.fill_thread = Some(self.start_idx_worker(split, seed as u64)?);
        Ok(())
    }

    pub fn get_batch<C>(&mut self) -> Result<Option<DsBatch<T>>>
    where
        C: Collate<T>,
    {
        let bs = self.batch_size(&self.current_split);
        let mut samples = Vec::with_capacity(bs);
        let mut i = 0;
        let mut tries = 0;
        let reciever = match self.out_receiver.as_ref() {
            None => {
                return Err(DfDatasetError::ChannelsNotInitializedError);
            }
            Some(r) => r,
        };
        'outer: while i < bs {
            match reciever.recv_timeout(Duration::from_millis(100)) {
                Err(_e) => {
                    let isempty = if let Ok(idcs) = self.idcs.try_lock() {
                        idcs.is_empty()
                    } else {
                        false
                    };
                    if isempty {
                        self.join_fill_thread()?;
                        return Ok(None);
                    }
                    if tries > 1000 {
                        return Err(DfDatasetError::TimeoutError);
                    }
                    tries += 1;
                    continue 'outer;
                }
                Ok(s) => samples.push(s?),
            }
            i += 1;
            tries = 0;
        }

        if samples.is_empty() {
            println!("No more samples.");
            return Ok(None);
        }
        let out = C::collate(
            samples.as_mut_slice(),
            self.datasets.get(self.current_split).max_sample_len(),
        )?;
        Ok(Some(out))
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
        })
    }
}

impl<T> Drop for DataLoader<T>
where
    T: Data,
{
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
}
impl<T> fmt::Debug for DsBatch<T>
where
    T: Data,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "Dataset Batch with batch_size: '{}, len: '{}', snrs: '{:?}', gain: '{:?}')",
            self.speech.len_of(Axis(0)),
            self.speech.len_of(Axis(2)),
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
    fn get_sample(&self, idx: usize) -> Result<Sample<T>>;
    fn sr(&self) -> usize;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn max_sample_len(&self) -> usize;
    fn set_seed(&mut self, seed: u64);
}

#[derive(Clone)]
pub struct DatasetBuilder<'a> {
    ds_dir: &'a str,
    sr: usize,
    fft_size: Option<usize>,
    datasets: Vec<Hdf5Cfg>,
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
            datasets: Vec::new(),
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
        if self.datasets.is_empty() {
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
    pub fn build_td_dataset(mut self) -> Result<TdDataset> {
        if self.datasets.is_empty() {
            panic!("No datasets provided")
        }
        let max_samples: usize = (self.max_len_s.unwrap_or(10.) * self.sr as f32).round() as usize;
        let mut hdf5_handles = Vec::new();
        let mut sp_keys: Vec<(usize, String)> = Vec::new();
        let mut ns_keys: Vec<(usize, String)> = Vec::new();
        let mut rir_keys: Vec<(usize, String)> = Vec::new();
        let mut config: Vec<Hdf5Cfg> = Vec::new();
        let mut i = 0;
        for cfg in self.datasets.drain(..) {
            let name = cfg.filename();
            let path = Path::new(self.ds_dir).join(name);
            if (!path.is_file())
                || match path.read_link() {
                    Err(_) => false,
                    Ok(p) => !p.is_file(),
                }
            {
                eprintln!("Dataset {:?} not found. Skipping.", path);
                continue;
            }
            let ds = Hdf5Dataset::new(path.to_str().unwrap())?;
            let n_samples = (cfg.sampling_factor() * ds.len() as f32).round() as usize;
            let keys: Vec<(usize, String)> =
                ds.keys()?.iter().cycle().take(n_samples).map(|k| (i, k.clone())).collect();
            match ds.dstype {
                DsType::Speech => sp_keys.extend(keys),
                DsType::Noise => ns_keys.extend(keys),
                DsType::RIR => rir_keys.extend(keys),
            }
            hdf5_handles.push(ds);
            config.push(cfg);
            i += 1;
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
        if p_reverb > 0. && rir_keys.is_empty() {
            eprintln!("Warning: Reverb augmentation enabled but no RIRs provided!");
        }
        let reverb = RandReverbSim::new(p_reverb, self.sr);
        let seed = self.seed.unwrap_or(0);
        Ok(TdDataset {
            config,
            hdf5_handles,
            max_samples,
            sr: self.sr,
            sp_keys,
            ns_keys,
            rir_keys,
            snrs,
            gains,
            attenuation_range,
            p_fill_speech,
            p_atten_lim,
            sp_transforms,
            ns_transforms,
            reverb,
            seed,
        })
    }
    pub fn dataset(mut self, datasets: Vec<Hdf5Cfg>) -> Self {
        self.datasets.extend(datasets);
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
    fn get_sample(&self, idx: usize) -> Result<Sample<Complex32>> {
        let sample: Sample<f32> = self.ds.get_sample(idx)?;
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
}

pub struct TdDataset {
    config: Vec<Hdf5Cfg>,
    hdf5_handles: Vec<Hdf5Dataset>,
    max_samples: usize,
    sr: usize,
    sp_keys: Vec<(usize, String)>,
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
    fn get_sample(&self, idx: usize) -> Result<Sample<f32>> {
        seed_from_u64(idx as u64 + self.seed);
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
}

#[derive(Debug)]
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
    let clean_mix = clean_rev.map(|c| &c * g).unwrap_or_else(|| clean_out.clone());
    // For energy calculation use clean speech to also consider direct-to-reverberant ratio
    let k = mix_f(clean.view(), noise.view(), snr_db);
    if let Some(atten_db) = atten_db {
        // Create a mixture with a higher SNR as target signal
        let k_target = mix_f(clean.view(), noise.view(), snr_db + atten_db);
        for (c, &n) in clean_out.iter_mut().zip(noise.iter()) {
            *c += n * k_target;
        }
    }
    // Create mixture at given SNR
    noise *= k;
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
    use std::time::Instant;

    use dirs::home_dir;

    use super::*;
    use crate::util::seed_from_u64;
    use crate::wav_utils::*;

    #[test]
    pub fn test_mix_audio_signal() -> Result<()> {
        seed_from_u64(42);
        // 2ch 10 second speech signal
        let reader = ReadWav::new("../assets/clean_freesound_33711.wav")?;
        let (sr, ch) = (reader.sr as u32, reader.channels as u16);
        let clean = reader.samples_arr2()?;
        // 1ch shorter then clean
        let noise1 = ReadWav::new("../assets/noise_freesound_573577.wav")?.samples_arr2()?;
        // 2ch longer then clean
        let noise2 = ReadWav::new("../assets/noise_freesound_2530.wav")?.samples_arr2()?;
        let noise = combine_noises(
            ch as usize,
            clean.len_of(Axis(1)),
            &mut [noise1, noise2],
            None,
        )?;
        let (clean, noise, noisy) =
            mix_audio_signal(clean, None, noise, 0., 6., None, None).unwrap();
        dbg!(noisy.len());
        write_wav_iter("../out/clean.wav", clean.iter(), sr, ch)?;
        write_wav_iter("../out/noise.wav", noise.iter(), sr, ch)?;
        write_wav_iter("../out/noisy.wav", noisy.iter(), sr, ch)?;
        Ok(())
    }

    #[test]
    pub fn test_hdf5_read() -> Result<()> {
        let hdf5 = Hdf5Dataset::new(
            home_dir().unwrap().join("data/hdf5/EDINBURGH_56.hdf5").to_str().unwrap(),
        )?;
        let sr = hdf5.sr.unwrap() as u32;
        let keys = hdf5.keys()?;
        let signal = hdf5.read(&keys[0])?;
        dbg!(signal.shape());
        let max_len = 3 * sr as usize; // 1 second
        let key = &keys[0];
        let signal = hdf5.read_slc(key, 0..max_len.min(hdf5.sample_len(key)?))?;
        dbg!(signal.shape());
        let ch = signal.len_of(Axis(0));
        write_wav_iter("../out/hdf5_signal.wav", &signal, sr, ch as u16)?;
        Ok(())
    }

    #[test]
    pub fn test_hdf5_vorbis_read() -> Result<()> {
        seed_from_u64(42);
        let hdf5 = Hdf5Dataset::new(
            home_dir().unwrap().join("data/hdf5/OWN_NOISES_TRAIN.hdf5").to_str().unwrap(),
        )?;
        let sr = hdf5.sr.unwrap() as u32;
        let keys = hdf5.keys()?;
        let key = &keys[0];
        let signal = hdf5.read(key)?;
        write_wav_arr2("../out/hdf5_signal.wav", signal.view(), sr)?;
        dbg!(signal.shape());
        let max_len = 3 * sr as usize;
        let signal = hdf5.read_slc(key, 0..max_len.min(hdf5.sample_len(key)?))?;
        dbg!(signal.shape());
        write_wav_arr2("../out/hdf5_signal_slc.wav", signal.view(), sr)?;
        Ok(())
    }

    #[test]
    pub fn test_data_loader() -> Result<()> {
        println!("******** Start test_data_loader() ********");
        seed_from_u64(42);
        let batch_size = 1;
        let sr = 48000;
        let ds_dir = home_dir().unwrap().join("data/hdf5").to_str().unwrap().to_string();
        let cfg = DatasetConfig::open("../assets/dataset.cfg")?;
        let builder = DatasetBuilder::new(&ds_dir, sr);
        let ds = Datasets::new(
            Arc::new(builder.clone().dataset(cfg.train).build_td_dataset()?),
            Arc::new(builder.clone().dataset(cfg.valid).build_td_dataset()?),
            Arc::new(builder.clone().dataset(cfg.test).build_td_dataset()?),
        );
        let mut loader = DataLoader::builder(ds).batch_size(batch_size).build()?;
        loader.start_epoch("train", 1)?;
        for i in 0..10 {
            let t0 = Instant::now();
            let batch = loader.get_batch::<f32>()?.unwrap();
            dbg!(i, &batch);
            let t1 = Instant::now();
            println!("test_data_loader: {:?}", t1 - t0);
            write_wav_iter(
                "../out/clean.wav",
                &batch.speech.slice(s![0, 0, ..]),
                sr as u32,
                1,
            )?;
            write_wav_iter(
                "../out/noise.wav",
                &batch.noise.slice(s![0, 0, ..]),
                sr as u32,
                1,
            )?;
            write_wav_iter(
                "../out/noisy.wav",
                &batch.noisy.slice(s![0, 0, ..]),
                sr as u32,
                1,
            )?;
        }
        loader.start_epoch("train", 2)?;
        for i in 0..2 {
            dbg!(i, loader.get_batch::<f32>()?);
        }
        println!("Dropping loader");
        drop(loader);
        println!("Done");
        Ok(())
    }

    #[test]
    pub fn test_fft_dataset() -> Result<()> {
        println!("******** Start test_data_loader() ********");
        seed_from_u64(42);
        let batch_size = 2;
        let fft_size = 960;
        let hop_size = Some(480);
        let nb_erb = Some(32);
        let nb_spec = None;
        let norm_alpha = None;
        let sr = 48000;
        let ds_dir = home_dir().unwrap().join("data/hdf5").to_str().unwrap().to_string();
        let cfg = DatasetConfig::open("../assets/dataset.cfg")?;
        let builder = DatasetBuilder::new(&ds_dir, sr)
            .df_params(fft_size, hop_size, nb_erb, nb_spec, norm_alpha);
        let ds = Datasets::new(
            Arc::new(builder.clone().dataset(cfg.train).build_fft_dataset()?),
            Arc::new(builder.clone().dataset(cfg.valid).build_fft_dataset()?),
            Arc::new(builder.clone().dataset(cfg.test).build_fft_dataset()?),
        );
        let mut loader = DataLoader::builder(ds).num_threads(1).batch_size(batch_size).build()?;
        loader.start_epoch("train", 1)?;
        for i in 0..2 {
            let batch = loader.get_batch::<Complex32>()?;
            if let Some(batch) = batch {
                dbg!(i, &batch, batch.feat_erb.as_ref().unwrap().shape());
            }
        }
        Ok(())
    }
}
