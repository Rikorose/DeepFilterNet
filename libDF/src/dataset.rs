use std::collections::{hash_map::DefaultHasher, HashMap};
use std::env;
use std::ffi::OsStr;
use std::fmt;
#[cfg(feature = "timings")]
use std::fmt::Write as _;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter};
#[cfg(feature = "vorbis")]
use std::io::{Cursor, Read, Seek};
use std::ops::Range;
use std::path::Path;
use std::str::FromStr;
use std::sync::mpsc::sync_channel;
#[cfg(feature = "timings")]
use std::time::Instant;
use std::time::SystemTime;

use anyhow::Context;
#[cfg(feature = "flac")]
use claxon;
use hdf5::{types::VarLenUnicode, File};
use ndarray::concatenate;
use ndarray::{prelude::*, Slice};
use ndarray_rand::rand::prelude::{IteratorRandom, SliceRandom};
use rayon::prelude::*;
use realfft::num_traits::Zero;
use serde::{Deserialize, Serialize};
use thiserror::Error;
#[cfg(feature = "vorbis")]
use {lewton::inside_ogg::OggStreamReader, ogg::reading::PacketReader as OggPacketReader};

use crate::{augmentations::*, transforms::*, util::*, *};

type Result<T> = std::result::Result<T, DfDatasetError>;

#[derive(Error, Debug)]
pub enum DfDatasetError {
    #[error("No Hdf5 datasets found")]
    NoDatasetFoundError,
    #[error("No Hdf5 dataset type found")]
    Hdf5DsTypeNotFoundError,
    #[error("{codec:?} codec not supported in dataset {ds:?}")]
    CodecNotSupportedError { codec: Codec, ds: String },
    #[error("Unsupported during PCM decode: {0}")]
    PcmUnspportedDimension(usize),
    #[error("Wav Reader Error")]
    WavReadError(#[from] crate::wav_utils::WavUtilsError),
    #[error("Input Range ({range:?}) larger than dataset size ({size:?})")]
    PcmRangeToLarge {
        range: Range<usize>,
        size: Vec<usize>,
    },
    #[error("Data Processing Error: {0:?}")]
    DataProcessingError(String),
    #[error("DF Transforms Error")]
    TransformError(#[from] crate::transforms::TransformError),
    #[error("DF Augmentation Error")]
    AugmentationError(#[from] crate::augmentations::AugmentationError),
    #[error("DF Utils Error")]
    UtilsError(#[from] crate::util::UtilsError),
    #[error("Error Detail")]
    ErrorDetail { source: Box<Self>, msg: String },
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
    #[cfg(feature = "vorbis")]
    #[error("Vorbis Decode Error")]
    VorbisError(#[from] lewton::VorbisError),
    #[cfg(feature = "vorbis")]
    #[error("Ogg Decode Error")]
    OggReadError(#[from] ogg::reading::OggReadError),
    #[cfg(feature = "flac")]
    #[error("Flac Decode Error")]
    FlacError(#[from] claxon::Error),
    #[error("Multithreading Send Error: {0:?}")]
    SendError(String),
    #[error("Multithreading Recv Error: {0:?}")]
    RecvError(String),
    #[error("Thread Join Error: {0:?}")]
    ThreadJoinError(String),
    #[error("Crossbeam Multithreading Send Error: {0:?}")]
    CrossbeamSendError(String),
    #[error("Not enough {0} samples in the dataset.")]
    NotEnoughSamplesError(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error), // source and Display delegate to anyhow::Error
}

impl<T> From<crossbeam_channel::SendError<T>> for DfDatasetError {
    fn from(error: crossbeam_channel::SendError<T>) -> Self {
        DfDatasetError::CrossbeamSendError(error.to_string())
    }
}

impl From<std::sync::mpsc::RecvError> for DfDatasetError {
    fn from(error: std::sync::mpsc::RecvError) -> Self {
        DfDatasetError::RecvError(error.to_string())
    }
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
/// Struct to check if a dataset was modified on disk.
/// This is used for caching the HDF5 keys, i.e. speech and noise file names.
/// The modification is checked via last modification time and file size.
#[derive(Hash, Debug, Clone)]
pub struct DatasetModified(SystemTime, u64);
impl DatasetModified {
    fn new(file_name: &str) -> Result<Self> {
        let meta_data = fs::metadata(file_name)?;
        let modified = meta_data.modified()?;
        let size = meta_data.len();
        log::trace!(
            "Checking HDF5 key cache for {} (m: {:?}, s: {})",
            file_name,
            modified,
            size
        );
        Ok(DatasetModified(modified, size))
    }
}
/// HDF5 config for each dataset.
///
/// Contains:
///     - file name of the HDF5 dataset on disk.
///     - sampling factor to over/under sample the dataset during training.
///     - fallback sampling rate if the HDF5 dataset does not store the sampling rate.
///     - fallback max_freq (unused)
///     - cached key list may additionally be used for loading the HDF5 keys from a cache file.
///     - modified hash is used to indicate if the cached key list is up to date.
#[derive(Deserialize, Debug, Clone)]
pub struct Hdf5Cfg(
    pub String,                                                 // file name
    #[serde(default = "one")] pub f32,                          // dataset sampling factor
    #[serde(default = "Option::default")] pub Option<usize>,    // fallback sampling rate
    #[serde(default = "Option::default")] pub Option<usize>,    // fallback max freq
    #[serde(default = "Option::default")] pub Option<Hdf5Keys>, // cached key list
    #[serde(default = "Option::default")] pub Option<u64>,      // modified hash
);
impl Hdf5Cfg {
    pub fn filename(&self) -> &str {
        self.0.as_str()
    }
    pub fn sampling_factor(&self) -> f32 {
        self.1
    }
    pub fn set_sampling_factor(&mut self, f: f32) {
        self.1 = f
    }
    pub fn fallback_sr(&self) -> Option<usize> {
        self.2
    }
    pub fn fallback_max_freq(&self) -> Option<usize> {
        self.3
    }
    pub fn keys_unchecked(&self) -> Option<&Hdf5Keys> {
        self.4.as_ref()
    }
    pub fn hash(&self) -> Option<u64> {
        self.5
    }
    pub fn store_modified_hash(&mut self, hash: u64) {
        self.5 = Some(hash);
    }
    pub fn hash_from_ds_path(&self, ds_path: &str) -> Result<u64> {
        Ok(calculate_hash(&DatasetModified::new(ds_path)?))
    }
    pub fn load_keys(&self, hash: u64) -> Result<Option<&Hdf5Keys>> {
        if let Some(keys) = self.keys_unchecked() {
            if keys.hash == hash {
                return Ok(Some(keys));
            }
            log::warn!(
                "Hash does not match for {} (found {}, expected {})",
                self.filename(),
                keys.hash,
                hash
            );
            return Ok(None);
        }
        Ok(None)
    }
    pub fn set_keys_new(&mut self, hash: u64, keys: Vec<String>) -> Result<()> {
        self.set_keys(Hdf5Keys {
            filename: self.filename().to_string(),
            hash,
            keys,
        })
    }
    pub fn set_keys(&mut self, keys: Hdf5Keys) -> Result<()> {
        self.4.replace(keys);
        Ok(())
    }
}
fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

/// Keys within a HDF5 dataset.
///
/// - filename: File name of the HDF5 dataset.
/// - hash: Modified hash of the dataset corresponding to loaded keys.
/// - keys: Speech/noise file names contained in the dataset.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Hdf5Keys {
    pub filename: String,
    hash: u64,
    keys: Vec<String>,
}
/// Dataset Configuration for train, validation, and test.
#[derive(Deserialize, Debug)]
pub struct DatasetConfigJson {
    pub train: Vec<Hdf5Cfg>,
    pub valid: Vec<Hdf5Cfg>,
    pub test: Vec<Hdf5Cfg>,
}
impl DatasetConfigJson {
    pub fn open(cfg_path: &str) -> Result<Self> {
        let file = fs::File::open(cfg_path)?;
        let reader = BufReader::new(file);
        let cfg = serde_json::from_reader(reader)?;
        Ok(cfg)
    }
    pub fn set_keys<S: Into<Split>>(&mut self, split: S, keys: &[Hdf5Keys]) -> Result<()> {
        let s = self.get_mut(split);
        for cfg in s.iter_mut() {
            if let Some(key_cache) = keys.iter().find(|c| c.filename == cfg.filename()) {
                cfg.set_keys(key_cache.clone())?;
            } else {
                log::warn!("Could not find cached keys for {}", cfg.filename());
            }
        }
        Ok(())
    }
    pub fn split_config(&self, split: Split) -> DatasetSplitConfig {
        DatasetSplitConfig {
            hdf5s: self.get(split).clone(),
            split,
        }
    }
    pub fn get<S: Into<Split>>(&self, split: S) -> &Vec<Hdf5Cfg> {
        match split.into() {
            Split::Train => &self.train,
            Split::Valid => &self.valid,
            Split::Test => &self.test,
        }
    }
    pub fn get_mut<S: Into<Split>>(&mut self, split: S) -> &mut Vec<Hdf5Cfg> {
        match split.into() {
            Split::Train => &mut self.train,
            Split::Valid => &mut self.valid,
            Split::Test => &mut self.test,
        }
    }
}
/// Helper struct to load cached HDF5 keys from a JSON file.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DatasetConfigCacheJson(Vec<Hdf5Keys>);
impl DatasetConfigCacheJson {
    pub fn new(keys: Vec<Hdf5Keys>) -> Self {
        DatasetConfigCacheJson(keys)
    }
    pub fn keys(&self) -> &Vec<Hdf5Keys> {
        &self.0
    }
    pub fn open(cache_path: &str) -> Result<Self> {
        let file = fs::File::open(cache_path)?;
        let reader = BufReader::new(file);
        log::trace!("Opening HDF5 json key cache {}", cache_path);
        let cfg = serde_json::from_reader(reader)?;
        Ok(cfg)
    }
    pub fn write(&self, cache_path: &str) -> Result<()> {
        let file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(cache_path)?;
        let writer = BufWriter::new(file);
        log::trace!("Writing HDF5 keys to cache {}", cache_path);
        serde_json::to_writer(writer, &self)?;
        Ok(())
    }
}

/// Helper struct containing all `Hdf5Cfg`s of a single split (train/valid/test).
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

/// Dataset base struct for the `DataLoader`.
/// Contains all (train, valid, test) `FftDataset`s.
pub struct Datasets {
    pub train: FftDataset,
    pub valid: FftDataset,
    pub test: FftDataset,
}

impl Datasets {
    pub fn get<S: Into<Split>>(&self, split: S) -> &FftDataset {
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
            s => panic!("Split '{s}' does not exist."),
        }
    }
}

pub trait Data: Sized + Clone + Default + Send + Sync + Zero + 'static {}
impl Data for f32 {}
impl Data for Complex32 {}

/// A dataset sample containing a noisy/clean pair for training.
///
/// This struct is used for holding the augmented time-domain signal as well as the complex
/// STFT-domain signal from `FftDataset`.
#[derive(Clone, Serialize, Deserialize)]
pub struct Sample<T>
where
    T: Data,
{
    pub speech: ArrayD<T>,
    pub noisy: ArrayD<T>,
    pub feat_erb: Option<ArrayD<f32>>,
    pub feat_spec: Option<ArrayD<Complex32>>,
    pub max_freq: usize,
    pub snr: i8,
    pub gain: i8,
    pub idx: usize,
    pub downsample_freq: Option<usize>,
}
impl Sample<f32> {
    fn get_speech_view(&self) -> Result<ArrayView2<f32>> {
        Ok(self.speech.view().into_dimensionality()?)
    }
    fn get_noisy_view(&self) -> Result<ArrayView2<f32>> {
        Ok(self.noisy.view().into_dimensionality()?)
    }
    fn dim(&self) -> usize {
        2
    }
}
impl Sample<Complex32> {
    fn get_speech_view(&self) -> Result<ArrayView3<Complex32>> {
        Ok(self.speech.view().into_dimensionality()?)
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
    fn generate_keys(&mut self, epoch_seed: Option<u64>) -> Result<()>;
}

#[derive(Clone)]
pub struct DatasetBuilder {
    ds_dir: String,
    sr: usize,
    fft_size: Option<usize>,
    datasets: Option<DatasetSplitConfig>,
    max_len_s: Option<f32>,
    hop_size: Option<usize>,
    nb_erb: Option<usize>,
    nb_spec: Option<usize>,
    norm_alpha: Option<f32>,
    p_reverb: Option<f32>,
    p_fill_speech: Option<f32>,
    seed: Option<u64>,
    min_nb_freqs: Option<usize>,
    global_sampling_f: Option<f32>,
    snrs: Option<Vec<i8>>,
    gains: Option<Vec<i8>>,
    num_threads: Option<usize>,
    p_bandwidth_ext: Option<f32>,
    p_clipping: Option<f32>,
    p_zeroing: Option<f32>,
    p_air_absorption: Option<f32>,
}
impl DatasetBuilder {
    pub fn new(ds_dir: &str, sr: usize) -> Self {
        DatasetBuilder {
            ds_dir: ds_dir.to_string(),
            sr,
            datasets: None,
            max_len_s: None,
            fft_size: None,
            hop_size: None,
            nb_erb: None,
            nb_spec: None,
            norm_alpha: None,
            p_reverb: None,
            p_fill_speech: None,
            seed: None,
            min_nb_freqs: None,
            global_sampling_f: None,
            snrs: None,
            gains: None,
            num_threads: None,
            p_bandwidth_ext: None,
            p_clipping: None,
            p_zeroing: None,
            p_air_absorption: None,
        }
    }
    pub fn build_fft_dataset(self) -> Result<FftDataset> {
        if self.datasets.is_none() {
            panic!("No datasets provided")
        }
        let ds = self.clone().build_td_dataset()?;
        let split = self.datasets.unwrap().split;
        log::trace!("Building FftDataset {}", split);
        if self.fft_size.is_none() {
            panic!("No fft size provided when building FFT dataset.")
        }
        let fft_size = self.fft_size.unwrap();
        let hop_size = self.hop_size.unwrap_or(fft_size / 2);
        let nb_erb = self.nb_erb.unwrap_or(32);
        if let Some(b) = self.nb_spec {
            let nfreqs = fft_size / 2 + 1;
            if b > nfreqs {
                let msg = format!("Number of spectrogram bins ({b}) is larger then number of available frequency bins ({nfreqs})");
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
        let datasets = match self.datasets {
            None => panic!("No datasets provided"),
            Some(ds) => ds,
        };
        // TODO: Return all sample by default and not only 10 seconds
        let max_samples: usize = (self.max_len_s.unwrap_or(10.) * self.sr as f32).round() as usize;
        // Get dataset handles and keys. Each key is a unique String.
        let ds_path = Path::new(&self.ds_dir);
        let (sender, receiver) = sync_channel(datasets.hdf5s.len() + 1);
        if let Some(n) = self.num_threads {
            hdf5::sync::sync(|| {});
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .thread_name(|idx| format!("DataLoader Worker {idx}"))
                .start_handler(|_| hdf5::sync::sync(|| {}))
                .build_global()
                .unwrap_or(());
        }
        datasets.hdf5s.par_iter().try_for_each(|cfg| -> Result<()> {
            let path = ds_path.join(cfg.filename());
            log::trace!("Opening hdf5 dataset {}", path.display());
            if (!path.is_file()) && path.read_link().is_err() {
                log::warn!("Dataset {:?} not found. Skipping.", path);
                return Ok(());
            }
            let mut cfg = cfg.clone();
            let ds = Hdf5Dataset::new(path.to_str().unwrap())?;
            let modified_hash = cfg.hash_from_ds_path(path.to_str().unwrap())?;
            cfg.store_modified_hash(modified_hash);
            let keys = cfg.load_keys(modified_hash)?.cloned();
            match keys {
                Some(keys) => {
                    log::trace!("Found cached hdf5 keys for {}", cfg.filename());
                    cfg.set_keys(keys)?
                }
                None => {
                    log::trace!("No cached hdf5 keys found for {}", cfg.filename());
                    cfg.set_keys_new(modified_hash, ds.keys()?)?
                }
            };
            if let Some(f) = self.global_sampling_f {
                cfg.set_sampling_factor(cfg.sampling_factor() * f)
            }
            sender.send(Some((cfg, ds))).unwrap();
            Ok(())
        })?;
        sender.send(None).unwrap();
        let mut config = HashMap::new();
        let mut hdf5_handles = HashMap::new();
        let mut ds_keys = Vec::new();
        let mut has_rirs = false;
        let mut ds_len: usize = 0;
        while let Some((cfg, ds)) = receiver.try_recv().unwrap() {
            has_rirs = has_rirs || ds.dstype == DsType::RIR;
            let keys = cfg.keys_unchecked().unwrap().keys.clone();
            if ds.dstype == DsType::Speech {
                ds_len += ((keys.len() as f32 * cfg.sampling_factor()).round() as usize).max(1);
            }
            ds_keys.push((ds.dstype, ds.name(), keys));
            assert!(
                !config.contains_key(&ds.name()),
                "Config does not contain ds {}",
                ds.name()
            );
            config.insert(ds.name(), cfg);
            log::debug!(
                "Opened {} {} dataset {} with {} samples",
                &ds.dstype,
                datasets.split,
                ds.name(),
                ds.len()
            );
            hdf5_handles.insert(ds.name(), ds);
        }
        ds_keys.sort_by(|a, b| a.1.cmp(&b.1));
        if hdf5_handles.is_empty() {
            return Err(DfDatasetError::NoDatasetFoundError);
        }
        let snrs = self.snrs.unwrap_or_else(|| vec![-5, 0, 5, 10, 20, 40]);
        let gains = self.gains.unwrap_or_else(|| vec![-6, 0, 6]);
        let p_fill_speech = self.p_fill_speech.unwrap_or(0.);
        let ds_split = datasets.split;
        let sp_augmentations = Compose::new(vec![
            Box::new(RandRemoveDc::default_with_prob(
                get_env("DF_P_REMVOE_DC").unwrap_or(0.25),
            )),
            Box::new(RandLFilt::default_with_prob(
                get_env("DF_P_LFILT").unwrap_or(0.25),
            )),
            Box::new(
                RandBiquadFilter::default_with_prob(get_env("DF_P_BIQUAD").unwrap_or(0.0))
                    .with_sr(self.sr),
            ),
            Box::new(
                RandResample::default_with_prob(get_env("DF_P_LFILT").unwrap_or(0.1))
                    .with_sr(self.sr),
            ),
        ]);
        let mut sp_distortions_td = Compose::new(Vec::new());
        let mut sp_distortions_fd = Compose::new(Vec::new());
        if ds_split == Split::Train {
            let p_clipping: Option<f32> = match get_env("DF_P_CLIPPING") {
                Some(p) => Some(p),
                None => self.p_clipping,
            };
            if let Some(p) = p_clipping {
                if p > 0. {
                    sp_distortions_td.push(Box::new(
                        RandClipping::default_with_prob(p).with_c(0.05..0.9),
                    ))
                }
            }
            let p_zeroing: Option<f32> = match get_env("DF_P_ZEROING") {
                Some(p) => Some(p),
                None => self.p_zeroing,
            };
            if let Some(p) = p_zeroing {
                sp_distortions_td.push(Box::new(RandZeroingTD::default_with_prob(p)))
            }
            let p_air_absorption: Option<f32> = match get_env("DF_P_AIR_AUG") {
                Some(p) => Some(p),
                None => self.p_air_absorption,
            };
            if let Some(p) = p_air_absorption {
                if p > 0. {
                    sp_distortions_fd
                        .push(Box::new(AirAbsorptionAugmentation::default_with_prob(p)));
                }
            }
        }
        let mut ns_augmentations = Compose::new(vec![
            Box::new(RandLFilt::default_with_prob(
                get_env("DF_P_LFILT").unwrap_or(0.25),
            )),
            Box::new(
                RandBiquadFilter::default_with_prob(get_env("DF_P_BIQUAD").unwrap_or(0.0))
                    .with_sr(self.sr),
            ),
            Box::new(
                RandResample::default_with_prob(get_env("DF_P_LFILT").unwrap_or(0.1))
                    .with_sr(self.sr),
            ),
        ]);
        if ds_split == Split::Train {
            ns_augmentations.push(Box::new(
                RandClipping::default_with_prob(0.1).with_c(0.01..0.5),
            ))
        }
        let p_reverb = self.p_reverb.unwrap_or(0.);
        if p_reverb > 0. && !has_rirs {
            log::warn!("Reverb augmentation enabled but no RIRs provided!",);
        }
        let reverb = RandReverbSim::new(p_reverb, self.sr).with_drr(0.5);
        let seed = self.seed.unwrap_or(0);
        // 5% of noises used for mixing will contain randomly generated noise.
        // This has the advantage that the noise will actually contain frequencies up 24 kHz.
        let noise_generator =
            NoiseGenerator::new(self.sr, if ds_split == Split::Train { 0.05 } else { 0.0 });
        let bw_limiter = if let Some(p) = self.p_bandwidth_ext {
            if p > 0. {
                Some(BandwidthLimiterAugmentation::new(p, self.sr))
            } else {
                None
            }
        } else {
            None
        };
        log::trace!("Built TdDataset {}", ds_split);
        Ok(TdDataset {
            config,
            hdf5_handles,
            max_samples,
            sr: self.sr,
            ds_keys,
            ds_split,
            sp_keys: Vec::new(),
            ns_keys: Vec::new(),
            rir_keys: Vec::new(),
            snrs,
            gains,
            p_fill_speech,
            sp_augmentations,
            sp_distortions_td,
            sp_distortions_fd,
            ns_augmentations,
            noise_generator,
            reverb,
            seed,
            ds_len,
            bw_limiter,
        })
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
    pub fn global_sample_factor(mut self, f: f32) -> Self {
        self.global_sampling_f = Some(f);
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
    pub fn snrs(mut self, snrs: Vec<i8>) -> Self {
        self.snrs = Some(snrs);
        self
    }
    pub fn gains(mut self, gains: Vec<i8>) -> Self {
        self.gains = Some(gains);
        self
    }
    pub fn num_threads(mut self, n: usize) -> Self {
        self.num_threads = Some(n);
        self
    }
    pub fn bandwidth_extension(mut self, p: f32) -> Self {
        self.p_bandwidth_ext = Some(p);
        self
    }
    pub fn clipping_distortion(mut self, p: f32) -> Self {
        self.p_clipping = Some(p);
        self
    }
    pub fn zeroing_distortion(mut self, p: f32) -> Self {
        self.p_zeroing = Some(p);
        self
    }
    pub fn air_absorption_distortion(mut self, p: f32) -> Self {
        self.p_air_absorption = Some(p);
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
impl FftDataset {
    pub fn get_hdf5cfg(&self, filename: &str) -> Option<&Hdf5Cfg> {
        self.ds.config.values().find(|&cfg| cfg.filename() == filename)
    }
}
impl Dataset<Complex32> for FftDataset {
    fn get_sample(&self, idx: usize, seed: Option<u64>) -> Result<Sample<Complex32>> {
        #[cfg(feature = "timings")]
        let t0 = Instant::now();
        let sample: Sample<f32> = self.ds.get_sample(idx, seed)?;

        // To frequency domain
        let nb_erb = self.nb_erb.unwrap_or(1);
        let sr = self.sr();
        let fft_size = self.fft_size;
        let mut state = DFState::new(sr, fft_size, self.hop_size, nb_erb, 1);

        let speech = stft(sample.get_speech_view()?, &mut state, false);
        let mut noisy = stft(sample.get_noisy_view()?, &mut state, true);
        if let Some(f) = sample.downsample_freq {
            let max_bin = (f as f32 / (sr as f32 / fft_size as f32)) as usize;
            ext_bandwidth_spectral(&mut noisy, max_bin, sr, Some(4));
        }

        // Feature calculation (normalization)
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
        let sample = Sample {
            speech: speech.into_dyn(),
            noisy: noisy.into_dyn(),
            feat_erb: erb,
            feat_spec: spec,
            max_freq: sample.max_freq,
            gain: sample.gain,
            snr: sample.snr,
            idx: sample.idx,
            downsample_freq: sample.downsample_freq,
        };
        #[cfg(feature = "timings")]
        log::trace!(
            "FD sample: {:?} ms",
            (std::time::Instant::now() - t0).as_millis()
        );
        Ok(sample)
    }

    fn len(&self) -> usize {
        self.ds.len()
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

    fn generate_keys(&mut self, epoch_seed: Option<u64>) -> Result<()> {
        self.ds.generate_keys(epoch_seed)
    }
}

pub struct TdDataset {
    config: HashMap<String, Hdf5Cfg>,            // config
    hdf5_handles: HashMap<String, Hdf5Dataset>,  // Handles to access opened Hdf5 datasets
    max_samples: usize,                          // Number of samples in time domain
    sr: usize,                                   // Sampling rate
    ds_keys: Vec<(DsType, String, Vec<String>)>, // Dataset keys as a vector of [DS Type, hdf5 index, Vec<str keys>].
    ds_split: Split,                             // Train/Valid/Test
    sp_keys: Vec<(String, String)>, // Pair of hdf5 name and dataset keys. Will be generated at each epoch start
    ns_keys: Vec<(String, String)>,
    rir_keys: Vec<(String, String)>,
    snrs: Vec<i8>,                   // in dB; SNR to sample from
    gains: Vec<i8>,                  // in dB; Speech (loudness) to sample from
    p_fill_speech: f32, // Probability to completely fill the speech signal to `max_samples` with a different speech sample
    noise_generator: NoiseGenerator, // Create random noises
    sp_augmentations: Compose, // Transforms to augment speech samples
    sp_distortions_td: Compose, // Transforms to distort speech samples in time domain for used generating the mixture
    sp_distortions_fd: Compose, // Transforms to distort speech samples in frequency domain for used generating the mixture
    ns_augmentations: Compose,  // Transforms to augment noise samples
    reverb: RandReverbSim, // Separate reverb transform that may be applied to both speech and noise
    seed: u64,
    ds_len: usize,
    bw_limiter: Option<BandwidthLimiterAugmentation>, // Extend bandwidth via spectal translation
}

impl TdDataset {
    fn _read_from_hdf5(
        &self,
        key: &str,
        name: &str,
        max_len: Option<usize>,
    ) -> Result<Array2<f32>> {
        let h = &self.hdf5_handles.get(name).unwrap();
        let sr =
            h.sr.unwrap_or_else(|| self.config.get(name).unwrap().fallback_sr().unwrap_or(self.sr));
        let slc = if let Some(l) = max_len {
            let l_sr = l * sr / self.sr;
            let sample_len = h.sample_len(key)?;
            let max_len = sample_len.min(l_sr);
            let s = sample_len as i64 - max_len as i64;
            if s > 0 {
                let s = thread_rng()?.uniform(0, s as usize);
                Some(s..s + l_sr)
            } else {
                None
            }
        } else {
            None
        };
        let mut x = if let Some(slc) = slc {
            h.read_slc(key, slc)
        } else {
            h.read(key)
        }
        .map_err(move |e: DfDatasetError| -> DfDatasetError {
            DfDatasetError::ErrorDetail {
                source: Box::new(e),
                msg: format!("Error reading sample '{key}' from dataset {name}",),
            }
        })?;
        if sr != self.sr {
            x = resample(x.view(), sr, self.sr, None)?;
            if let Some(l) = max_len {
                if x.len_of(Axis(1)) > l {
                    x.slice_axis_inplace(Axis(1), Slice::from(0..l))
                }
            }
            return Ok(x);
        }
        Ok(x)
    }

    fn read(&self, name: &str, key: &str) -> Result<Array2<f32>> {
        let x = self._read_from_hdf5(key, name, None)?;
        Ok(x)
    }

    fn read_all_channels(&self, name: &str, key: &str) -> Result<Array2<f32>> {
        let x = self._read_from_hdf5(key, name, None)?;
        Ok(x)
    }

    fn read_max_len(
        &self,
        ds_name: &str,
        key: &str,
        max_samples: Option<usize>,
    ) -> Result<Array2<f32>> {
        #[cfg(feature = "timings")]
        let t0 = Instant::now();
        let max_samples = max_samples.unwrap_or(self.max_samples);
        let x = match self._read_from_hdf5(key, ds_name, Some(max_samples)) {
            Err(e) => {
                log::warn!(
                    "Error during {} read_max_len() for key '{}' from dataset {}: {:?}",
                    self.ds_type(ds_name),
                    key,
                    ds_name,
                    e
                );
                let e_str = e.to_string();
                if e_str.contains("inflate") || e_str.contains("Flac") {
                    // Get a different speech then
                    let idx = thread_rng()?.uniform(0, self.len());
                    let (sp_idx, sp_key) = &self.sp_keys[idx];
                    log::warn!(
                        "Returning a different speech sample from {} due to {}",
                        ds_name,
                        e_str
                    );
                    self.read_max_len(sp_idx, sp_key, Some(max_samples))?
                } else {
                    return Err(e);
                }
            }
            Ok(s) => s,
        };
        if log::log_enabled!(log::Level::Trace) {
            #[allow(unused_mut)]
            let mut msg = format!(
                "Loaded sample {} with codec {:?}",
                key,
                self.ds_codec(ds_name)
            );
            #[cfg(feature = "timings")]
            let _ = write!(msg, " in {} ms", (Instant::now() - t0).as_millis());
            log::trace!("{}", msg);
        }
        debug_assert!(x.len_of(Axis(1)) <= max_samples);
        Ok(x)
    }

    fn max_freq(&self, name: &str) -> Result<usize> {
        let ds = &self.hdf5_handles.get(name).unwrap();
        let max_freq = match ds.max_freq {
            Some(x) if x > 0 => x,
            _ => {
                let cfg = self.config.get(name).unwrap();
                cfg.fallback_max_freq().unwrap_or_else(|| {
                    ds.sr.unwrap_or_else(|| cfg.fallback_sr().unwrap_or(self.sr)) / 2
                })
            }
        };
        Ok(max_freq)
    }

    fn ds_type(&self, name: &str) -> String {
        self.hdf5_handles.get(name).unwrap().ds_type()
    }

    fn ds_codec(&self, name: &str) -> Codec {
        self.hdf5_handles.get(name).unwrap().codec.clone().unwrap_or_default()
    }

    fn load_aug_speech(&self, idx: usize, rng: &mut SeededRng) -> Result<(Array2<f32>, usize)> {
        let (mut sp_name, mut sp_key) = self.sp_keys[idx].clone();
        let mut max_freq = self.sr / 2;
        let mut cur_len = 0;
        let mut speech_samples = Vec::new();
        // Used for bandwidth extension
        let fft_size = 2048;
        let mut state = if self.bw_limiter.is_some() {
            Some(DFState::new(self.sr, fft_size, fft_size / 2, 1, 1))
        } else {
            None
        };
        loop {
            // Read 10% more samples since augmentation might shorten the signal
            let n_read = (self.max_samples as f32 * 1.1) as usize - cur_len;
            let mut sample = self.read_max_len(&sp_name, &sp_key, Some(n_read))?;
            if sample.len_of(Axis(0)) > 1 {
                sample.slice_axis_inplace(Axis(0), Slice::from(0usize..1));
            }
            max_freq = max_freq.min(self.max_freq(&sp_name)?);
            if self.bw_limiter.is_some() {
                if sample.len_of(Axis(1)) < 2048 {
                    log::debug!(
                        "Found sample with length {}. Skipping.",
                        sample.len_of(Axis(1))
                    );
                    (sp_name, sp_key) =
                        self.sp_keys.choose(rng).context("Failed to sample speech signal")?.clone();
                    continue;
                }
                // Extend clean speech to make sure it covers the full spectrum
                let mut spec = stft(sample.view(), state.as_mut().unwrap(), false);
                let max_bin = estimate_bandwidth(spec.view(), self.sr, -120., 10);
                let n_bins = fft_size / 2 + 1;
                if max_bin < n_bins {
                    ext_bandwidth_spectral(&mut spec, max_bin, self.sr, Some(16));
                    sample = istft(spec.view_mut(), state.as_mut().unwrap(), false);
                }
            }
            if crate::rms(sample.iter()) < 1e-10 {
                log::debug!(
                    "Speech sample before augmentation {} is zero! Choosing new one.",
                    idx
                );
                (sp_name, sp_key) =
                    self.sp_keys.choose(rng).context("Failed to sample speech signal")?.clone();
                continue;
            }
            self.sp_augmentations.transform(&mut (&mut sample).into())?;
            if crate::rms(sample.iter()) < 1e-10 {
                log::debug!(
                    "Speech sample after augmentation {} is zero! Choosing new one.",
                    idx
                );
                (sp_name, sp_key) =
                    self.sp_keys.choose(rng).context("Failed to sample speech signal")?.clone();
                continue;
            }
            cur_len += sample.len_of(Axis(1));
            speech_samples.push(sample);
            if cur_len < self.max_sample_len() {
                (sp_name, sp_key) =
                    self.sp_keys.choose(rng).context("Failed to sample speech signal")?.clone();
            } else {
                break;
            }
        }
        let speech_views: Vec<ArrayView2<f32>> = speech_samples.iter().map(|s| s.view()).collect();
        let mut speech = concatenate(Axis(1), speech_views.as_slice())?;
        let len = speech.len_of(Axis(1));
        if len > self.max_samples {
            let start = rng.uniform(0, len - self.max_samples);
            speech.slice_axis_inplace(Axis(1), Slice::from(start..(start + self.max_samples)))
        }
        Ok((speech, max_freq))
    }

    fn load_aug_noise(&self, rng: &mut SeededRng) -> Result<(Array2<f32>, f32)> {
        // In 5% us a randomly generated noise signal instead of a real noise.
        if let Some(ns) =
            self.noise_generator.maybe_generate_random_noise(-2., 2., 1, self.max_samples)?
        {
            return Ok((ns, *[-24., -12., -6., 0.].choose(rng).unwrap()));
        }
        loop {
            let (ns_name, ns_key) =
                self.ns_keys.iter().choose(rng).context("Failed to sample noise signal")?;
            let mut ns = match self.read_max_len(ns_name, ns_key, None) {
                Err(e) => {
                    log::warn!("Error during noise reading get_sample(): {}", e);
                    continue;
                }
                Ok(n) => n,
            };
            if ns.len_of(Axis(1)) < 100 {
                continue;
            }
            if find_max_abs(ns.as_slice().unwrap()).expect("NaN") < 1e-10 {
                log::debug!("No energy found in noise {}, ds {}", ns_key, ns_name);
                continue;
            }
            self.ns_augmentations.transform(&mut (&mut ns).into())?;
            if ns.len_of(Axis(1)) > self.max_samples {
                ns.slice_axis_inplace(Axis(1), Slice::from(..self.max_samples));
            }
            return Ok((ns, *self.gains.choose(rng).unwrap() as f32));
        }
    }
}

impl Dataset<f32> for TdDataset {
    fn get_sample(&self, idx: usize, seed: Option<u64>) -> Result<Sample<f32>> {
        #[cfg(feature = "timings")]
        let t0 = Instant::now();
        let sample_seed = seed.unwrap_or(idx as u64);
        log::trace!("get_sample() idx {} with seed {:?}", idx, sample_seed,);
        seed_from_u64(self.seed + seed.unwrap_or(idx as u64));
        let mut rng = thread_rng()?;
        let (mut speech, max_freq) = self.load_aug_speech(idx, &mut rng)?;
        if crate::rms(speech.iter()) < 1e-10 {
            log::warn!(
                "No speech signal found for idx {}, seed {}",
                idx,
                sample_seed
            );
        }
        #[cfg(feature = "timings")]
        let t_sp = Instant::now();
        // Apply low pass to the noise as well
        let mut noise_low_pass = if max_freq < self.sr / 2 {
            Some(LpParam {
                cut_off: max_freq,
                sr: self.sr,
            })
        } else {
            None
        };
        let ch = speech.len_of(Axis(0));
        let len = speech.len_of(Axis(1));
        // Sample 2-5 noises and augment each
        let n_noises = rng.uniform(2, 6);
        let mut noises = Vec::with_capacity(n_noises);
        let mut noise_gains = Vec::with_capacity(n_noises);
        for _ in 0..n_noises {
            let (ns, gain) = self.load_aug_noise(&mut rng)?;
            noises.push(ns);
            noise_gains.push(gain);
        }
        // Sample SNR and gain
        let &snr = self.snrs.choose(&mut rng).unwrap();
        let &gain = self.gains.choose(&mut rng).unwrap();
        // Truncate to speech len, combine noises and mix to noisy
        let mut noise = combine_noises(ch, len, &mut noises, Some(noise_gains.as_slice()))?;
        #[cfg(feature = "timings")]
        let t_ns = Instant::now();
        // Optionally we may also introduce some distortions to the speech signal.
        // These distortions will be only present in the noisy mixture, with the aim to reconstruce
        // the original undistorted signal. Example distortions are reverberation, or clipping.
        //
        // Apply reverberation using a randomly sampled RIR
        let mut speech_distorted = {
            if !self.rir_keys.is_empty() {
                self.reverb.transform(&mut speech, &mut noise, || {
                    let (rir_name, rir_key) = self.rir_keys.iter().choose(&mut rng).unwrap();
                    let rir = self.read(rir_name, rir_key)?;
                    log::trace!("Sampled RIR {} with shape {:?}", rir_key, rir.shape());
                    Ok(rir)
                })?
            } else {
                None
            }
        }
        .unwrap_or_else(|| speech.clone());
        // TD distortions like clipping
        if !self.sp_distortions_td.is_empty() {
            self.sp_distortions_td.transform(&mut (&mut speech_distorted).into())?;
        }
        // Bandwidth limitation
        let downsample_freq = if let Some(limiter) = self.bw_limiter.as_ref() {
            let f = limiter.transform(&mut speech_distorted, max_freq)?;
            noise_low_pass = Some(LpParam {
                cut_off: f,
                sr: self.sr,
            });
            Some(f)
        } else {
            None
        };
        if let Some(re) = noise_low_pass {
            // Low pass filtering via resampling to match speech cut off frequency
            noise = low_pass_resample(noise.view(), re.cut_off, re.sr)?;
            noise.slice_axis_inplace(Axis(1), Slice::from(..len));
        }
        // FD distortions
        if !self.sp_distortions_fd.is_empty() {
            let fft_size = 2048;
            let mut state = DFState::new(self.sr, fft_size, fft_size / 2, 1, 1);
            let mut x = stft(speech_distorted.view(), &mut state, false);
            self.sp_distortions_fd.transform(&mut (&mut x).into())?;
            speech_distorted = istft(x.view_mut(), &mut state, false);
            speech_distorted.slice_axis_inplace(Axis(1), Slice::from(0..speech.len_of(Axis(1))));
        }
        #[cfg(feature = "timings")]
        let t_d = Instant::now(); // distortions
        let (speech, _, noisy) = mix_audio_signal(
            speech,
            Some(speech_distorted),
            noise,
            snr as f32,
            gain as f32,
        )?;
        #[cfg(feature = "timings")]
        if log::log_enabled!(log::Level::Trace) {
            let te = std::time::Instant::now();
            log::trace!(
                "TD sample: {:?} ms (speech: {:?} ms, noise: {:?} ms, distortions: {:?}, mix: {:?} ms)",
                (te - t0).as_millis(),
                (t_sp - t0).as_millis(),
                (t_ns - t_sp).as_millis(),
                (t_d - t_ns).as_millis(),
                (te - t_d).as_millis(),
            );
        }
        Ok(Sample {
            speech: speech.into_dyn(),
            noisy: noisy.into_dyn(),
            feat_erb: None,
            feat_spec: None,
            max_freq,
            snr,
            gain,
            idx,
            downsample_freq,
        })
    }

    fn len(&self) -> usize {
        self.ds_len
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
            for (_, name, _) in self.ds_keys.iter() {
                let f = self.config.get(name).unwrap().sampling_factor();
                // if not a natural number, then we need to regenerate.
                if f != f.round() {
                    return true;
                }
            }
        }
        false
    }

    fn generate_keys(&mut self, epoch_seed: Option<u64>) -> Result<()> {
        self.sp_keys.clear();
        self.ns_keys.clear();
        self.rir_keys.clear();

        if let Some(s) = epoch_seed {
            seed_from_u64(s)
        }

        for (dstype, name, keys) in self.ds_keys.iter() {
            debug_assert_eq!(&self.hdf5_handles.get(name).unwrap().keys().unwrap(), keys);
            let len = keys.len();
            let n_samples =
                ((self.config.get(name).unwrap().sampling_factor() * len as f32).round() as usize)
                    .max(1);
            let mut keys = keys.clone();
            if self.ds_split == Split::Train {
                keys.shuffle(&mut thread_rng()?);
            }
            let keys: Vec<(String, String)> =
                keys.iter().cycle().take(n_samples).map(|k| (name.clone(), k.clone())).collect();
            match dstype {
                DsType::Speech => self.sp_keys.extend(keys),
                DsType::Noise => self.ns_keys.extend(keys),
                DsType::RIR => self.rir_keys.extend(keys),
            }
        }
        if self.sp_keys.is_empty() {
            return Err(DfDatasetError::NotEnoughSamplesError("speech".to_string()));
        }
        if self.ns_keys.is_empty() {
            return Err(DfDatasetError::NotEnoughSamplesError("noise".to_string()));
        }
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
        write!(f, "{self:?}")
    }
}
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Codec {
    PCM = 0,
    Vorbis = 1,
    FLAC = 2,
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
    pub dstype: DsType,
    pub sr: Option<usize>,
    pub codec: Option<Codec>,
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
    pub fn new(path: &str) -> Result<Self> {
        let file = Self::open_file(path, false)?;
        Self::init_impl(file)
    }
    pub fn new_rw(path: &str) -> Result<Self> {
        let file = Self::open_file(path, true)?;
        Self::init_impl(file)
    }
    fn open_file(path: &str, rw: bool) -> Result<File> {
        let file = if rw {
            File::open_rw(path)
        } else {
            File::open(path)
        };
        file.map_err(move |e: hdf5::Error| -> DfDatasetError {
            DfDatasetError::Hdf5ErrorDetail {
                source: e,
                msg: format!("Error during File::open of dataset {path}"),
            }
        })
    }
    pub fn inner(&self) -> &hdf5::File {
        &self.file
    }
    pub fn inner_mut(&mut self) -> &mut hdf5::File {
        &mut self.file
    }
    fn init_impl(file: File) -> Result<Self> {
        let n = file.filename();
        log::trace!("Init Dataset {}", n);
        match get_dstype(&file) {
            None => Err(DfDatasetError::Hdf5DsTypeNotFoundError),
            Some(dstype) => {
                let sr = match file.attr("sr") {
                    Err(_e) => {
                        log::warn!("Failed to get sr from {}", n);
                        None
                    }
                    Ok(attr) => Some(attr.read_scalar::<usize>().unwrap()),
                };
                let max_freq = match file.attr("max_freq") {
                    Err(_e) => {
                        log::warn!("Failed to get max_freq from {}", n);
                        None
                    }
                    Ok(attr) => Some(attr.read_scalar::<usize>().unwrap()),
                };
                let codec = match file.attr("codec") {
                    Err(_e) => {
                        log::warn!("Failed to get codec from {}", n);
                        None
                    }
                    Ok(attr) => match attr.read_scalar::<VarLenUnicode>().unwrap().as_str() {
                        "pcm" => Some(Codec::PCM),
                        "vorbis" => Some(Codec::Vorbis),
                        "flac" => Some(Codec::FLAC),
                        _ => None,
                    },
                };
                let dtype = match file.attr("dtype") {
                    Err(_e) => {
                        log::warn!("Failed to get dtype from {}", n);
                        None
                    }
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
    pub fn ds_type(&self) -> String {
        self.dstype.to_string().to_lowercase()
    }
    fn group(&self) -> Result<hdf5::Group> {
        Ok(self.file.group(&self.ds_type())?)
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
    pub fn ds(&self, key: &str) -> Result<hdf5::Dataset> {
        Ok(self.group()?.dataset(key)?)
    }
    #[cfg(not(feature = "flac"))]
    fn sample_len_flac(&self, _ds: hdf5::Dataset) -> Result<usize> {
        Err(DfDatasetError::CodecNotSupportedError {
            codec: Codec::FLAC,
            ds: format!("{:?}", self.file),
        })
    }
    #[cfg(feature = "flac")]
    /// Get the sample length of a Flac encoded dataset
    ///
    /// Arguments:
    ///
    /// * `ds`: `hdf5::Dataset` containing a flac encoded audio sample.
    fn sample_len_flac(&self, ds: hdf5::Dataset) -> Result<usize> {
        let reader = claxon::FlacReader::new(ds.as_byte_reader()?)?;
        Ok(reader.streaminfo().samples.unwrap_or(0) as usize)
    }
    #[cfg(not(feature = "vorbis"))]
    fn sample_len_vorbis(&self, _ds: hdf5::Dataset) -> Result<usize> {
        Err(DfDatasetError::CodecNotSupportedError {
            codec: Codec::Vorbis,
            ds: format!("{:?}", self.file),
        })
    }
    #[cfg(feature = "vorbis")]
    fn sample_len_vorbis_rdr<T: Read + Seek>(&self, rdr: &mut OggPacketReader<T>) -> Result<usize> {
        // Seek almost to end to get the last ogg package
        rdr.seek_bytes(std::io::SeekFrom::End(-4096))?;
        let mut pkg = rdr.read_packet();
        if pkg.is_err() {
            // Maybe seek a little further or start entirely from the beginning.
            rdr.seek_bytes(std::io::SeekFrom::End(-8192))?;
            pkg = rdr.read_packet();
            if pkg.is_err() {
                rdr.seek_absgp(None, 0)?;
                pkg = rdr.read_packet();
            };
        }
        // Also check if there are some packges left
        let mut absgp = 0;
        while let Some(p) = pkg? {
            absgp = p.absgp_page();
            pkg = rdr.read_packet();
        }
        Ok(absgp as usize)
    }
    #[cfg(feature = "vorbis")]
    /// Get the sample length of a vorbis encoded dataset
    ///
    /// Arguments:
    ///
    /// * `ds`: `hdf5::Dataset` containing a vorbis (ogg) encoded audio sample.
    fn sample_len_vorbis(&self, ds: hdf5::Dataset) -> Result<usize> {
        let mut rdr = OggPacketReader::new(ds.as_byte_reader()?);
        self.sample_len_vorbis_rdr(&mut rdr)
    }
    fn sample_len_from_ds(&self, ds: hdf5::Dataset) -> Result<usize> {
        Ok(match self.codec.as_ref().unwrap_or(&Codec::PCM) {
            Codec::PCM => *ds.shape().last().unwrap_or(&0),
            Codec::Vorbis => self.sample_len_vorbis(ds)?,
            Codec::FLAC => self.sample_len_flac(ds)?,
        })
    }
    pub fn sample_len(&self, key: &str) -> Result<usize> {
        let ds = self.ds(key)?;
        let n = match ds.attr("n_samples") {
            Ok(a) => {
                let n: usize = match a.ndim() {
                    0 => a.read_scalar::<usize>()?,
                    1 => a.read_1d()?[0],
                    _ => unreachable!(),
                };
                if n < 100 {
                    self.sample_len_from_ds(ds)?
                } else {
                    n
                }
            }
            Err(_) => self.sample_len_from_ds(ds)?,
        };
        Ok(n)
    }
    fn match_ch<T, D: ndarray::Dimension>(
        &self,
        mut x: Array<T, D>,
        ch_dim: usize,
        ch_idx: Option<isize>,
    ) -> Result<Array2<T>> {
        Ok(match x.ndim() {
            1 => {
                // Return in channels first
                let len = x.len_of(Axis(0));
                x.into_shape((1, len))?
            }
            2 => match ch_idx {
                Some(-1) => {
                    let idx = thread_rng()?.uniform(0, x.len_of(Axis(ch_dim)));
                    x.slice_axis_inplace(Axis(ch_dim), Slice::from(idx..idx + 1));
                    x
                }
                Some(idx) => {
                    x.slice_axis_inplace(Axis(ch_dim), Slice::from(idx..idx + 1));
                    x
                }
                None => x,
            }
            .into_dimensionality()?,
            n => return Err(DfDatasetError::PcmUnspportedDimension(n)),
        })
    }
    /// Read a PCM encoded sample from an `hdf5::Dataset`.
    ///
    /// Arguments:
    ///
    /// * `key`: String idendifier to load the dataset.
    /// * `channel`: Optional channel. `-1` will load a random channel, `None` will return all channels.
    /// * `r`: Optional range in samples (time axis). `None` will return all samples.
    pub fn read_pcm(
        &self,
        key: &str,
        channel: Option<isize>,
        r: Option<Range<usize>>,
    ) -> Result<Array2<f32>> {
        let ds = self.ds(key)?;
        let arr = if let Some(r) = r {
            // Directly to a sliced dataset read
            if r.end > *ds.shape().last().unwrap_or(&0) {
                return Err(DfDatasetError::PcmRangeToLarge {
                    range: r,
                    size: ds.shape(),
                });
            }
            match ds.ndim() {
                1 => ds.read_slice(s![r])?,
                2 => match channel {
                    Some(-1) => {
                        let nch = ds.shape()[1];
                        ds.read_slice(s![thread_rng()?.uniform(0, nch), r])
                    } // rand ch
                    Some(channel) => ds.read_slice(s![channel, r]), // specified channel
                    None => ds.read_slice(s![.., r]),               // all channels
                }?,
                n => return Err(DfDatasetError::PcmUnspportedDimension(n)),
            }
        } else {
            ds.read_dyn::<f32>()?
        };
        let mut arr = self.match_ch(arr, 0, channel)?;
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
    #[cfg(not(feature = "flac"))]
    fn read_flac(
        &self,
        _key: &str,
        _channel: Option<isize>,
        _r: Option<Range<usize>>,
    ) -> Result<Array2<f32>> {
        Err(DfDatasetError::CodecNotSupportedError {
            codec: Codec::FLAC,
            ds: format!("{:?}", self.file),
        })
    }
    #[cfg(feature = "flac")]
    fn _read_flac<R: std::io::Read>(
        &self,
        key: &str,
        mut reader: claxon::FlacReader<R>,
    ) -> Result<Array2<f32>> {
        let info = reader.streaminfo();
        assert_eq!(
            info.bits_per_sample, 16,
            "Flac decoding is only supported for 16 bit samples"
        );
        let ch = info.channels as usize;
        let samples = info.samples.unwrap_or_default() as usize;
        let mut frame_reader = reader.blocks();
        let mut out: Array2<f32> = Array2::zeros((ch, samples));
        let mut block = claxon::Block::empty();
        let mut idx = 0;
        loop {
            let next = match frame_reader.read_next_or_eof(block.into_buffer()) {
                Ok(Some(n)) => n,
                Ok(None) => break,
                Err(e) => {
                    log::warn!("Error decoding flac dataset {} {:?}", key, e);
                    if e.to_string().contains("CRC") {
                        break;
                    } else {
                        return Err(e.into());
                    }
                }
            };
            let numel = (next.len() / next.channels()) as usize;
            debug_assert_eq!(ch, next.channels() as usize);
            for i in 0..ch {
                debug_assert!(out.len_of(Axis(1)) >= idx + numel);
                let mut out_ch = out.slice_mut(s![i, idx..idx + numel]);
                debug_assert_eq!(out_ch.len(), next.channel(i as u32).len());
                for (i_s, o_s) in next.channel(i as u32).iter().zip(out_ch.iter_mut()) {
                    *o_s = *i_s as f32 / std::i16::MAX as f32
                }
            }
            idx += numel;
            block = next
        }
        Ok(out)
    }
    #[cfg(feature = "flac")]
    fn read_flac_byte_reader(&self, key: &str) -> Result<Array2<f32>> {
        let ds = self.ds(key)?;
        let reader = claxon::FlacReader::new(ds.as_byte_reader()?)?;
        self._read_flac(key, reader)
    }
    #[cfg(feature = "flac")]
    fn read_flac_ds(&self, key: &str) -> Result<Array2<f32>> {
        let ds = self.ds(key)?;
        let encoded = ds.read_1d()?;
        let reader = claxon::FlacReader::new(encoded.as_slice().unwrap())?;
        self._read_flac(key, reader)
    }
    /// Read a Flac encoded sample from an `hdf5::Dataset`.
    ///
    /// Arguments:
    ///
    /// * `key`: String idendifier to load the dataset.
    /// * `channel`: Optional channel. `-1` will load a random channel, `None` will return all channels.
    /// * `r`: Optional range in samples (time axis). `None` will return all samples.
    #[cfg(feature = "flac")]
    fn read_flac(
        &self,
        key: &str,
        channel: Option<isize>,
        r: Option<Range<usize>>,
    ) -> Result<Array2<f32>> {
        let out = self.read_flac_ds(key)?;
        let mut out = self.match_ch(out, 0, channel)?;
        if let Some(r) = r {
            out.slice_axis_inplace(Axis(1), Slice::from(r));
        }
        Ok(out)
    }
    #[cfg(not(feature = "vorbis"))]
    fn read_vorbis(
        &self,
        _key: &str,
        _channel: Option<isize>,
        _r: Option<Range<usize>>,
    ) -> Result<Array2<f32>> {
        Err(DfDatasetError::CodecNotSupportedError {
            codec: Codec::Vorbis,
            ds: format!("{:?}", self.file),
        })
    }
    #[inline(never)]
    #[cfg(feature = "vorbis")]
    /// Read a vorbis encoded sample from an `hdf5::Dataset`.
    ///
    /// Arguments:
    ///
    /// * `key`: String idendifier to load the dataset.
    /// * `channel`: Optional channel. `-1` will load a random channel, `None` will return all channels.
    /// * `r`: Optional range in samples (time axis). `None` will return all samples.
    fn read_vorbis(
        &self,
        key: &str,
        channel: Option<isize>,
        r: Option<Range<usize>>,
    ) -> Result<Array2<f32>> {
        let ds = self.ds(key)?;
        let encoded = ds.read_1d()?;
        let mut rdr = OggPacketReader::new(Cursor::new(encoded.as_slice().unwrap()));
        let (start, end) = if let Some(r) = r.as_ref() {
            (r.start, r.end)
        } else {
            (0, self.sample_len_vorbis_rdr(&mut rdr)?)
        };
        let len = end - start;
        rdr.seek_absgp(None, 0)?;
        let mut srr = OggStreamReader::from_ogg_reader(rdr)?;
        if start > 0 {
            match srr.seek_absgp_pg(start as u64) {
                Ok(()) => (),
                Err(e) => {
                    log::trace!("Error seeking in vorbis file {}: {:?}", key, e);
                    // Decode the vorbis file from start and truncate at the end.
                }
            }
        }
        let ch = srr.ident_hdr.audio_channels as usize;
        let mut pck = loop {
            match srr.read_dec_packet_itl() {
                Ok(p) => break p,
                Err(lewton::VorbisError::BadAudio(
                    lewton::audio::AudioReadError::AudioIsHeader,
                )) => (),
                Err(e) => return Err(e.into()),
            }
        };

        let mut out: Vec<i16> = Vec::with_capacity((len + 1024) * ch); // Allocate a little extra
        while let Some(mut p) = pck {
            out.append(&mut p);
            if let Some(pos) = srr.get_last_absgp().map(|p| p as usize) {
                if pos >= end {
                    // We might get some extra samples at the end.
                    out.truncate((out.len() - (pos - end) * ch).max(len * ch));
                    debug_assert!(out.len() >= len);
                    break;
                }
            }
            pck = srr.read_dec_packet_itl()?;
        }
        let start_pos = (out.len() / ch).saturating_sub(len);
        let mut out = Array2::from_shape_vec((out.len() / ch, ch), out)?;
        // We already have a coarse range. The start may contain more samples from its
        // corresponding ogg page. The end is already exact. Thus, truncate the beginning.
        let cur_len = out.len_of(Axis(0));
        out.slice_axis_inplace(
            Axis(0),
            Slice::from(start_pos..(len + start_pos).min(cur_len)),
        );
        // Select channel
        let out = self.match_ch(out, 1, channel)?;
        // Transpose to channels first and convert to float
        let out = out.t().mapv(|x| x as f32 / std::i16::MAX as f32);
        Ok(out)
    }

    pub fn read(&self, key: &str) -> Result<Array2<f32>> {
        match *self.codec.as_ref().unwrap_or_default() {
            Codec::PCM => self.read_pcm(key, Some(0), None),
            Codec::Vorbis => self.read_vorbis(key, Some(0), None),
            Codec::FLAC => self.read_flac(key, Some(0), None),
        }
    }
    pub fn read_slc(&self, key: &str, r: Range<usize>) -> Result<Array2<f32>> {
        match *self.codec.as_ref().unwrap_or_default() {
            Codec::PCM => self.read_pcm(key, Some(0), Some(r)),
            Codec::Vorbis => self.read_vorbis(key, Some(0), Some(r)),
            Codec::FLAC => self.read_flac(key, Some(0), Some(r)),
        }
    }
    pub fn read_all_channels(&self, key: &str) -> Result<Array2<f32>> {
        match *self.codec.as_ref().unwrap_or_default() {
            Codec::PCM => self.read_pcm(key, None, None),
            Codec::Vorbis => self.read_vorbis(key, None, None),
            Codec::FLAC => self.read_flac(key, None, None),
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
            let start: usize = rng.uniform(0, too_large);
            ns.slice_collapse(s![.., start..start + len]);
        }
    }
    // Adjust number of noise channels to clean channels
    for ns in noises.iter_mut() {
        while ns.len_of(Axis(0)) > ch {
            ns.remove_index(Axis(0), rng.uniform(0, ns.len_of(Axis(0))))
        }
        while ns.len_of(Axis(0)) < ch {
            let r = rng.uniform(0, ns.len_of(Axis(0)));
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
/// * `clean_distorted` - An optional distorted speech signal of shape `[C, N]`. If provided, this signal
///                  will be used for creating the noisy mixture. `clean` may be used as a training
///                  target and usually contains no or less distortions. This can be used to learn
///                  some dereverberation or declipping.
/// * `noise` - A noise signal of shape `[C, N]`. Will be modified in place.
/// * `snr_db` - Signal to noise ratio in decibel used for mixing.
/// * `gain_db` - Gain to apply to the clean signal in decibel before mixing.
/// * `noise_resample`: Optional resample parameters which will be used to apply a low-pass via
///                     resampling to the noise signal. This may be used to make sure a speech
///                     signal with a lower sampling rate will also be mixed with noise having the
///                     same sampling rate.
fn mix_audio_signal(
    clean: Array2<f32>,
    clean_distorted: Option<Array2<f32>>,
    mut noise: Array2<f32>,
    snr_db: f32,
    gain_db: f32,
) -> Result<(Signal, Signal, Signal)> {
    // Apply gain to speech
    let g = 10f32.powf(gain_db / 20.);
    let mut clean_out = &clean * g;
    // clean_mix may contain distorted speech
    let clean_mix = clean_distorted.map(|c| &c * g).unwrap_or_else(|| clean_out.clone());
    // For energy calculation use clean speech to also consider direct-to-reverberant ratio
    noise *= mix_f(clean_out.view(), noise.view(), snr_db);
    let mut mixture = clean_mix + &noise;
    // Guard against clipping
    let max = &([&clean_out, &noise, &mixture].iter().map(|x| find_max_abs(x.iter())))
        .collect::<Option<Vec<f32>>>()
        .expect("Found NaN");
    let max = find_max(max).expect("Found NaN");
    if (max - 1.) > 1e-10 {
        let f = 1. / (max + 1e-10);
        clean_out *= f;
        noise *= f;
        mixture *= f;
    }
    Ok((clean_out, noise, mixture))
}

fn get_env<T, K>(var: K) -> Option<T>
where
    K: AsRef<OsStr>,
    T: FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    match env::var(var) {
        Ok(e) => Some(e.parse::<T>().expect("Failed to parse env {var}: {e}")),
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Once;

    use rstest::rstest;

    use super::*;
    use crate::util::seed_from_u64;
    use crate::wav_utils::*;

    static INIT: Once = Once::new();

    /// Setup function that is only run once, even if called multiple times.
    fn setup() {
        INIT.call_once(|| {
            let _ = env_logger::builder()
                // Include all events in tests
                .filter_module("df", log::LevelFilter::max())
                // Ensure events are captured by `cargo test`
                .is_test(true)
                // Ignore errors initializing the logger if tests race to configure it
                .try_init();
        });
    }

    fn calc_rms(x: &[f32]) -> f32 {
        let n = x.len() as f32;
        (x.iter().map(|x| x.powi(2)).sum::<f32>() * (1. / n)).sqrt()
    }
    /// Calculates the SNR for a given clean signal and a noisy mixture.
    ///
    /// Arguments
    ///
    /// * `y` - A clean signal iterator.
    /// * `v` - A noise signal iterator.
    ///
    /// `x = s + v`, where x is the resulting mixture.
    fn calc_snr<'a, I>(s: I, v: I) -> f32
    where
        I: IntoIterator<Item = &'a f32>,
    {
        let e_clean = s.into_iter().fold(0f32, |acc, x| acc + x.powi(2));
        let e_noise = v.into_iter().fold(0f32, |acc, x| acc + x.powi(2));
        10. * (e_clean / e_noise).log10()
    }
    /// Calculates the SNR for a given mixture signal and the noise signal.
    ///
    /// Arguments
    ///
    /// * `x` - A noisy mixture signal iterator.
    /// * `v` - A noise signal iterator.
    ///
    /// `x = s + v`, where `y` is the clean signal component.
    fn calc_snr_xv<'a, I>(x: I, v: I) -> f32
    where
        I: IntoIterator<Item = &'a f32>,
    {
        let mut e_clean = 0.;
        let mut e_noise = 0.;
        for (xx, xv) in x.into_iter().zip(v.into_iter()) {
            e_clean += (xx - xv).powi(2);
            e_noise += xv.powi(2);
        }
        10. * (e_clean / e_noise).log10()
    }
    /// Calculates the SNR for a given clean signal and a noise/clean mixture.
    ///
    /// Arguments
    ///
    /// * `s` - A clean signal iterator.
    /// * `x` - A noisy signal iterator.
    ///
    /// `x = s + v`, where v is the noise component.
    fn calc_snr_sx<'a, I>(s: I, x: I) -> f32
    where
        I: IntoIterator<Item = &'a f32>,
    {
        let mut e_clean = 0.;
        let mut e_noise = 0.;
        for (xs, xx) in s.into_iter().zip(x.into_iter()) {
            e_clean += xs.powi(2);
            e_noise += (xx - xs).powi(2);
        }
        10. * (e_clean / e_noise).log10()
    }
    #[inline]
    fn is_close(a: &[f32], b: &[f32], rtol: f32, atol: f32) -> Vec<bool> {
        // like numpy
        assert_eq!(a.len(), b.len());
        let mut out = vec![true; b.len()];
        for ((a_s, b_s), o) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *o = (a_s - b_s).abs() <= atol + rtol * b_s.abs()
        }
        out
    }
    fn hdf5_noise_keys<'a>() -> BTreeSet<&'a str> {
        BTreeSet::from([
            "assets_noise_freesound_573577.wav",
            "assets_noise_freesound_2530.wav",
        ])
    }

    #[test]
    pub fn test_hdf5_read_pcm() -> Result<()> {
        setup();
        seed_from_u64(0);
        let hdf5 = Hdf5Dataset::new("../assets/noise.hdf5")?;
        for key in hdf5.keys()?.iter() {
            dbg!(key);
            assert!(hdf5_noise_keys().contains(key.as_str()));
            let mut samples_raw =
                ReadWav::new(&str::replace(key, "assets_", "../assets/"))?.samples_arr2()?;
            dbg!(samples_raw.shape());
            assert_eq!(hdf5.sample_len(key)?, samples_raw.len_of(Axis(1)));
            samples_raw.slice_axis_inplace(Axis(0), Slice::from(0..1));
            let sample_hdf5 = hdf5.read(key)?;
            dbg!(sample_hdf5.shape());
            assert_eq!(sample_hdf5.shape(), samples_raw.shape());
            assert_eq!(sample_hdf5, samples_raw);
            assert!(dbg!(calc_snr_sx(samples_raw.iter(), sample_hdf5.iter())) > 100.);
        }
        Ok(())
    }
    #[test]
    pub fn test_hdf5_read_vorbis() -> Result<()> {
        setup();
        seed_from_u64(0);
        let hdf5 = Hdf5Dataset::new("../assets/noise_vorbis.hdf5")?;
        for key in hdf5.keys()?.iter() {
            dbg!(key);
            assert!(hdf5_noise_keys().contains(key.as_str()));
            let mut samples_raw =
                ReadWav::new(&str::replace(key, "assets_", "../assets/"))?.samples_arr2()?;
            dbg!(samples_raw.shape());
            assert_eq!(dbg!(hdf5.sample_len(key)?), samples_raw.len_of(Axis(1)));
            samples_raw.slice_axis_inplace(Axis(0), Slice::from(0..1));
            let sample_hdf5 = hdf5.read(key)?;
            dbg!(sample_hdf5.shape());
            assert_eq!(sample_hdf5.shape(), samples_raw.shape());
            assert!(dbg!(calc_snr_sx(samples_raw.iter(), sample_hdf5.iter())) > 25.);
            let filename = &str::replace(key, "assets_", "../out/").replace(".wav", "_vorbis.wav");
            write_wav_arr2(filename, sample_hdf5.view(), hdf5.sr.unwrap() as u32)?;
        }
        Ok(())
    }
    #[test]
    pub fn test_hdf5_read_flac() -> Result<()> {
        setup();
        seed_from_u64(0);
        let hdf5 = Hdf5Dataset::new("../assets/noise_flac.hdf5")?;
        for key in hdf5.keys()?.iter() {
            dbg!(key);
            assert!(hdf5_noise_keys().contains(key.as_str()));
            let mut samples_raw =
                ReadWav::new(&str::replace(key, "assets_", "../assets/"))?.samples_arr2()?;
            dbg!(samples_raw.shape());
            assert_eq!(hdf5.sample_len(key)?, samples_raw.len_of(Axis(1)));
            samples_raw.slice_axis_inplace(Axis(0), Slice::from(0..1));
            let sample_hdf5 = hdf5.read(key)?;
            dbg!(sample_hdf5.shape());
            assert_eq!(sample_hdf5.shape(), samples_raw.shape());
            assert_eq!(sample_hdf5, samples_raw);
            assert!(dbg!(calc_snr_sx(samples_raw.iter(), sample_hdf5.iter())) > 100.);
            let filename = &str::replace(key, "assets_", "../out/").replace(".wav", "_flac.wav");
            write_wav_arr2(filename, sample_hdf5.view(), hdf5.sr.unwrap() as u32)?;
        }
        Ok(())
    }
    #[rstest]
    #[case("../assets/noise.hdf5", "assets_noise_freesound_573577.wav", 3..4, 100.)]
    #[case("../assets/noise_flac.hdf5", "assets_noise_freesound_573577.wav", 3..4, 100.)]
    #[case("../assets/noise_vorbis.hdf5", "assets_noise_freesound_573577.wav", 3..4, 20.)]
    #[should_panic(expected = "snr")]
    #[case("../assets/noise_vorbis.hdf5", "assets_noise_freesound_573577.wav", 3..4, 40.)]
    #[should_panic(expected = "Slice end")]
    #[case("../assets/noise.hdf5", "assets_noise_freesound_573577.wav", 4..5, 0.)]
    #[should_panic(expected = "Slice end")]
    #[case("../assets/noise_flac.hdf5", "assets_noise_freesound_573577.wav", 4..5, 0.)]
    #[should_panic(expected = "Slice end")]
    #[case("../assets/noise_vorbis.hdf5", "assets_noise_freesound_573577.wav", 4..5, 0.)]
    // 2 channel sample
    #[case("../assets/noise.hdf5", "assets_noise_freesound_2530.wav", 1..4, 100.)]
    #[case("../assets/noise_flac.hdf5", "assets_noise_freesound_2530.wav", 1..4, 100.)]
    #[case("../assets/noise_vorbis.hdf5", "assets_noise_freesound_2530.wav", 1..4, 20.)]
    pub fn test_hdf5_slice(
        #[case] ds: &str,
        #[case] key: &str,
        #[case] r: Range<usize>,
        #[case] snr: f32,
    ) {
        setup();
        seed_from_u64(0);
        let hdf5 = Hdf5Dataset::new(ds).unwrap();
        // "assets_noise_freesound_573577.wav" has a length of approx 4.8s
        // "assets_noise_freesound_2530.wav" has a length of approx 34.2s and 2 channels
        let sr = hdf5.sr.unwrap();
        let r = r.start * sr..r.end * sr;
        dbg!(&r);
        let samples_raw = ReadWav::new(&str::replace(key, "assets_", "../assets/"))
            .unwrap()
            .samples_arr2()
            .unwrap()
            .slice_move(s![0..1, r.clone()]);
        let samples_hdf5 = hdf5.read_slc(key, r.clone()).unwrap();
        dbg!(samples_hdf5.shape(), samples_raw.shape());
        {
            // Write to disk for debugging
            let basen = &str::replace(key, "assets_", "../out/");
            let filename =
                &str::replace(basen, ".wav", &format!("_{}_{}_raw.wav", &r.start, &r.end));
            dbg!(hdf5.sample_len(key).unwrap());
            write_wav_arr2(filename, samples_raw.view(), hdf5.sr.unwrap() as u32).unwrap();
            let dsn =
                &str::replace(ds, "../assets/noise", "").replace('_', "").replace(".hdf5", "");
            let filename = &str::replace(
                basen,
                ".wav",
                &format!("_{}_{}_{}.wav", &r.start, &r.end, dsn),
            );
            dbg!(&filename);
            write_wav_arr2(filename, samples_raw.view(), hdf5.sr.unwrap() as u32).unwrap();
        }
        assert_eq!(samples_hdf5.shape(), samples_raw.shape());
        assert!(dbg!(calc_snr_sx(samples_raw.iter(), samples_hdf5.iter())) > snr);
    }
    #[test]
    pub fn test_mix_audio_signal() -> Result<()> {
        setup();
        seed_from_u64(0);
        let sr = 48_000;
        let n = sr;
        let clean = arr1(rng_uniform(n, -0.1, 0.1)?.as_slice()).into_shape([1, n])?;
        let noise = arr1(rng_uniform(n, -0.1, 0.1)?.as_slice()).into_shape([1, n])?;
        let gains = [-6., 0., 6.];
        let snrs = [-10., -5., 0., 5., 10., 20., 40.];
        let atol = 1e-4;
        for clean_rev in [None, Some(clean.clone())] {
            for gain in gains {
                for snr in snrs {
                    let (c, n, m) = mix_audio_signal(
                        clean.clone(),
                        clean_rev.clone(),
                        noise.clone(),
                        snr,
                        gain,
                    )?;
                    assert_eq!(&c + &n, m);
                    dbg!(clean_rev.is_some(), gain, snr);
                    // Input SNR of mixture
                    let snr_inp_m = calc_snr_xv(m.iter(), n.iter());
                    assert!(
                        (snr_inp_m - snr).abs() < atol,
                        "Input SNR does not match: {snr_inp_m}, {snr}"
                    );
                    // Target SNR between noise and target (clean) speech.
                    let snr_target_c = calc_snr(c.iter(), n.iter());
                    assert!(
                        (snr_target_c - snr).abs() < atol,
                        "Target SNR does not match: {snr_target_c}, {snr}",
                    );
                    // Test the SNR difference between input and target
                    assert!((snr_inp_m - snr_target_c).abs() < atol);
                }
            }
        }
        Ok(())
    }
    #[test]
    pub fn test_td_dataset() -> Result<()> {
        setup();
        seed_from_u64(0);
        let sr = 48_000;
        let dir = "../assets/";
        let cfg = DatasetConfigJson::open("../assets/dataset.cfg").unwrap();
        let mut ds = DatasetBuilder::new(dir, sr)
            .dataset(cfg.split_config(Split::Train))
            .bandwidth_extension(1.0)
            .build_td_dataset()
            .unwrap();
        ds.generate_keys(Some(0))?;
        let mut rng = thread_rng()?;
        let sample = ds.get_sample(rng.uniform(0, ds.len()), Some(0)).unwrap();
        write_wav_arr2(
            "../out/speech.wav",
            sample.speech.view().into_dimensionality()?,
            sr as u32,
        )
        .unwrap();
        write_wav_arr2(
            "../out/noisy.wav",
            sample.noisy.view().into_dimensionality()?,
            sr as u32,
        )
        .unwrap();
        Ok(())
    }
    #[test]
    pub fn test_fft_dataset() -> Result<()> {
        setup();
        seed_from_u64(0);
        let (sr, n_fft, n_hop) = (48_000, 1024, 512);
        let nb = 1;
        let dir = "../assets/";
        let cfg = DatasetConfigJson::open("../assets/dataset.cfg").unwrap();
        let mut ds = DatasetBuilder::new(dir, sr)
            .dataset(cfg.split_config(Split::Train))
            .df_params(n_fft, Some(n_hop), Some(nb), Some(1), Some(0.1))
            .build_fft_dataset()?;
        ds.generate_keys(Some(0))?;
        let mut rng = thread_rng()?;
        let mut sample = ds.get_sample(rng.uniform(0, ds.len()), Some(0)).unwrap();
        let mut state = DFState::new(sr, n_fft, n_hop, nb, 1);
        let speech = istft(
            sample.speech.view_mut().into_dimensionality().unwrap(),
            &mut state,
            false,
        );
        let noisy = istft(
            sample.noisy.view_mut().into_dimensionality().unwrap(),
            &mut state,
            false,
        );
        write_wav_arr2("../out/speech.wav", speech.view(), sr as u32).unwrap();
        write_wav_arr2("../out/noisy.wav", noisy.view(), sr as u32).unwrap();
        Ok(())
    }
}
