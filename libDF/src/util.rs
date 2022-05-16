use std::cell::{RefCell, UnsafeCell};
use std::rc::Rc;
use std::sync::Once;
use std::thread_local;

use crossbeam_channel::{unbounded, Receiver, Sender};
use log::{Level, Metadata, Record};
use ndarray_rand::rand::distributions::{
    uniform::{SampleUniform, Uniform},
    Distribution,
};
use ndarray_rand::rand::{Error as RandError, Rng, RngCore};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use thiserror::Error;

type Result<T> = std::result::Result<T, UtilsError>;

#[derive(Error, Debug)]
pub enum UtilsError {
    #[error("NaN detected")]
    NaN,
    #[error("Random seed is not initialized using seed_from_u64(x)")]
    SeedNotInitialized,
    #[error("Could not inititalize logger")]
    SetLoggerError(#[from] log::SetLoggerError),
}

static LOGGER_INIT: Once = Once::new();
pub(crate) struct SeededRng {
    rng: Rc<UnsafeCell<Xoshiro256PlusPlus>>,
}
thread_local!(
    static THREAD_SEEDED_RNG: Rc<UnsafeCell<Xoshiro256PlusPlus>> =
        Rc::new(UnsafeCell::new(Xoshiro256PlusPlus::seed_from_u64(0)));
    static SEEDED: RefCell<bool> = RefCell::new(false);
);

pub fn seed_from_u64(x: u64) {
    SEEDED.with(|s| s.replace(true));
    unsafe { THREAD_SEEDED_RNG.with(|rng| *rng.get() = Xoshiro256PlusPlus::seed_from_u64(x)) }
}

impl RngCore for SeededRng {
    fn next_u32(&mut self) -> u32 {
        unsafe { (*self.rng.get()).next_u32() }
    }
    fn next_u64(&mut self) -> u64 {
        unsafe { (*self.rng.get()).next_u64() }
    }
    fn fill_bytes(&mut self, slice: &mut [u8]) {
        unsafe { (*self.rng.get()).fill_bytes(slice) }
    }
    fn try_fill_bytes(&mut self, slice: &mut [u8]) -> std::result::Result<(), RandError> {
        unsafe { (*self.rng.get()).try_fill_bytes(slice) }
    }
}

pub(crate) fn thread_rng() -> Result<SeededRng> {
    if !(SEEDED.with(|s| *s.borrow())) {
        return Err(UtilsError::SeedNotInitialized);
    }
    Ok(SeededRng {
        rng: THREAD_SEEDED_RNG.with(|rng| rng.clone()),
    })
}

impl SeededRng {
    #[inline]
    pub fn log_uniform(&mut self, low: f32, high: f32) -> f32 {
        self.gen_range(low.ln()..=high.ln()).exp()
    }
    #[inline]
    pub fn uniform<T: SampleUniform + PartialOrd>(&mut self, low: T, high: T) -> T {
        if low >= high {
            low
        } else {
            self.gen_range(low..high)
        }
    }
    #[inline]
    pub fn uniform_inclusive<T: SampleUniform + PartialOrd>(&mut self, low: T, high: T) -> T {
        if low >= high {
            low
        } else {
            self.gen_range(low..=high)
        }
    }
}

pub(crate) fn rng_uniform<T>(n: usize, low: T, high: T) -> Result<Vec<T>>
where
    T: Default + Clone + SampleUniform,
{
    let mut rng = thread_rng()?;
    let mut v = vec![T::default(); n];
    let dist = Uniform::new_inclusive(low, high);
    for x in v.iter_mut() {
        *x = dist.sample(&mut rng);
    }
    Ok(v)
}

pub(crate) struct NonNan(f32);

impl NonNan {
    fn new(val: f32) -> Option<NonNan> {
        if val.is_nan() {
            None
        } else {
            Some(NonNan(val))
        }
    }
    fn get(&self) -> f32 {
        self.0
    }
}

pub(crate) fn find_max<'a, I>(vals: I) -> Result<f32>
where
    I: IntoIterator<Item = &'a f32>,
{
    vals.into_iter().try_fold(f32::MIN, |acc, v| {
        let nonnan: NonNan = match NonNan::new(*v) {
            None => return Err(UtilsError::NaN),
            Some(x) => x,
        };
        Ok(nonnan.get().max(acc))
    })
}

pub(crate) fn find_max_abs<'a, I>(vals: I) -> Result<f32>
where
    I: IntoIterator<Item = &'a f32>,
{
    vals.into_iter().try_fold(0., |acc, v| {
        let nonnan: NonNan = match NonNan::new(v.abs()) {
            None => return Err(UtilsError::NaN),
            Some(x) => x,
        };
        Ok(nonnan.get().max(acc))
    })
}

pub(crate) fn find_min<'a, I>(vals: I) -> Result<f32>
where
    I: IntoIterator<Item = &'a f32>,
{
    vals.into_iter().try_fold(f32::MAX, |acc, v| {
        let nonnan: NonNan = match NonNan::new(*v) {
            None => return Err(UtilsError::NaN),
            Some(x) => x,
        };
        Ok(nonnan.get().min(acc))
    })
}

pub(crate) fn find_min_abs<'a, I>(vals: I) -> Result<f32>
where
    I: IntoIterator<Item = &'a f32>,
{
    vals.into_iter().try_fold(0., |acc, v| {
        let nonnan: NonNan = match NonNan::new(v.abs()) {
            None => return Err(UtilsError::NaN),
            Some(x) => x,
        };
        Ok(nonnan.get().min(acc))
    })
}

pub(crate) fn median(x: &mut [f32]) -> f32 {
    x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = x.len() / 2;
    x[mid]
}

pub(crate) fn argmax<'a, I>(vals: I) -> Result<usize>
where
    I: IntoIterator<Item = &'a f32>,
{
    let mut index = 0;
    let mut high = std::f32::MIN;
    vals.into_iter().enumerate().for_each(|(i, v)| {
        if v > &high {
            high = *v;
            index = i;
        }
    });
    Ok(index)
}

pub(crate) fn argmax_abs<'a, I>(vals: I) -> Result<usize>
where
    I: IntoIterator<Item = &'a f32>,
{
    let mut index = 0;
    let mut high = std::f32::MIN;
    vals.into_iter().enumerate().for_each(|(i, v)| {
        if v > &high {
            high = v.abs();
            index = i;
        }
    });
    Ok(index)
}

pub type LogMessage = (Level, String, Option<String>, Option<u32>); // level, message, module, lineno
pub struct DfLogger {
    sender: Sender<LogMessage>,
    level: Level,
}

impl DfLogger {
    pub fn build(level: Level) -> (DfLogger, Receiver<LogMessage>) {
        let (sender, receiver) = unbounded();
        let logger = DfLogger { sender, level };
        (logger, receiver)
    }
}

impl log::Log for DfLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.level && metadata.target().starts_with("df:")
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            self.sender
                .send((
                    record.level(),
                    format!("{}", record.args()),
                    record.module_path().map(|f| f.replace("::reexport_dataset_modules:", "")),
                    record.line(),
                ))
                .unwrap_or_else(|_| {
                    println!("DfDataloader | {} | {}", record.level(), record.args())
                });
        }
    }

    fn flush(&self) {}
}

pub fn init_logger(logger: DfLogger) {
    LOGGER_INIT.call_once(|| {
        let level = logger.level;
        log::set_boxed_logger(Box::new(logger)).expect("Could not set logger");
        log::set_max_level(level.to_level_filter());
    });
}

#[test]
fn test_find_max_abs() -> Result<()> {
    let mut x = vec![vec![0f32; 10]; 1];
    x[0][2] = 3f32;
    x[0][5] = -10f32;
    let max = find_max_abs(x.iter().flatten())?;
    assert_eq!(max, 10.);
    Ok(())
}
