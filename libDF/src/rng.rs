use core::cell::UnsafeCell;
use std::cell::RefCell;
use std::rc::Rc;
use std::thread_local;

pub use rand::Rng;
use rand::RngCore;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use thiserror::Error;
type Result<T> = std::result::Result<T, RngError>;

#[derive(Error, Debug)]
pub enum RngError {
    #[error("Random seed is not initalized using seed_from_u64(x)")]
    SeedNotInitialized,
}

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
    fn try_fill_bytes(&mut self, slice: &mut [u8]) -> std::result::Result<(), rand::Error> {
        unsafe { (*self.rng.get()).try_fill_bytes(slice) }
    }
}

pub(crate) fn thread_rng() -> Result<SeededRng> {
    if !(SEEDED.with(|s| *s.borrow())) {
        return Err(RngError::SeedNotInitialized);
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
}
