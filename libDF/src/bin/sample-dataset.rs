use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use df::{
    dataset::{Dataset, DatasetBuilder, DatasetConfigJson, Split},
    transforms::istft,
    util::{seed_from_u64, thread_rng},
    wav_utils::write_wav_iter,
    DFState,
};
use ndarray::Axis;
use rand::prelude::IteratorRandom;

/// Simple program to sample from a hd5 dataset directory
#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Dataset configuration file
    cfg: PathBuf,
    /// Dataset directory containing the hdf5 datasets
    ds_dir: PathBuf,
    /// Save directory for sampled output wavs
    out_dir: PathBuf,
    /// Dataset split
    #[clap(long, arg_enum)]
    split: Option<DsSplit>,
    /// Number of samples to generate
    #[clap(short, long)]
    num: Option<usize>,
    /// Random seed
    #[clap(short, long)]
    seed: Option<u64>,
    /// Randomize sampling
    #[clap(short, long)]
    randomize: bool,
    #[clap(short, long)]
    verbose: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum DsSplit {
    Train,
    Valid,
    Test,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let level = match args.verbose {
        true => log::LevelFilter::max(),
        _ => log::LevelFilter::Info,
    };
    env_logger::builder().filter_level(level).init();

    seed_from_u64(args.seed.unwrap_or(42));
    let cfg = DatasetConfigJson::open(args.cfg.to_str().unwrap()).unwrap();
    let out_dir = args.out_dir.to_str().unwrap();
    let sr = 48000;
    let n_fft = 960;
    let mut state = DFState::new(sr, n_fft, n_fft / 2, 1, 1);
    let mut ds_builder = DatasetBuilder::new(args.ds_dir.to_str().unwrap(), sr)
        .df_params(n_fft, None, None, None, None);
    ds_builder = ds_builder.p_sample_full_speech(1.0);
    ds_builder = ds_builder.clipping_distortion(1.0);
    ds_builder = ds_builder.prob_reverberation(1.0);
    ds_builder = ds_builder.air_absorption_distortion(1.0);
    ds_builder = ds_builder.bandwidth_extension(1.0);
    let split = match args.split.unwrap_or(DsSplit::Train) {
        DsSplit::Train => Split::Train,
        DsSplit::Valid => Split::Valid,
        DsSplit::Test => Split::Test,
    };
    let mut ds = ds_builder.dataset(cfg.split_config(split)).build_fft_dataset()?;
    ds.generate_keys()?;
    let n_samples = args.num.unwrap_or_else(|| ds.len());
    let mut rng = thread_rng().unwrap();
    let iter = match args.randomize {
        true => (0..ds.len()).choose_multiple(&mut rng, n_samples),
        false => (0..n_samples).collect(),
    };
    for idx in iter {
        log::info!("Loading sample {}", idx);
        let mut sample = ds.get_sample(idx, Some(0)).unwrap();
        let ch = sample.speech.len_of(Axis(0)) as u16;
        log::info!("Got sample with idx {}", sample.idx);
        let speech = istft(
            sample.speech.view_mut().into_dimensionality().unwrap(),
            &mut state,
            true,
        );
        write_wav_iter(
            &format!("{}/{}_speech.wav", out_dir, sample.idx),
            speech.iter(),
            sr as u32,
            ch,
        )?;
        let noisy = istft(
            sample.noisy.view_mut().into_dimensionality().unwrap(),
            &mut state,
            true,
        );
        write_wav_iter(
            &format!("{}/{}_noisy.wav", out_dir, sample.idx),
            noisy.iter(),
            sr as u32,
            ch,
        )?;
    }
    Ok(())
}
