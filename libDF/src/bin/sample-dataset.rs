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
use ini::Ini;
use ndarray::Axis;
use rand::prelude::IteratorRandom;

/// Simple program to sample from a hd5 dataset directory
#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    df_cfg: PathBuf,
    /// Dataset directory containing the hdf5 datasets
    /// Dataset configuration file
    ds_cfg: PathBuf,
    /// DeepFilterNet framework configuration file
    ds_dir: PathBuf,
    /// Save directory for sampled output wavs
    out_dir: PathBuf,
    /// Dataset split
    #[clap(long, arg_enum)]
    split: Option<DsSplit>,
    /// Dataset indices
    #[clap(short, long)]
    idx: Vec<usize>,
    /// Number of samples to generate
    #[clap(short, long)]
    num: Option<usize>,
    /// Random seed
    #[clap(short, long)]
    seed: Option<u64>,
    /// Sample seed
    #[clap(long)]
    sample_seed: Vec<u64>,
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
    let global_seed = args.seed.unwrap_or(42);
    seed_from_u64(global_seed);

    // Load configs
    let ds_cfg = DatasetConfigJson::open(args.ds_cfg.to_str().unwrap()).unwrap();
    let ini = Ini::load_from_file(args.df_cfg)?;
    // let model_cfg = ini.section(Some("deepfilternet")).unwrap();
    let df_cfg = ini.section(Some("df")).unwrap();
    let distortion_cfg = ini.section(Some("distortion")).unwrap();
    let train_cfg = ini.section(Some("train")).unwrap();

    // Setup variables and dataset
    let out_dir = args.out_dir.to_str().unwrap();
    let sr = df_cfg.get("sr").unwrap().parse::<usize>()?;
    let hop_size = df_cfg.get("hop_size").unwrap().parse::<usize>()?;
    let fft_size = df_cfg.get("fft_size").unwrap().parse::<usize>()?;
    let min_nb_erb_freqs = df_cfg.get("min_nb_erb_freqs").unwrap().parse::<usize>()?;
    let nb_erb = df_cfg.get("nb_erb").unwrap().parse::<usize>()?;
    let nb_df = df_cfg.get("nb_df").unwrap().parse::<usize>()?;
    // let nb_freq_bins = fft_size / 2 + 1;
    let mut state = DFState::new(sr, fft_size, hop_size, nb_erb, min_nb_erb_freqs);
    let mut ds_builder = DatasetBuilder::new(args.ds_dir.to_str().unwrap(), sr).df_params(
        fft_size,
        Some(hop_size),
        Some(nb_erb),
        Some(nb_df),
        None,
    );
    ds_builder = ds_builder.seed(global_seed);
    ds_builder = ds_builder.max_len(train_cfg.get("max_sample_len_s").unwrap().parse::<f32>()?);
    ds_builder = ds_builder.p_sample_full_speech(1.0);
    ds_builder = ds_builder.prob_reverberation(train_cfg.get("p_reverb").unwrap().parse::<f32>()?);
    ds_builder =
        ds_builder.clipping_distortion(distortion_cfg.get("p_clipping").unwrap().parse::<f32>()?);
    ds_builder = ds_builder
        .air_absorption_distortion(distortion_cfg.get("p_air_absorption").unwrap().parse::<f32>()?);
    ds_builder = ds_builder
        .bandwidth_extension(distortion_cfg.get("p_bandwidth_ext").unwrap().parse::<f32>()?);
    let split = match args.split.unwrap_or(DsSplit::Train) {
        DsSplit::Train => Split::Train,
        DsSplit::Valid => Split::Valid,
        DsSplit::Test => Split::Test,
    };
    let mut ds = ds_builder.dataset(ds_cfg.split_config(split)).build_fft_dataset()?;
    ds.generate_keys()?;
    let mut rng = thread_rng().unwrap();
    let indices = {
        if args.idx.len() > 0 {
            args.idx
        } else {
            let n_samples = args.num.unwrap_or_else(|| ds.len());
            if args.randomize {
                (0..ds.len()).choose_multiple(&mut rng, n_samples)
            } else {
                (0..n_samples).collect()
            }
        }
    };
    let n_sample_seeds = args.sample_seed.len();
    if n_sample_seeds > 0 {
        assert_eq!(
            args.sample_seed.len(),
            indices.len(),
            "Number of samples and number of sample_seed does not match."
        );
    }
    let mut seeds: Vec<u64> = args.sample_seed;
    seeds.extend(std::iter::repeat(0u64).take(indices.len() - n_sample_seeds));
    for (&idx, &seed) in indices.iter().zip(seeds.iter()) {
        log::info!("Loading sample {}", idx);
        let mut sample = ds.get_sample(idx, Some(seed)).unwrap();
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
