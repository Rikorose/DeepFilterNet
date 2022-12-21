use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::{io, io::Write};

use anyhow::Result;
use clap::{Parser, ValueEnum};
use df::{
    dataset::{Dataset, DatasetBuilder, DatasetConfigJson, Split},
    hdf5_key_cache::*,
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
#[command(author, version, about, long_about = None)]
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
    #[arg(long, value_enum)]
    split: Option<DsSplit>,
    /// Dataset indices
    #[arg(short, long)]
    idx: Vec<usize>,
    /// Number of samples to generate
    #[arg(short, long)]
    num: Option<usize>,
    /// Random seed
    #[arg(short, long)]
    seed: Option<u64>,
    /// Random seed
    #[arg(short, long)]
    epoch: Option<usize>,
    /// Randomize sampling
    #[arg(short, long)]
    randomize: bool,
    #[arg(short, long)]
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

    // Load configs
    let ds_cfg_path = args.ds_cfg.to_str().unwrap();
    let mut ds_cfg = DatasetConfigJson::open(ds_cfg_path).unwrap();
    load_hdf5_key_cache(ds_cfg_path, &mut ds_cfg);
    let ini = Ini::load_from_file(args.df_cfg)?;
    let df_cfg = ini.section(Some("df")).unwrap();
    let distortion_cfg = ini.section(Some("distortion")).unwrap();
    let train_cfg = ini.section(Some("train")).unwrap();

    let global_seed = args.seed.unwrap_or(train_cfg.get("seed").unwrap().parse::<u64>()?);
    seed_from_u64(global_seed);

    let split = match args.split.unwrap_or(DsSplit::Train) {
        DsSplit::Train => Split::Train,
        DsSplit::Valid => Split::Valid,
        DsSplit::Test => Split::Test,
    };

    // Setup variables and dataset
    let out_dir = args.out_dir.to_str().unwrap();
    if !Path::new(out_dir).is_dir() {
        fs::create_dir(out_dir)?;
    }
    let sr = df_cfg.get("sr").unwrap().parse::<usize>()?;
    let hop_size = df_cfg.get("hop_size").unwrap().parse::<usize>()?;
    let fft_size = df_cfg.get("fft_size").unwrap().parse::<usize>()?;
    let min_nb_erb_freqs = df_cfg.get("min_nb_erb_freqs").unwrap().parse::<usize>()?;
    let nb_erb = df_cfg.get("nb_erb").unwrap().parse::<usize>()?;
    let nb_df = df_cfg.get("nb_df").unwrap().parse::<usize>()?;
    let snrs = train_cfg
        .get("dataloader_snrs")
        .unwrap()
        .split(',')
        .map(|x| x.parse::<i8>().unwrap())
        .collect();
    let mut state = DFState::new(sr, fft_size, hop_size, nb_erb, min_nb_erb_freqs);
    let ds_dir = args.ds_dir.to_str().unwrap();
    let mut ds_builder = DatasetBuilder::new(ds_dir, sr).df_params(
        fft_size,
        Some(hop_size),
        Some(nb_erb),
        Some(nb_df),
        None,
    );
    ds_builder = ds_builder
        .seed(global_seed)
        .max_len(train_cfg.get("max_sample_len_s").unwrap().parse::<f32>()?)
        .snrs(snrs)
        .global_sample_factor(train_cfg.get("global_ds_sampling_f").unwrap().parse::<f32>()?)
        .prob_reverberation(distortion_cfg.get("p_reverb").unwrap().parse::<f32>()?)
        .clipping_distortion(distortion_cfg.get("p_clipping").unwrap().parse::<f32>()?)
        .zeroing_distortion(distortion_cfg.get("p_zeroing").unwrap().parse::<f32>()?)
        .interfer_distortion(distortion_cfg.get("p_interfer_sp").unwrap().parse::<f32>()?)
        .air_absorption_distortion(distortion_cfg.get("p_air_absorption").unwrap().parse::<f32>()?)
        .bandwidth_extension(distortion_cfg.get("p_bandwidth_ext").unwrap().parse::<f32>()?);
    if split == Split::Train {
        ds_builder = ds_builder.p_sample_full_speech(1.0);
    }
    log::info!("Opening dataset with config {}", args.ds_cfg.display());
    io::stdout().flush()?;
    let mut ds = ds_builder.dataset(ds_cfg.split_config(split)).build_fft_dataset()?;
    log::info!("Opened dataset with config {}", args.ds_cfg.display());
    for s in Split::iter() {
        fetch_hdf5_keys_from_ds(ds_dir, ds_cfg.get_mut(s), &ds);
    }
    write_hdf5_key_cache(ds_cfg_path, &ds_cfg);
    let epoch_seed = args.epoch.unwrap_or_default() as u64;
    ds.generate_keys(Some(epoch_seed))?;
    let mut rng = thread_rng().unwrap();
    let indices = {
        if !args.idx.is_empty() {
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
    for &idx in indices.iter() {
        log::info!("Loading sample {}", idx);
        let mut sample = ds.get_sample(idx, Some(epoch_seed + idx as u64)).unwrap();
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
