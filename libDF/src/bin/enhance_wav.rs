use std::{path::PathBuf, process::exit, time::Instant};

use anyhow::Result;
use clap::{Parser, ValueHint};
use df::{tract::*, transforms::resample, wav_utils::*};
use ndarray::{prelude::*, Axis};

#[cfg(all(
    not(windows),
    not(target_os = "android"),
    not(target_os = "macos"),
    not(target_os = "freebsd"),
    not(target_env = "musl"),
    not(target_arch = "riscv64"),
    feature = "use-jemalloc"
))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

/// Simple program to sample from a hd5 dataset directory
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to model tar.gz
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    model: Option<PathBuf>,
    /// Enable post-filter
    #[arg(long = "pf")]
    post_filter: bool,
    /// Post-filter beta. Higher beta results in stronger attenuation.
    #[arg(long = "pf-beta", default_value_t = 0.02)]
    post_filter_beta: f32,
    /// Compensate delay of STFT and model lookahead
    #[arg(short = 'D', long)]
    compensate_delay: bool,
    /// Attenuation limit in dB by mixing the enhanced signal with the noisy signal.
    /// An attenuation limit of 0 dB means no noise reduction will be performed, 100 dB means full
    /// noise reduction, i.e. no attenuation limit.
    #[arg(short, long, default_value_t = 100.)]
    atten_lim_db: f32,
    /// Min dB local SNR threshold for running the decoder DNN side
    #[arg(long, value_parser, allow_negative_numbers = true, default_value_t = -15.)]
    min_db_thresh: f32,
    /// Max dB local SNR threshold for running ERB decoder
    #[arg(
        long,
        value_parser,
        allow_negative_numbers = true,
        default_value_t = 35.
    )]
    max_db_erb_thresh: f32,
    /// Max dB local SNR threshold for running DF decoder
    #[arg(
        long,
        value_parser,
        allow_negative_numbers = true,
        default_value_t = 35.
    )]
    max_db_df_thresh: f32,
    /// If used with multiple channels, reduce the mask with max (1) or mean (2)
    #[arg(long, value_parser, default_value_t = 1)]
    reduce_mask: i32,
    /// Logging verbosity
    #[arg(
        long,
        short = 'v',
        action = clap::ArgAction::Count,
        global = true,
        help = "Increase logging verbosity with multiple `-vv`",
    )]
    verbose: u8,
    // Output directory with enhanced audio files. Defaults to 'out'
    #[arg(short, long, default_value = "out", value_hint = ValueHint::DirPath)]
    output_dir: PathBuf,
    // Audio files
    #[arg(required = true)]
    files: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let level = match args.verbose {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    let tract_level = match args.verbose {
        0..=3 => log::LevelFilter::Error,
        4 => log::LevelFilter::Info,
        5 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    env_logger::Builder::from_env(env_logger::Env::default())
        .filter_level(level)
        .filter_module("tract_onnx", tract_level)
        .filter_module("tract_hir", tract_level)
        .filter_module("tract_core", tract_level)
        .filter_module("tract_linalg", tract_level)
        .init();

    // Initialize with 1 channel
    let mut r_params = RuntimeParams::default();
    r_params = r_params.with_atten_lim(args.atten_lim_db).with_thresholds(
        args.min_db_thresh,
        args.max_db_erb_thresh,
        args.max_db_df_thresh,
    );
    if args.post_filter {
        r_params = r_params.with_post_filter(args.post_filter_beta);
    }
    if let Ok(red) = args.reduce_mask.try_into() {
        r_params = r_params.with_mask_reduce(red);
    } else {
        log::warn!("Input not valid for `reduce_mask`.")
    }
    let df_params = if let Some(tar) = args.model.as_ref() {
        match DfParams::new(tar.clone()) {
            Ok(p) => p,
            Err(e) => {
                log::error!("Error opening model {}: {}", tar.display(), e);
                exit(1)
            }
        }
    } else if cfg!(any(feature = "default-model", feature = "default-model-ll")) {
        DfParams::default()
    } else {
        log::error!("deep-filter was not compiled with a default model. Please provide a model via '--model <path-to-model.tar.gz>'");
        exit(2)
    };
    let mut model: DfTract = DfTract::new(df_params.clone(), &r_params)?;
    let mut sr = model.sr;
    let mut delay = model.fft_size - model.hop_size; // STFT delay
    delay += model.lookahead * model.hop_size; // Add model latency due to lookahead
    if !args.output_dir.is_dir() {
        log::info!("Creating output directory: {}", args.output_dir.display());
        std::fs::create_dir_all(args.output_dir.clone())?
    }
    for file in args.files {
        let reader = ReadWav::new(file.to_str().unwrap())?;
        // Check if we need to adjust to multiple channels
        if r_params.n_ch != reader.channels {
            r_params.n_ch = reader.channels;
            model = DfTract::new(df_params.clone(), &r_params)?;
            sr = model.sr;
        }
        let sample_sr = reader.sr;
        let mut noisy = reader.samples_arr2()?;
        if sr != sample_sr {
            noisy = resample(noisy.view(), sample_sr, sr, None).expect("Error during resample()");
        }
        let noisy = noisy.as_standard_layout();
        let mut enh: Array2<f32> = ArrayD::default(noisy.shape()).into_dimensionality()?;
        let t0 = Instant::now();
        for (ns_f, enh_f) in noisy
            .view()
            .axis_chunks_iter(Axis(1), model.hop_size)
            .zip(enh.view_mut().axis_chunks_iter_mut(Axis(1), model.hop_size))
        {
            if ns_f.len_of(Axis(1)) < model.hop_size {
                break;
            }
            model.process(ns_f, enh_f)?;
        }
        let elapsed = t0.elapsed().as_secs_f32();
        let t_audio = noisy.len_of(Axis(1)) as f32 / sr as f32;
        log::info!(
            "Enhanced audio file {} in {:.2} (RTF: {})",
            file.display(),
            elapsed,
            elapsed / t_audio
        );
        let mut enh_file = args.output_dir.clone();
        enh_file.push(file.file_name().unwrap());
        if args.compensate_delay {
            enh.slice_axis_inplace(Axis(1), ndarray::Slice::from(delay..));
        }
        if sr != sample_sr {
            enh = resample(enh.view(), sr, sample_sr, None).expect("Error during resample()");
        }
        write_wav_arr2(enh_file.to_str().unwrap(), enh.view(), sample_sr as u32)?;
    }

    Ok(())
}
