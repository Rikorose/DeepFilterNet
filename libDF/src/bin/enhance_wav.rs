use std::{path::PathBuf, time::Instant};

use anyhow::Result;
use clap::Parser;
use df::{tract::*, wav_utils::*};
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
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Encoder onnx file
    onnx_enc: PathBuf,
    /// Erb decoder onnx file
    onnx_erb_dec: PathBuf,
    /// DF decoder onnx file
    onnx_df_dec: PathBuf,
    /// Model config file
    cfg: PathBuf,
    /// Enable postfilter
    #[clap(short, long)]
    post_filter: bool,
    /// Min dB local SNR threshold for running the decoder DNN side
    #[clap(long, value_parser, default_value_t=-15.)]
    min_db_thresh: f32,
    /// Max dB local SNR threshold for running ERB decoder
    #[clap(long, value_parser, default_value_t = 35.)]
    max_db_erb_thresh: f32,
    /// Max dB local SNR threshold for running DF decoder
    #[clap(long, value_parser, default_value_t = 35.)]
    max_db_df_thresh: f32,
    /// Logging verbosity
    #[clap(short, long)]
    verbose: bool,
    // Output directory with enhanced audio files. Defaults to 'out'
    #[clap(short, long, default_value = "out")]
    out_dir: PathBuf,
    // Audio files
    files: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let level = match args.verbose {
        true => log::LevelFilter::Debug,
        _ => log::LevelFilter::Info,
    };
    env_logger::builder().filter_level(level).init();

    // Initialize with 1 channel
    let models = DfModelParams::new(args.onnx_enc, args.onnx_erb_dec, args.onnx_df_dec);
    let mut params = DfParams::new(
        args.cfg,
        1,
        args.post_filter,
        args.min_db_thresh,
        args.max_db_erb_thresh,
        args.max_db_df_thresh,
    );
    let mut model: DfTract = DfTract::new(&models, &params)?;
    let mut sr = model.sr;
    assert!(args.out_dir.is_dir());
    for file in args.files {
        let reader = ReadWav::new(file.to_str().unwrap())?;
        // Check if we need to adjust to multiple channels
        if params.n_ch != reader.channels {
            params.n_ch = reader.channels;
            model = DfTract::new(&models, &params)?;
            sr = model.sr;
        }
        assert_eq!(sr, reader.sr);
        let noisy = reader.samples_arr2()?;
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
            model.process(ns_f, enh_f)?
        }
        let elapsed = t0.elapsed().as_secs_f32();
        let t_audio = noisy.len_of(Axis(1)) as f32 / sr as f32;
        println!(
            "Enhanced audio file {} in {:.2} (RTF: {})",
            file.display(),
            elapsed,
            elapsed / t_audio
        );
        let mut enh_file = args.out_dir.clone();
        enh_file.push(file.file_name().unwrap());
        write_wav_arr2(enh_file.to_str().unwrap(), enh.view(), sr as u32)?;
    }

    // let mut n_erb_pause: isize = 0;
    // let df_init_delay = -(df_init_delay as isize);
    // let mut n_df_frames: isize = df_init_delay;
    // let mut n_df_pause: isize = 0;
    // let min_f_t: isize = df_order as isize;

    Ok(())
}
