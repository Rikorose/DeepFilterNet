use std::{collections::VecDeque, path::PathBuf, process::exit, time::Instant};

use anyhow::Result;
use clap::{Parser, ValueHint};
use df::{tract::*, wav_utils::*};
use hound::{SampleFormat, WavSpec, WavWriter};
use ndarray::{prelude::*, Axis};
use rubato::{FftFixedInOut, Resampler};

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

struct ChunkReader<'a> {
    channels: usize,
    total_frames: usize,
    emitted_frames: usize,
    samples: Box<dyn Iterator<Item = f32> + 'a>,
    resampler: Option<InputResampler>,
}

impl<'a> ChunkReader<'a> {
    fn new(reader: &'a mut ReadWav, target_sr: usize) -> Result<Self> {
        let channels = reader.channels;
        let sample_sr = reader.sr;
        let total_frames = if sample_sr == target_sr {
            reader.len
        } else {
            ((reader.len as f64) * target_sr as f64 / sample_sr as f64).ceil() as usize
        };
        let samples = reader.iter();
        let resampler = if sample_sr == target_sr {
            None
        } else {
            Some(InputResampler::new(sample_sr, target_sr, channels)?)
        };
        Ok(Self {
            channels,
            total_frames,
            emitted_frames: 0,
            samples,
            resampler,
        })
    }

    fn total_frames(&self) -> usize {
        self.total_frames
    }

    fn next_chunk(&mut self, mut chunk: ArrayViewMut2<f32>) -> Result<usize> {
        if self.emitted_frames >= self.total_frames {
            return Ok(0);
        }
        let chunk_capacity = chunk.len_of(Axis(1));
        let frames_remaining = self.total_frames - self.emitted_frames;
        let frames_limit = frames_remaining.min(chunk_capacity);
        let filled = if let Some(resampler) = self.resampler.as_mut() {
            resampler.fill_chunk(&mut self.samples, chunk.view_mut(), frames_limit, self.channels)?
        } else {
            fill_chunk_direct(&mut self.samples, chunk.view_mut(), frames_limit, self.channels)
        };
        self.emitted_frames += filled;
        if filled < chunk_capacity {
            chunk.slice_mut(s![.., filled..]).fill(0.0);
        }
        Ok(filled)
    }
}

fn fill_chunk_direct(
    samples: &mut Box<dyn Iterator<Item = f32> + '_>,
    mut chunk: ArrayViewMut2<f32>,
    frames_limit: usize,
    channels: usize,
) -> usize {
    let mut frames_filled = 0;
    while frames_filled < frames_limit {
        let mut frame_complete = true;
        for ch in 0..channels {
            match samples.next() {
                Some(sample) => chunk[[ch, frames_filled]] = sample,
                None => {
                    frame_complete = false;
                    break;
                }
            }
        }
        if !frame_complete {
            break;
        }
        frames_filled += 1;
    }
    frames_filled
}

fn read_samples_into(
    samples: &mut Box<dyn Iterator<Item = f32> + '_>,
    mut dst: ArrayViewMut2<f32>,
    frames: usize,
    channels: usize,
) -> usize {
    let mut frames_read = 0;
    while frames_read < frames {
        let mut frame_complete = true;
        for ch in 0..channels {
            match samples.next() {
                Some(sample) => dst[[ch, frames_read]] = sample,
                None => {
                    frame_complete = false;
                    break;
                }
            }
        }
        if !frame_complete {
            break;
        }
        frames_read += 1;
    }
    frames_read
}

struct InputResampler {
    resampler: FftFixedInOut<f32>,
    inbuf: Vec<Vec<f32>>,
    outbuf: Vec<Vec<f32>>,
    pending: Vec<VecDeque<f32>>,
    tmp: Array2<f32>,
    chunk_in: usize,
    finished: bool,
    flush_remaining: usize,
}

impl InputResampler {
    fn new(sample_sr: usize, target_sr: usize, channels: usize) -> Result<Self> {
        let chunk_size = 2048;
        let resampler = FftFixedInOut::<f32>::new(sample_sr, target_sr, chunk_size, channels)?;
        let chunk_in = resampler.input_frames_max();
        let inbuf = resampler.input_buffer_allocate(true);
        let outbuf = resampler.output_buffer_allocate(true);
        let pending = vec![VecDeque::new(); channels];
        let tmp = Array2::zeros((channels, chunk_in));
        Ok(Self {
            resampler,
            inbuf,
            outbuf,
            pending,
            tmp,
            chunk_in,
            finished: false,
            flush_remaining: 0,
        })
    }

    fn fill_chunk(
        &mut self,
        samples: &mut Box<dyn Iterator<Item = f32> + '_>,
        mut chunk: ArrayViewMut2<f32>,
        frames_limit: usize,
        channels: usize,
    ) -> Result<usize> {
        if frames_limit == 0 {
            return Ok(0);
        }
        let mut frames_filled = 0;
        while frames_filled < frames_limit {
            if self.pending.iter().all(|p| p.is_empty()) {
                if self.is_done() {
                    break;
                }
                self.generate(samples, channels)?;
                if self.pending.iter().all(|p| p.is_empty()) && self.is_done() {
                    break;
                }
            }
            let available = self.pending[0].len();
            if available == 0 {
                break;
            }
            let take = (frames_limit - frames_filled).min(available);
            for ch in 0..channels {
                for idx in 0..take {
                    let sample = self.pending[ch]
                        .pop_front()
                        .expect("pending channel length mismatch");
                    chunk[[ch, frames_filled + idx]] = sample;
                }
            }
            frames_filled += take;
        }
        Ok(frames_filled)
    }

    fn generate(
        &mut self,
        samples: &mut Box<dyn Iterator<Item = f32> + '_>,
        channels: usize,
    ) -> Result<()> {
        if self.finished {
            if self.flush_remaining == 0 {
                return Ok(());
            }
            self.tmp.fill(0.0);
            self.process_tmp()?;
            self.flush_remaining -= 1;
            return Ok(());
        }
        let frames_read = read_samples_into(samples, self.tmp.view_mut(), self.chunk_in, channels);
        if frames_read < self.chunk_in {
            self.tmp.slice_mut(s![.., frames_read..]).fill(0.0);
            self.finished = true;
            self.flush_remaining = 1;
        }
        self.process_tmp()?;
        Ok(())
    }

    fn process_tmp(&mut self) -> Result<()> {
        for (tmp_ch, buf_ch) in self
            .tmp
            .axis_iter(Axis(0))
            .zip(self.inbuf.iter_mut())
        {
            buf_ch.copy_from_slice(tmp_ch.as_slice().unwrap());
        }
        self.resampler
            .process_into_buffer(&self.inbuf, &mut self.outbuf, None)?;
        for (out_ch, pending_ch) in self.outbuf.iter().zip(self.pending.iter_mut()) {
            pending_ch.extend(out_ch.iter().copied());
        }
        Ok(())
    }

    fn is_done(&self) -> bool {
        self.finished && self.flush_remaining == 0 && self.pending.iter().all(|p| p.is_empty())
    }
}

struct OutputResampler {
    resampler: FftFixedInOut<f32>,
    inbuf: Vec<Vec<f32>>,
    outbuf: Vec<Vec<f32>>,
    input_queue: Vec<VecDeque<f32>>,
    pending: Vec<VecDeque<f32>>,
    chunk_in: usize,
    flushed: bool,
}

impl OutputResampler {
    fn new(model_sr: usize, output_sr: usize, channels: usize) -> Result<Self> {
        let chunk_size = 2048;
        let resampler = FftFixedInOut::<f32>::new(model_sr, output_sr, chunk_size, channels)?;
        let inbuf = resampler.input_buffer_allocate(true);
        let outbuf = resampler.output_buffer_allocate(true);
        Ok(Self {
            chunk_in: resampler.input_frames_max(),
            resampler,
            inbuf,
            outbuf,
            input_queue: vec![VecDeque::new(); channels],
            pending: vec![VecDeque::new(); channels],
            flushed: false,
        })
    }

    fn push_frames(&mut self, frames: ArrayView2<f32>) -> Result<()> {
        for (queue, ch_samples) in self.input_queue.iter_mut().zip(frames.axis_iter(Axis(0))) {
            queue.extend(ch_samples.iter().copied());
        }
        self.process_available()
    }

    fn process_available(&mut self) -> Result<()> {
        loop {
            let available = self
                .input_queue
                .get(0)
                .map(|q| q.len())
                .unwrap_or(0);
            if available < self.chunk_in {
                break;
            }
            for (queue, buf) in self.input_queue.iter_mut().zip(self.inbuf.iter_mut()) {
                for idx in 0..self.chunk_in {
                    buf[idx] = queue.pop_front().expect("input queue underflow");
                }
            }
            self.resampler
                .process_into_buffer(&self.inbuf, &mut self.outbuf, None)?;
            for (out_ch, pend) in self.outbuf.iter().zip(self.pending.iter_mut()) {
                pend.extend(out_ch.iter().copied());
            }
        }
        Ok(())
    }

    fn drain_to_writer(
        &mut self,
        writer: &mut WavWriter<std::io::BufWriter<std::fs::File>>,
        frames_budget: &mut usize,
        channels: usize,
    ) -> Result<()> {
        if *frames_budget == 0 {
            self.discard_pending();
            return Ok(());
        }
        loop {
            let available = self.pending.get(0).map(|p| p.len()).unwrap_or(0);
            if available == 0 {
                break;
            }
            let to_write = available.min(*frames_budget);
            if to_write == 0 {
                break;
            }
            for _ in 0..to_write {
                for ch in 0..channels {
                    let sample = self.pending[ch]
                        .pop_front()
                        .expect("pending channel underflow");
                    writer.write_sample(float_to_i16(sample))?;
                }
            }
            *frames_budget -= to_write;
            if *frames_budget == 0 {
                break;
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.flushed {
            return Ok(());
        }
        let remaining = self.input_queue.get(0).map(|q| q.len()).unwrap_or(0);
        if remaining > 0 {
            for (queue, buf) in self.input_queue.iter_mut().zip(self.inbuf.iter_mut()) {
                let mut idx = 0;
                while idx < remaining {
                    buf[idx] = queue.pop_front().expect("input queue underflow");
                    idx += 1;
                }
                for j in idx..self.chunk_in {
                    buf[j] = 0.0;
                }
            }
            self.resampler
                .process_into_buffer(&self.inbuf, &mut self.outbuf, None)?;
            for (out_ch, pend) in self.outbuf.iter().zip(self.pending.iter_mut()) {
                pend.extend(out_ch.iter().copied());
            }
        }
        self.inbuf.iter_mut().for_each(|buf| buf.fill(0.0));
        self.resampler
            .process_into_buffer(&self.inbuf, &mut self.outbuf, None)?;
        for (out_ch, pend) in self.outbuf.iter().zip(self.pending.iter_mut()) {
            pend.extend(out_ch.iter().copied());
        }
        self.flushed = true;
        Ok(())
    }

    fn discard_pending(&mut self) {
        for pend in self.pending.iter_mut() {
            pend.clear();
        }
    }
}

struct ChunkWriter {
    writer: WavWriter<std::io::BufWriter<std::fs::File>>,
    channels: usize,
    delay_remaining: usize,
    model_frames_remaining: usize,
    remaining_output_frames: usize,
    resampler: Option<OutputResampler>,
}

impl ChunkWriter {
    fn new(
        writer: WavWriter<std::io::BufWriter<std::fs::File>>,
        channels: usize,
        model_sr: usize,
        output_sr: usize,
        delay_samples: usize,
        total_model_frames: usize,
    ) -> Result<Self> {
        let frames_after_delay = total_model_frames.saturating_sub(delay_samples);
        let remaining_output_frames = if model_sr == output_sr {
            frames_after_delay
        } else {
            ((frames_after_delay as f64) * output_sr as f64 / model_sr as f64).ceil() as usize
        };
        let resampler = if model_sr == output_sr {
            None
        } else {
            Some(OutputResampler::new(model_sr, output_sr, channels)?)
        };
        Ok(Self {
            writer,
            channels,
            delay_remaining: delay_samples,
            model_frames_remaining: total_model_frames,
            remaining_output_frames,
            resampler,
        })
    }

    fn write_chunk(&mut self, chunk: ArrayView2<f32>, filled: usize) -> Result<()> {
        if filled == 0 || self.model_frames_remaining == 0 {
            return Ok(());
        }
        let frames_to_consume = filled.min(self.model_frames_remaining);
        self.model_frames_remaining -= frames_to_consume;
        let mut start = 0;
        let mut frames_to_process = frames_to_consume;
        if self.delay_remaining > 0 {
            let skip = self.delay_remaining.min(frames_to_process);
            self.delay_remaining -= skip;
            start += skip;
            frames_to_process -= skip;
        }
        if frames_to_process == 0 {
            return Ok(());
        }
        let slice = chunk.slice(s![.., start..start + frames_to_process]);
        if let Some(resampler) = self.resampler.as_mut() {
            resampler.push_frames(slice)?;
            resampler.drain_to_writer(
                &mut self.writer,
                &mut self.remaining_output_frames,
                self.channels,
            )?;
        } else {
            write_direct(
                &mut self.writer,
                slice,
                &mut self.remaining_output_frames,
                self.channels,
            )?;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        if let Some(resampler) = self.resampler.as_mut() {
            resampler.flush()?;
            resampler.drain_to_writer(
                &mut self.writer,
                &mut self.remaining_output_frames,
                self.channels,
            )?;
            if self.remaining_output_frames == 0 {
                resampler.discard_pending();
            }
        }
        self.writer.finalize()?;
        Ok(())
    }
}

fn write_direct(
    writer: &mut WavWriter<std::io::BufWriter<std::fs::File>>,
    chunk: ArrayView2<f32>,
    frames_budget: &mut usize,
    channels: usize,
) -> Result<()> {
    if *frames_budget == 0 {
        return Ok(());
    }
    let frames_to_write = chunk.len_of(Axis(1)).min(*frames_budget);
    for frame_idx in 0..frames_to_write {
        for ch in 0..channels {
            writer.write_sample(float_to_i16(chunk[[ch, frame_idx]]))?;
        }
    }
    *frames_budget -= frames_to_write;
    Ok(())
}

fn float_to_i16(sample: f32) -> i16 {
    let clipped = sample.clamp(-1.0, 1.0);
    (clipped * i16::MAX as f32) as i16
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
        let mut reader = ReadWav::new(file.to_str().unwrap())?;
        let channels = reader.channels;
        let sample_sr = reader.sr;
        let audio_len = reader.len;
        // Check if we need to adjust to multiple channels
        if r_params.n_ch != channels {
            r_params.n_ch = channels;
            model = DfTract::new(df_params.clone(), &r_params)?;
            sr = model.sr;
        }
        let mut chunk_reader = ChunkReader::new(&mut reader, sr)?;
        let total_model_frames = chunk_reader.total_frames();
        let mut enh_file = args.output_dir.clone();
        enh_file.push(file.file_name().unwrap());
        let delay_samples = if args.compensate_delay { delay } else { 0 };
        let spec = WavSpec {
            channels: channels as u16,
            sample_rate: sample_sr as u32,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let writer = WavWriter::create(enh_file, spec)?;
        let mut chunk_writer = ChunkWriter::new(
            writer,
            channels,
            sr,
            sample_sr,
            delay_samples,
            total_model_frames,
        )?;
        let mut chunk = Array2::<f32>::zeros((channels, model.hop_size));
        let mut enh = chunk.clone();
        let t0 = Instant::now();
        loop {
            let filled = chunk_reader.next_chunk(chunk.view_mut())?;
            if filled == 0 {
                break;
            }
            model.process(chunk.view(), enh.view_mut())?;
            chunk_writer.write_chunk(enh.view(), filled)?;
        }
        chunk_writer.finish()?;
        let elapsed = t0.elapsed().as_secs_f32();
        let t_audio = audio_len as f32 / sample_sr as f32;
        log::info!(
            "Enhanced audio file {} in {:.2} (RTF: {})",
            file.display(),
            elapsed,
            elapsed / t_audio
        );
    }

    Ok(())
}
