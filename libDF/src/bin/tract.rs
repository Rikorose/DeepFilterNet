use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{anyhow, Result};
use clap::Parser;
use df::{wav_utils::*, zip3, zip5, Complex32, DFState};
use ini::Ini;
use ndarray::{prelude::*, Axis};
use tract_core::internal::tract_smallvec::alloc::collections::VecDeque;
use tract_core::prelude::*;
use tract_onnx::{prelude::*, tract_hir::shapefactoid};
use tract_pulse::{internal::ToDim, model::*};

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

pub struct DfModelParams {
    enc: PathBuf,
    erb_dec: PathBuf,
    df_dec: PathBuf,
}
impl DfModelParams {
    fn new(enc: PathBuf, erb_dec: PathBuf, df_dec: PathBuf) -> Self {
        Self {
            enc,
            erb_dec,
            df_dec,
        }
    }
}

pub struct DfParams {
    config: PathBuf,
    pub n_ch: usize,
    post_filter: bool,
    min_db_thresh: f32,
    max_db_erb_thresh: f32,
    max_db_df_thresh: f32,
}
impl DfParams {
    fn new(
        config: PathBuf,
        n_ch: usize,
        post_filter: bool,
        min_db_thresh: f32,
        max_db_erb_thresh: f32,
        max_db_df_thresh: f32,
    ) -> Self {
        Self {
            config,
            n_ch,
            post_filter,
            min_db_thresh,
            max_db_erb_thresh,
            max_db_df_thresh,
        }
    }
}

pub type TractModel = TypedSimpleState<TypedModel, TypedSimplePlan<TypedModel>>;

pub struct DfTract {
    enc: TractModel,
    erb_dec: TractModel,
    df_dec: TractModel,
    df_lookahead: usize,
    conv_lookahead: usize,
    pub sr: usize,
    pub ch: usize,
    fft_size: usize,
    hop_size: usize,
    nb_erb: usize,
    min_nb_erb_freqs: usize,
    nb_df: usize,
    n_freqs: usize,
    df_order: usize,
    alpha: f32,
    post_filter: bool,
    min_db_thresh: f32,
    max_db_erb_thresh: f32,
    max_db_df_thresh: f32,
    df_states: Vec<DFState>,
    spec_buf: Tensor,
    erb_buf: Tensor,
    cplx_buf: Tensor,
    m_zeros: Vec<f32>,
    rolling_spec_buf: VecDeque<Tensor>,
}

impl DfTract {
    pub fn new(models: &DfModelParams, params: &DfParams) -> Result<Self> {
        if !models.enc.is_file() {
            return Err(anyhow!("Encoder file not found"));
        }
        if !models.erb_dec.is_file() {
            return Err(anyhow!("ERB decoder file not found"));
        }
        if !models.df_dec.is_file() {
            return Err(anyhow!("DF decoder file not found"));
        }
        if !params.config.is_file() {
            return Err(anyhow!("Config file not found"));
        }
        let config = Ini::load_from_file(params.config.as_path())?;
        let model_cfg = config.section(Some("deepfilternet")).unwrap();
        let df_cfg = config.section(Some("df")).unwrap();

        let ch = params.n_ch;
        let enc = SimpleState::new(init_encoder(models.enc.as_path(), df_cfg, ch)?)?;
        let erb_dec = SimpleState::new(init_erb_decoder(
            models.erb_dec.as_path(),
            model_cfg,
            df_cfg,
            ch,
        )?)?;
        let (df_dec, _df_init_delay) =
            init_df_decoder(models.df_dec.as_path(), model_cfg, df_cfg, ch)?;
        let df_dec = SimpleState::new(df_dec)?;

        let sr = df_cfg.get("sr").unwrap().parse::<usize>()?;
        let hop_size = df_cfg.get("hop_size").unwrap().parse::<usize>()?;
        let fft_size = df_cfg.get("fft_size").unwrap().parse::<usize>()?;
        let min_nb_erb_freqs = df_cfg.get("min_nb_erb_freqs").unwrap().parse::<usize>()?;
        let nb_erb = df_cfg.get("nb_erb").unwrap().parse::<usize>()?;
        let nb_df = df_cfg.get("nb_df").unwrap().parse::<usize>()?;
        let df_order = df_cfg
            .get("df_order")
            .unwrap_or_else(|| model_cfg.get("df_order").unwrap())
            .parse::<usize>()?;
        let df_lookahead = df_cfg
            .get("df_lookahead")
            .unwrap_or_else(|| model_cfg.get("df_lookahead").unwrap())
            .parse::<usize>()?
            .max(0);
        // TODO: Why do I need 2 frames of delay here?
        let conv_lookahead = model_cfg.get("conv_lookahead").unwrap().parse::<usize>()?.max(2);
        let n_freqs = fft_size / 2 + 1;
        let alpha = if let Some(a) = df_cfg.get("norm_alpha") {
            a.parse::<f32>()?
        } else {
            let tau = df_cfg.get("norm_tau").unwrap().parse::<f32>()?;
            calc_norm_alpha(sr, hop_size, tau)
        };
        let spec_shape = [1, 1, 1, n_freqs, 2];
        let spec_buf = unsafe { Tensor::uninitialized_dt(f32::datum_type(), &spec_shape)? };
        let erb_buf = unsafe { Tensor::uninitialized_dt(f32::datum_type(), &[1, 1, 1, nb_erb])? };
        let cplx_buf = unsafe { Tensor::uninitialized_dt(f32::datum_type(), &[1, 1, nb_df, 2])? };
        // let mut cplx_buf_t = unsafe { Tensor::uninitialized_dt(f32::datum_type(), &[1, 2, 1, nb_df])? };
        let m_zeros = vec![0.; nb_erb];

        let rolling_spec_buf = VecDeque::with_capacity(df_order + df_lookahead + conv_lookahead);

        let mut state = DFState::new(sr as usize, fft_size, hop_size, nb_erb, min_nb_erb_freqs);
        state.init_norm_states(nb_df);
        let df_states = vec![state];

        let mut m = Self {
            enc,
            erb_dec,
            df_dec,
            conv_lookahead,
            df_lookahead,
            sr,
            ch,
            fft_size,
            hop_size,
            nb_erb,
            min_nb_erb_freqs,
            nb_df,
            n_freqs,
            df_order,
            alpha,
            min_db_thresh: params.min_db_thresh,
            max_db_erb_thresh: params.max_db_erb_thresh,
            max_db_df_thresh: params.max_db_df_thresh,
            spec_buf,
            erb_buf,
            cplx_buf,
            m_zeros,
            rolling_spec_buf,
            df_states,
            post_filter: params.post_filter,
        };
        m.init()?;

        Ok(m)
    }

    pub fn init(&mut self) -> Result<()> {
        let ch = self.ch;
        let spec_shape = [ch, 1, 1, self.n_freqs, 2];
        self.rolling_spec_buf.clear();
        for _ in 0..(self.df_order + self.df_lookahead + self.conv_lookahead) {
            self.rolling_spec_buf
                .push_front(tensor0(0f32).broadcast_scalar_to_shape(&spec_shape)?);
        }
        if ch > self.df_states.len() {
            for _ in self.df_states.len()..ch {
                let mut state = DFState::new(
                    self.sr,
                    self.fft_size,
                    self.hop_size,
                    self.nb_erb,
                    self.min_nb_erb_freqs,
                );
                state.init_norm_states(self.nb_df);
                self.df_states.push(state)
            }
        }
        self.spec_buf = unsafe { Tensor::uninitialized_dt(f32::datum_type(), &spec_shape)? };
        self.erb_buf =
            unsafe { Tensor::uninitialized_dt(f32::datum_type(), &[ch, 1, 1, self.nb_erb])? };
        self.cplx_buf =
            unsafe { Tensor::uninitialized_dt(f32::datum_type(), &[ch, 1, self.nb_df, 2])? };
        Ok(())
    }

    pub fn process(&mut self, noisy: ArrayView2<f32>, mut enh: ArrayViewMut2<f32>) -> Result<()> {
        let ch = noisy.len_of(Axis(0));
        debug_assert_eq!(noisy.len_of(Axis(0)), enh.len_of(Axis(0)));
        debug_assert_eq!(noisy.len_of(Axis(1)), enh.len_of(Axis(1)));
        debug_assert_eq!(noisy.len_of(Axis(1)), self.hop_size);

        self.rolling_spec_buf.rotate_left(1);
        for (ns_ch, mut rbuf, mut erb_ch, mut cplx_ch, state) in zip5(
            noisy.axis_iter(Axis(0)),
            self.rolling_spec_buf
                .back_mut()
                .unwrap()
                .to_array_view_mut()?
                .axis_iter_mut(Axis(0)),
            self.erb_buf.to_array_view_mut()?.axis_iter_mut(Axis(0)),
            self.cplx_buf.to_array_view_mut()?.axis_iter_mut(Axis(0)),
            self.df_states.iter_mut(),
        ) {
            let spec = slice_as_mut_complex(rbuf.as_slice_mut().unwrap());
            state.analysis(ns_ch.as_slice().unwrap(), spec);
            state.feat_erb(spec, self.alpha, erb_ch.as_slice_mut().unwrap());
            // TODO: Workaround transpose by directly computing transposed complex features:
            // state.feat_cplx_t(&spec[..nb_df], alpha, cplx_buf_t.as_slice_mut()?);
            state.feat_cplx(
                &spec[..self.nb_df],
                self.alpha,
                slice_as_mut_complex(cplx_ch.as_slice_mut().unwrap()),
            );
        }

        // Run encoder
        let mut enc_emb = self.enc.run(tvec!(
            self.erb_buf.clone(),
            self.cplx_buf.clone().permute_axes(&[0, 3, 1, 2])?
        ))?;
        let &lsnr = enc_emb.pop().unwrap().to_scalar::<f32>()?;
        let c0 = enc_emb.pop().unwrap().into_tensor();
        let emb = enc_emb.pop().unwrap().into_tensor();
        //         let (apply_erb, apply_df) = match lsnr {
        // (self.min_db_thresh..self.max_db_erb_thresh).contains(&lsnr) =>
        //         }
        let (apply_erb, apply_df) = if lsnr < self.min_db_thresh || lsnr > self.max_db_erb_thresh {
            (false, false)
        } else if lsnr > self.max_db_df_thresh {
            (true, false)
        } else {
            (true, true)
        };
        let (run_erb, run_df) = (true, true); // for now

        let mut spec =
            self.rolling_spec_buf.get_mut(self.df_order - 1).unwrap().to_array_view_mut()?;
        let mut m = if run_erb {
            let dec_input = tvec!(
                emb.clone(),
                enc_emb.pop().unwrap().into_tensor(), // e3
                enc_emb.pop().unwrap().into_tensor(), // e2
                enc_emb.pop().unwrap().into_tensor(), // e1
                enc_emb.pop().unwrap().into_tensor(), // e0
            );
            if apply_erb {
                let mut m = self.erb_dec.run(dec_input)?;
                let m = m.pop().unwrap().into_tensor().into_shape(&[self.ch, self.nb_erb])?;
                Some(m.into_array()?.mean_axis(Axis(0)).unwrap())
            } else {
                None
            }
        } else {
            None
        };
        let pf = if apply_erb { self.post_filter } else { false };
        let m = m
            .as_mut()
            .map(|x| x.as_slice_mut().unwrap())
            .unwrap_or(self.m_zeros.as_mut_slice());
        for (state, mut spec_ch) in self.df_states.iter().zip(spec.axis_iter_mut(Axis(0))) {
            state.apply_mask(slice_as_mut_complex(spec_ch.as_slice_mut().unwrap()), m, pf);
        }

        let spec = self.rolling_spec_buf.get_mut(self.df_order - 1 - self.df_lookahead).unwrap();
        if run_df {
            let coefs = self
                .df_dec
                .run(tvec!(emb, c0))?
                .pop()
                .unwrap()
                .into_tensor()
                .into_shape(&[ch, self.df_order, self.nb_df, 2])?;
            self.spec_buf.clone_from(spec);
            if apply_df {
                df(
                    &self.rolling_spec_buf,
                    coefs,
                    self.nb_df,
                    self.df_order,
                    self.n_freqs,
                    &mut self.spec_buf,
                )?;
            }
        }
        for (state, spec_ch, mut enh_ch) in zip3(
            self.df_states.iter_mut(),
            self.spec_buf.to_array_view_mut()?.axis_iter(Axis(0)),
            enh.axis_iter_mut(Axis(0)),
        ) {
            state.synthesis(
                slice_as_mut_complex(spec_ch.to_owned().as_slice_mut().unwrap()),
                enh_ch.as_slice_mut().unwrap(),
            );
        }
        Ok(())
    }
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

/// Deep Filtering.
///
/// Args:
///     - spec: Spectrogram buffer for the corresponding time steps. Needs to contain `df_order + df_lookahead + conv_lookahead` frames and applies DF to the oldest frames.
///     - coefs: Complex DF coefficients of shape `[C, N, F', 2]`, `N`: `df_order`, `F'`: `nb_df`
///     - nb_df: Number of DF frequency bins
///     - df_order: Deep Filtering order
///     - n_freqs: Number of FFT bins
///     - spec_out: Ouput buffer of shape `[C, F, 2]`, `F`: `n_freqs`
fn df(
    spec: &VecDeque<Tensor>,
    coefs: Tensor,
    nb_df: usize,
    df_order: usize,
    n_freqs: usize,
    spec_out: &mut Tensor,
) -> Result<()> {
    let ch = spec.back().unwrap().shape()[0];
    debug_assert_eq!(n_freqs, spec.back().unwrap().shape()[3]);
    debug_assert_eq!(n_freqs, spec_out.shape()[3]);
    debug_assert_eq!(ch, coefs.shape()[0]);
    debug_assert_eq!(df_order, coefs.shape()[1]);
    debug_assert_eq!(nb_df, coefs.shape()[2]);
    debug_assert_eq!(ch, spec_out.shape()[0]);
    let mut o_f: ArrayViewMut2<Complex32> =
        array_as_mut_complex(spec_out.to_array_view_mut::<f32>()?, &[ch, n_freqs])
            .into_dimensionality()?;
    // Zero relevant frequency bins of output
    o_f.slice_mut(s![.., ..nb_df]).fill(Complex32::default());
    let coefs_arr: ArrayView3<Complex32> =
        array_as_complex(coefs.to_array_view::<f32>()?, &[ch, df_order, nb_df])
            .into_dimensionality()?;
    // Transform spec to an complex array and iterate over time frames of spec and coefs
    let spec_iter = spec.iter().map(|s| {
        array_as_complex(s.to_array_view::<f32>().unwrap(), &[ch, n_freqs])
            .into_dimensionality::<Ix2>()
            .unwrap()
    });
    // Iterate over DF frames
    for (s_f, c_f) in spec_iter.zip(coefs_arr.axis_iter(Axis(1))) {
        // Iterate over channels
        for (s_ch, c_ch, mut o_ch) in zip3(s_f.outer_iter(), c_f.outer_iter(), o_f.outer_iter_mut())
        {
            // Apply DF for each frequency bin up to `nb_df`
            for (&s, &c, o) in zip3(s_ch, c_ch, o_ch.iter_mut()) {
                *o += s * c
            }
        }
    }
    Ok(())
}

fn init_encoder(
    m: &Path,
    df_cfg: &ini::Properties,
    n_ch: usize,
) -> Result<TypedSimplePlan<TypedModel>> {
    let s = tract_pulse::fact::stream_dim();

    let nb_erb = df_cfg.get("nb_erb").unwrap().parse::<usize>()?;
    let nb_df = df_cfg.get("nb_df").unwrap().parse::<usize>()?;
    let feat_erb = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, 1, s, nb_erb));
    let feat_spec = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, 2, s, nb_df));

    let mut m = tract_onnx::onnx()
        .model_for_path(m)?
        .with_input_fact(0, feat_erb)?
        .with_input_fact(1, feat_spec)?;

    m = m
        .with_input_names(&["feat_erb", "feat_spec"])?
        .with_output_names(&["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"])?;

    m.analyse(true)?;
    let mut m = m.into_typed()?;

    m.declutter()?;
    let pulsed = PulsedModel::new(&m, 1)?;
    let delay = pulsed.output_fact(0)?.delay;
    log::info!("Init encoder with delay: {}", delay);
    let m = pulsed.into_typed()?.into_optimized()?.into_runnable()?;
    Ok(m)
}

fn init_erb_decoder(
    m: &Path,
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
    n_ch: usize,
) -> Result<TypedSimplePlan<TypedModel>> {
    let s = tract_pulse::fact::stream_dim();

    let nb_erb = df_cfg.get("nb_erb").unwrap().parse::<usize>()?;
    let n_hidden = net_cfg.get("emb_hidden_dim").unwrap().parse::<usize>()?;
    let layer_width = net_cfg.get("conv_ch").unwrap().parse::<usize>()?;

    let emb = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, s, n_hidden));
    let e3f = nb_erb / 4;
    let e3 = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, layer_width, s, e3f));
    let e2 = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, layer_width, s, e3f));
    let e1f = nb_erb / 2;
    let e1 = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, layer_width, s, e1f));
    let e0 = InferenceFact::dt_shape(
        f32::datum_type(),
        shapefactoid!(n_ch, layer_width, s, nb_erb),
    );

    let mut m = tract_onnx::onnx()
        .model_for_path(m)?
        .with_input_fact(0, emb)?
        .with_input_fact(1, e3)?
        .with_input_fact(2, e2)?
        .with_input_fact(3, e1)?
        .with_input_fact(4, e0)?;

    m = m
        .with_input_names(&["emb", "e3", "e2", "e1", "e0"])?
        .with_output_names(&["m"])?;

    m.analyse(true)?;
    let mut m = m.into_typed()?;

    m.declutter()?;
    let pulsed = PulsedModel::new(&m, 1)?;
    let delay = pulsed.output_fact(0)?.delay;
    log::info!("Init ERB decoder with delay: {}", delay);
    let m = pulsed.into_typed()?.into_optimized()?.into_runnable()?;
    Ok(m)
}

fn init_df_decoder(
    m: &Path,
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
    n_ch: usize,
) -> Result<(TypedSimplePlan<TypedModel>, usize)> {
    let s = tract_pulse::fact::stream_dim();

    let nb_df = df_cfg.get("nb_df").unwrap().parse::<usize>()?;
    let n_hidden = net_cfg.get("emb_hidden_dim").unwrap().parse::<usize>()?;
    let layer_width = net_cfg.get("conv_ch").unwrap().parse::<usize>()?;

    let c0 = InferenceFact::dt_shape(
        f32::datum_type(),
        shapefactoid!(n_ch, layer_width, s, nb_df),
    );
    let emb = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, s, n_hidden));

    let mut m = tract_onnx::onnx()
        .model_for_path(m)?
        .with_input_fact(0, emb)?
        .with_input_fact(1, c0)?;

    m = m.with_input_names(&["emb", "c0"])?.with_output_names(&["coefs"])?;

    m.analyse(true)?;
    let mut m = m.into_typed()?;

    m.declutter()?;
    let pulsed = PulsedModel::new(&m, 1)?;
    let delay = pulsed.output_fact(0)?.delay;
    log::info!("Init DF decoder with delay: {}", delay);
    let m = pulsed.into_typed()?.into_optimized()?.into_runnable()?;
    Ok((m, delay))
}

fn calc_norm_alpha(sr: usize, hop_size: usize, tau: f32) -> f32 {
    let dt = hop_size as f32 / sr as f32;
    let alpha = f32::exp(-dt / tau);
    let mut a = 1.0;
    let mut precision = 3;
    while a >= 1.0 {
        a = (alpha * 10i32.pow(precision) as f32).round() / 10i32.pow(precision) as f32;
        precision += 1;
    }
    a
}

pub fn slice_as_mut_complex(buffer: &mut [f32]) -> &mut [Complex32] {
    unsafe {
        let ptr = buffer.as_ptr() as *mut Complex32;
        let len = buffer.len();
        std::slice::from_raw_parts_mut(ptr, len / 2)
    }
}

pub fn slice_as_mut_real(buffer: &mut [Complex32]) -> &mut [f32] {
    unsafe {
        let ptr = buffer.as_ptr() as *mut f32;
        let len = buffer.len();
        std::slice::from_raw_parts_mut(ptr, len * 2)
    }
}

pub fn array_as_complex<'a>(
    buffer: ArrayViewD<'a, f32>,
    shape: &[usize], // having an explicit shape parameter allows to also squeeze axes.
) -> ArrayViewD<'a, Complex32> {
    debug_assert_eq!(buffer.shape().last().unwrap(), &2);
    debug_assert_eq!(buffer.len(), 2 * shape.iter().product::<usize>());
    unsafe {
        let ptr = buffer.as_ptr() as *mut Complex32;
        ArrayViewD::from_shape_ptr(shape, ptr)
    }
}
pub fn array_as_mut_complex<'a>(
    buffer: ArrayViewMutD<'a, f32>,
    shape: &[usize], // having an explicit shape parameter allows to also squeeze axes.
) -> ArrayViewMutD<'a, Complex32> {
    debug_assert_eq!(buffer.shape().last().unwrap(), &2);
    debug_assert_eq!(buffer.len(), 2 * shape.iter().product::<usize>());
    unsafe {
        let ptr = buffer.as_ptr() as *mut Complex32;
        ArrayViewMutD::from_shape_ptr(shape, ptr)
    }
}
