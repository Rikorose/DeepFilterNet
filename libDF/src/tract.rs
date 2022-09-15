use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
#[cfg(feature = "timings")]
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use flate2::read::GzDecoder;
use ini::Ini;
use ndarray::{prelude::*, Axis};
use tar::Archive;
use tract_core::internal::tract_smallvec::alloc::collections::VecDeque;
use tract_core::prelude::*;
use tract_onnx::{prelude::*, tract_hir::shapefactoid};
use tract_pulse::{internal::ToDim, model::*};

use crate::*;

#[derive(Clone)]
pub struct DfParams {
    config: Ini,
    enc: Vec<u8>,
    erb_dec: Vec<u8>,
    df_dec: Vec<u8>,
}

fn extract_targz<R: Read>(f: R) -> Result<(Ini, Vec<u8>, Vec<u8>, Vec<u8>)> {
    let tar = GzDecoder::new(f);
    let mut archive = Archive::new(tar);
    let mut enc = Vec::new();
    let mut erb_dec = Vec::new();
    let mut df_dec = Vec::new();
    let mut cfg = Ini::new();
    for e in archive.entries().context("Could not extract models from tar file.")? {
        let mut file = e.context("Could not open model tar entry.")?;
        let path = file.path().unwrap();
        if path.ends_with("enc.onnx") {
            file.read_to_end(&mut enc)?;
        } else if path.ends_with("erb_dec.onnx") {
            file.read_to_end(&mut erb_dec)?;
        } else if path.ends_with("df_dec.onnx") {
            file.read_to_end(&mut df_dec)?;
        } else if path.ends_with("config.ini") {
            cfg = Ini::read_from(&mut file).context("Could not load config from tar file.")?;
        } else {
            log::warn!("Found non-matching item in model tar file: {:?}", path)
        }
    }
    Ok((cfg, enc, erb_dec, df_dec))
}

impl DfParams {
    pub fn new(tar_file: PathBuf) -> Result<Self> {
        let file = File::open(tar_file).context("Could not open model tar file.")?;
        let (config, enc, erb_dec, df_dec) = extract_targz(file)?;
        Ok(Self {
            config,
            enc,
            erb_dec,
            df_dec,
        })
    }
    pub fn from_bytes(tar_buf: &'static [u8]) -> Result<Self> {
        let (config, enc, erb_dec, df_dec) = extract_targz(tar_buf)?;
        Ok(Self {
            config,
            enc,
            erb_dec,
            df_dec,
        })
    }
}

#[derive(Clone)]
pub enum ReduceMask {
    MAX = 1,
    MEAN = 2,
}
impl TryFrom<i32> for ReduceMask {
    type Error = ();

    fn try_from(v: i32) -> Result<Self, Self::Error> {
        match v {
            x if x == ReduceMask::MAX as i32 => Ok(ReduceMask::MAX),
            x if x == ReduceMask::MEAN as i32 => Ok(ReduceMask::MEAN),
            _ => Err(()),
        }
    }
}
pub struct RuntimeParams {
    pub n_ch: usize,
    post_filter: bool,
    min_db_thresh: f32,
    max_db_erb_thresh: f32,
    max_db_df_thresh: f32,
    reduce_mask: ReduceMask,
}
impl RuntimeParams {
    pub fn new(
        n_ch: usize,
        post_filter: bool,
        min_db_thresh: f32,
        max_db_erb_thresh: f32,
        max_db_df_thresh: f32,
        reduce_mask: ReduceMask,
    ) -> Self {
        Self {
            n_ch,
            post_filter,
            min_db_thresh,
            max_db_erb_thresh,
            max_db_df_thresh,
            reduce_mask,
        }
    }
}

pub type TractModel = TypedSimpleState<TypedModel, TypedSimplePlan<TypedModel>>;

pub struct DfTract {
    enc: TractModel,
    erb_dec: TractModel,
    df_dec: TractModel,
    pub df_lookahead: usize,
    pub conv_lookahead: usize,
    pub sr: usize,
    pub ch: usize,
    pub fft_size: usize,
    pub hop_size: usize,
    pub nb_erb: usize,
    pub min_nb_erb_freqs: usize,
    pub nb_df: usize,
    pub n_freqs: usize,
    pub df_order: usize,
    pub post_filter: bool,
    pub alpha: f32,
    pub min_db_thresh: f32,
    pub max_db_erb_thresh: f32,
    pub max_db_df_thresh: f32,
    pub reduce_mask: ReduceMask,
    df_states: Vec<DFState>,
    spec_buf: Tensor,
    erb_buf: Tensor,
    cplx_buf: Tensor,
    m_zeros: Vec<f32>,
    rolling_spec_buf: VecDeque<Tensor>,
}

impl DfTract {
    pub fn new(dfp: DfParams, rp: &RuntimeParams) -> Result<Self> {
        #[cfg(feature = "timings")]
        let t0 = Instant::now();
        let config = dfp.config;
        let model_cfg = config.section(Some("deepfilternet")).unwrap();
        let df_cfg = config.section(Some("df")).unwrap();
        let ch = rp.n_ch;

        let (enc, enc_delay) = init_encoder_from_read(&mut Cursor::new(dfp.enc), df_cfg, ch)?;
        let (erb_dec, erb_dec_delay) =
            init_erb_decoder_from_read(&mut Cursor::new(dfp.erb_dec), model_cfg, df_cfg, ch)?;
        let (df_dec, df_dec_delay) =
            init_df_decoder_from_read(&mut Cursor::new(dfp.df_dec), model_cfg, df_cfg, ch)?;
        let enc = SimpleState::new(enc.into_runnable()?)?;
        let erb_dec = SimpleState::new(erb_dec.into_runnable()?)?;
        let df_dec = SimpleState::new(df_dec.into_runnable()?)?;
        #[cfg(feature = "timings")]
        let t1 = Instant::now();

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

        let model_type = config.section(Some("train")).unwrap().get("model").unwrap();
        let lookahead = match model_type {
            "deepfilternet2" => df_lookahead + conv_lookahead,
            "deepfilternet3" => df_lookahead.max(conv_lookahead),
            _ => return Err(anyhow!("Unsupported model type")),
        };
        log::info!(
            "Running with model type {} lookahead {}",
            model_type,
            lookahead,
        );

        let rolling_spec_buf = VecDeque::with_capacity(df_order + lookahead);

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
            min_db_thresh: rp.min_db_thresh,
            max_db_erb_thresh: rp.max_db_erb_thresh,
            max_db_df_thresh: rp.max_db_df_thresh,
            reduce_mask: rp.reduce_mask.clone(),
            spec_buf,
            erb_buf,
            cplx_buf,
            m_zeros,
            rolling_spec_buf,
            df_states,
            post_filter: rp.post_filter,
        };
        m.init()?;
        #[cfg(feature = "timings")]
        log::info!(
            "Init DfTract in {:.2}ms (models in {:.2}ms)",
            t0.elapsed().as_secs_f32() * 1000.,
            (t1 - t0).as_secs_f32() * 1000.
        );

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
        self.spec_buf = Tensor::zero::<f32>(&spec_shape)?;
        self.erb_buf = Tensor::zero::<f32>(&[ch, 1, 1, self.nb_erb])?;
        self.cplx_buf = Tensor::zero::<f32>(&[ch, 1, self.nb_df, 2])?;

        // for _ in 0..(enc_delay.max(erb_dec_delay.max(df_dec_delay))) {
        //     let mut enc_emb = self.enc.run(tvec!(
        //         self.erb_buf.clone(),
        //         self.cplx_buf.clone().permute_axes(&[0, 3, 1, 2])?
        //     ))?;
        //     dbg!(enc_emb.pop());
        //     let c0 = enc_emb.pop().unwrap().into_tensor();
        //     let cemb = enc_emb.pop().unwrap().into_tensor();
        //     let e3 = enc_emb.pop().unwrap().into_tensor();
        //     let e2 = enc_emb.pop().unwrap().into_tensor();
        //     let e1 = enc_emb.pop().unwrap().into_tensor();
        //     let e0 = enc_emb.pop().unwrap().into_tensor();

        //     self.erb_dec.run(tvec!(cemb.clone(), e3, e2, e1, e0))?;

        //     self.df_dec.run(tvec!(cemb.clone(), c0))?;
        // }

        Ok(())
    }

    pub fn process(&mut self, noisy: ArrayView2<f32>, mut enh: ArrayViewMut2<f32>) -> Result<()> {
        #[cfg(feature = "timings")]
        let t0 = Instant::now();
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
            let spec = as_slice_mut_complex(rbuf.as_slice_mut().unwrap());
            state.analysis(ns_ch.as_slice().unwrap(), spec);
            state.feat_erb(spec, self.alpha, erb_ch.as_slice_mut().unwrap());
            // TODO: Workaround transpose by directly computing transposed complex features:
            // state.feat_cplx_t(&spec[..nb_df], alpha, cplx_buf_t.as_slice_mut()?);
            state.feat_cplx(
                &spec[..self.nb_df],
                self.alpha,
                as_slice_mut_complex(cplx_ch.as_slice_mut().unwrap()),
            );
        }

        #[cfg(feature = "timings")]
        let t1 = Instant::now();
        // Run encoder
        let mut enc_emb = self.enc.run(tvec!(
            self.erb_buf.clone(),
            self.cplx_buf.clone().permute_axes(&[0, 3, 1, 2])?
        ))?;
        #[cfg(feature = "timings")]
        let t2 = Instant::now();
        let &lsnr = enc_emb.pop().unwrap().to_scalar::<f32>()?;
        let c0 = enc_emb.pop().unwrap().into_tensor();
        let emb = enc_emb.pop().unwrap().into_tensor();
        //         let (apply_erb, apply_df) = match lsnr {
        // (self.min_db_thresh..self.max_db_erb_thresh).contains(&lsnr) =>
        //         }
        let (apply_erb, apply_erb_zeros, apply_df) = if lsnr < self.min_db_thresh {
            (false, true, false)
        } else if lsnr > self.max_db_erb_thresh {
            (false, false, false)
        } else if lsnr > self.max_db_df_thresh {
            (true, false, false)
        } else {
            (true, false, true)
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
                let mut m = m
                    .pop()
                    .unwrap()
                    .into_tensor()
                    .into_shape(&[self.ch, self.nb_erb])?
                    .into_array()?;
                if self.ch > 1 {
                    m = match self.reduce_mask {
                        ReduceMask::MAX => m.fold_axis(Axis(0), 0., |&acc, &x| f32::max(x, acc)),
                        ReduceMask::MEAN => m.mean_axis(Axis(0)).unwrap(),
                    };
                }
                Some(m)
            } else {
                None
            }
        } else {
            None
        };
        if apply_erb || apply_erb_zeros {
            let pf = if apply_erb { self.post_filter } else { false };
            let m = m
                .as_mut()
                .map(|x| x.as_slice_mut().unwrap())
                .unwrap_or(self.m_zeros.as_mut_slice());
            for (state, mut spec_ch) in self.df_states.iter().zip(spec.axis_iter_mut(Axis(0))) {
                state.apply_mask(as_slice_mut_complex(spec_ch.as_slice_mut().unwrap()), m, pf);
            }
        }
        #[cfg(feature = "timings")]
        let t3 = Instant::now();

        let spec = self.rolling_spec_buf.get_mut(self.df_order - 1 - self.df_lookahead).unwrap();
        if run_df {
            let mut coefs = self.df_dec.run(tvec!(emb, c0))?.pop().unwrap().into_tensor();
            coefs.set_shape(&[ch, self.nb_df, self.df_order, 2])?;
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
        };
        #[cfg(feature = "timings")]
        let t4 = Instant::now();
        for (state, spec_ch, mut enh_ch) in zip3(
            self.df_states.iter_mut(),
            self.spec_buf.to_array_view_mut()?.axis_iter(Axis(0)),
            enh.axis_iter_mut(Axis(0)),
        ) {
            state.synthesis(
                as_slice_mut_complex(spec_ch.to_owned().as_slice_mut().unwrap()),
                enh_ch.as_slice_mut().unwrap(),
            );
        }
        #[cfg(feature = "timings")]
        log::info!(
            "Processed frame in {:.2}ms (analysis: {:.2}ms, encoder: {:.2}ms, erb_decoder: {:.2}ms, df_decoder: {:.2}ms, synthesis: {:.2}ms)",
            t0.elapsed().as_secs_f32() * 1000.,
            (t1 - t0).as_secs_f32() * 1000.,
            (t2 - t1).as_secs_f32() * 1000.,
            (t3 - t2).as_secs_f32() * 1000.,
            (t4 - t3).as_secs_f32() * 1000.,
            t4.elapsed().as_secs_f32() * 1000.,
        );
        Ok(())
    }
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
    debug_assert_eq!(nb_df, coefs.shape()[1]);
    debug_assert_eq!(df_order, coefs.shape()[2]);
    debug_assert_eq!(ch, spec_out.shape()[0]);
    let mut o_f: ArrayViewMut2<Complex32> =
        as_array_mut_complex(spec_out.to_array_view_mut::<f32>()?, &[ch, n_freqs])
            .into_dimensionality()?;
    // Zero relevant frequency bins of output
    o_f.slice_mut(s![.., ..nb_df]).fill(Complex32::default());
    let coefs_arr: ArrayView3<Complex32> =
        as_arrayview_complex(coefs.to_array_view::<f32>()?, &[ch, nb_df, df_order])
            .into_dimensionality()?;
    // Transform spec to an complex array and iterate over time frames of spec and coefs
    let spec_iter = spec.iter().map(|s| {
        as_arrayview_complex(s.to_array_view::<f32>().unwrap(), &[ch, n_freqs])
            .into_dimensionality::<Ix2>()
            .unwrap()
    });
    // Iterate over DF frames
    for (s_f, c_f) in spec_iter.zip(coefs_arr.axis_iter(Axis(2))) {
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

fn init_encoder_impl(
    mut m: InferenceModel,
    df_cfg: &ini::Properties,
    n_ch: usize,
) -> Result<(TypedModel, usize)> {
    let s = tract_pulse::fact::stream_dim();

    let nb_erb = df_cfg.get("nb_erb").unwrap().parse::<usize>()?;
    let nb_df = df_cfg.get("nb_df").unwrap().parse::<usize>()?;
    let feat_erb = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, 1, s, nb_erb));
    let feat_spec = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, 2, s, nb_df));

    m = m
        .with_input_fact(0, feat_erb)?
        .with_input_fact(1, feat_spec)?
        .with_input_names(&["feat_erb", "feat_spec"])?
        .with_output_names(&["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"])?;

    m.analyse(true)?;
    let mut m = m.into_typed()?;

    m.declutter()?;
    let pulsed = PulsedModel::new(&m, 1)?;
    let delay = pulsed.output_fact(0)?.delay;
    log::info!("Init encoder with delay: {}", delay);
    let m = pulsed.into_typed()?.into_optimized()?;
    Ok((m, delay))
}
fn init_encoder(m: &Path, df_cfg: &ini::Properties, n_ch: usize) -> Result<(TypedModel, usize)> {
    let m = tract_onnx::onnx().model_for_path(m)?;
    init_encoder_impl(m, df_cfg, n_ch)
}

fn init_encoder_from_read(
    m: &mut dyn Read,
    df_cfg: &ini::Properties,
    n_ch: usize,
) -> Result<(TypedModel, usize)> {
    let m = tract_onnx::onnx().model_for_read(m)?;
    init_encoder_impl(m, df_cfg, n_ch)
}

fn init_erb_decoder_impl(
    mut m: InferenceModel,
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
    n_ch: usize,
) -> Result<(TypedModel, usize)> {
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

    m = m
        .with_input_fact(0, emb)?
        .with_input_fact(1, e3)?
        .with_input_fact(2, e2)?
        .with_input_fact(3, e1)?
        .with_input_fact(4, e0)?
        .with_input_names(&["emb", "e3", "e2", "e1", "e0"])?
        .with_output_names(&["m"])?;

    m.analyse(true)?;
    let mut m = m.into_typed()?;

    m.declutter()?;
    let pulsed = PulsedModel::new(&m, 1)?;
    let delay = pulsed.output_fact(0)?.delay;
    log::info!("Init ERB decoder with delay: {}", delay);
    let m = pulsed.into_typed()?.into_optimized()?;
    Ok((m, delay))
}
fn init_erb_decoder(
    m: &Path,
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
    n_ch: usize,
) -> Result<(TypedModel, usize)> {
    let m = tract_onnx::onnx().model_for_path(m)?;
    init_erb_decoder_impl(m, net_cfg, df_cfg, n_ch)
}
fn init_erb_decoder_from_read(
    m: &mut dyn Read,
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
    n_ch: usize,
) -> Result<(TypedModel, usize)> {
    let m = tract_onnx::onnx().model_for_read(m)?;
    init_erb_decoder_impl(m, net_cfg, df_cfg, n_ch)
}

fn init_df_decoder_impl(
    mut m: InferenceModel,
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
    n_ch: usize,
) -> Result<(TypedModel, usize)> {
    let s = tract_pulse::fact::stream_dim();

    let nb_df = df_cfg.get("nb_df").unwrap().parse::<usize>()?;
    let n_hidden = net_cfg.get("emb_hidden_dim").unwrap().parse::<usize>()?;
    let layer_width = net_cfg.get("conv_ch").unwrap().parse::<usize>()?;

    let emb = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_ch, s, n_hidden));
    let c0 = InferenceFact::dt_shape(
        f32::datum_type(),
        shapefactoid!(n_ch, layer_width, s, nb_df),
    );

    m = m
        .with_input_fact(0, emb)?
        .with_input_fact(1, c0)?
        .with_input_names(&["emb", "c0"])?
        .with_output_names(&["coefs"])?;

    m.analyse(true)?;
    let mut m = m.into_typed()?;

    m.declutter()?;
    let pulsed = PulsedModel::new(&m, 1)?;
    let delay = pulsed.output_fact(0)?.delay;
    log::info!("Init DF decoder with delay: {}", delay);
    let m = pulsed.into_typed()?.into_optimized()?;
    Ok((m, delay))
}
fn init_df_decoder(
    m: &Path,
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
    n_ch: usize,
) -> Result<(TypedModel, usize)> {
    let m = tract_onnx::onnx().model_for_path(m)?;
    init_df_decoder_impl(m, net_cfg, df_cfg, n_ch)
}
fn init_df_decoder_from_read(
    m: &mut dyn Read,
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
    n_ch: usize,
) -> Result<(TypedModel, usize)> {
    let m = tract_onnx::onnx().model_for_read(m)?;
    init_df_decoder_impl(m, net_cfg, df_cfg, n_ch)
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

pub fn as_slice_mut_complex(buffer: &mut [f32]) -> &mut [Complex32] {
    unsafe {
        let ptr = buffer.as_ptr() as *mut Complex32;
        let len = buffer.len();
        std::slice::from_raw_parts_mut(ptr, len / 2)
    }
}

pub fn as_slice_mut_real(buffer: &mut [Complex32]) -> &mut [f32] {
    unsafe {
        let ptr = buffer.as_ptr() as *mut f32;
        let len = buffer.len();
        std::slice::from_raw_parts_mut(ptr, len * 2)
    }
}

pub fn slice_as_arrayview<'a>(
    buffer: &[f32],
    shape: &[usize], // having an explicit shape parameter allows to also squeeze axes.
) -> ArrayViewD<'a, f32> {
    debug_assert_eq!(buffer.len(), shape.iter().product::<usize>());
    unsafe {
        let ptr = buffer.as_ptr(); // as *mut Complex32;
        ArrayViewD::from_shape_ptr(shape, ptr)
    }
}
pub fn mut_slice_as_arrayviewmut<'a>(
    buffer: &mut [f32],
    shape: &[usize], // having an explicit shape parameter allows to also squeeze axes.
) -> ArrayViewMutD<'a, f32> {
    debug_assert_eq!(buffer.len(), shape.iter().product::<usize>());
    unsafe {
        let ptr = buffer.as_mut_ptr(); // as *mut Complex32;
        ArrayViewMutD::from_shape_ptr(shape, ptr)
    }
}
pub fn as_arrayview_complex<'a>(
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
pub fn as_array_mut_complex<'a>(
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
