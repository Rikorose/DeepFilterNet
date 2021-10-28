use std::env::args;
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Result};
use df::{wav_utils::*, Complex32, DFState};
use ini::Ini;
use ndarray::prelude::*;
use ndarray::Axis;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::shapefactoid;
use tract_pulse::internal::ToDim;
use tract_pulse::model::*;

fn constantize_input(
    model: &mut InferenceModel,
    name: &str,
    value: Arc<Tensor>,
) -> TractResult<()> {
    let node_id = model.node_by_name(name)?.id;
    model.node_mut(node_id).op =
        tract_onnx::tract_core::ops::konst::Const::new(value.clone()).into();
    model.node_mut(node_id).outputs[0].fact = InferenceFact::from(value);
    Ok(())
}

fn init_encoder(
    enc: &Path,
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
) -> Result<PulsedModel> {
    let s = tract_pulse::fact::stream_dim();

    let nb_erb = df_cfg.get("nb_erb").unwrap().parse::<usize>()?;
    let nb_df = df_cfg.get("nb_df").unwrap().parse::<usize>()?;
    let feat_erb = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(1, 1, s, nb_erb));
    let feat_spec = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(1, 2, s, nb_df));

    let mut enc = tract_onnx::onnx()
        .model_for_path(enc)?
        .with_input_fact(0, feat_erb)?
        .with_input_fact(1, feat_spec)?;

    // need to erase h0emb from network interface so pulse and scan do their thing
    // first the state inputs are made into constant
    let n_hidden = net_cfg.get("emb_hidden_dim").unwrap().parse::<usize>()?;
    let gru_groups = net_cfg.get("gru_groups").unwrap().parse::<usize>()?;
    let h0emb = Tensor::zero::<f32>(&[1, 1, n_hidden / gru_groups])?;
    if constantize_input(&mut enc, "h0emb", h0emb.into()).is_err() {
        eprintln!("No gru state found in onnx file. Skipping constantization.");
    };
    enc = enc
        .with_input_names(&["feat_erb", "feat_spec"])?
        .with_output_names(&["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"])?;
    //dbg!(&enc);
    enc.analyse(true)?;
    let enc = enc.into_typed()?;

    let enc = enc.declutter()?;
    let pulsed = PulsedModel::new(&enc, 1)?;
    let delay = pulsed.output_fact(0)?.delay;
    println!("Init encoder with delay: {}", delay);
    Ok(pulsed)
}

// Transposed convs are not fully supported by tract pulse; return normal typed model.
fn init_decoder(
    dec: &Path,
    net_cfg: &ini::Properties,

    df_cfg: &ini::Properties,
) -> Result<TypedModel> {
    let s = 1;

    let nb_erb = df_cfg.get("nb_erb").unwrap().parse::<usize>()?;
    let n_hidden = net_cfg.get("emb_hidden_dim").unwrap().parse::<usize>()?;
    let layer_width = net_cfg.get("conv_ch").unwrap().parse::<usize>()?;
    let wf = net_cfg.get("conv_width_factor").unwrap().parse::<usize>()?;
    //let emb_width = layer_width * wf.pow(2);
    //let emb_dim = emb_width * nb_erb / 4;
    dbg!(nb_erb, n_hidden, wf);
    let emb = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(1, s, n_hidden));
    let e3ch = layer_width * wf.pow(2);
    let e3f = nb_erb / 4;
    let e3 = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(1, e3ch, s, e3f));
    let e2 = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(1, e3ch, s, e3f));
    let e1ch = layer_width * wf.pow(1);
    let e1f = nb_erb / 2;
    let e1 = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(1, e1ch, s, e1f));
    let e0 = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(1, layer_width, s, nb_erb));

    let mut dec = tract_onnx::onnx()
        .model_for_path(dec)?
        .with_input_fact(0, emb)?
        .with_input_fact(1, e3)?
        .with_input_fact(2, e2)?
        .with_input_fact(3, e1)?
        .with_input_fact(4, e0)?;
    dec = dec
        .with_input_names(&["emb", "e3", "e2", "e1", "e0"])?
        .with_output_names(&["m"])?;
    let dec = dec.into_optimized()?;
    println!("Init decoder with delay: {}", 0.);
    Ok(dec)
}

fn init_dfmodule(
    dfmodule: &Path,
    net_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
) -> Result<PulsedModel> {
    println!("Init dfmodule");
    let s = tract_pulse::fact::stream_dim();

    let emb_n_hidden = net_cfg.get("emb_hidden_dim").unwrap().parse::<usize>()?;
    let df_n_hidden = net_cfg.get("df_hidden_dim").unwrap().parse::<usize>()?;
    let df_n_layers = net_cfg.get("df_num_layers").unwrap().parse::<usize>()?;
    let nb_df = df_cfg.get("nb_df").unwrap().parse::<usize>()?;
    let layer_width = net_cfg.get("conv_ch").unwrap().parse::<usize>()?;
    let gru_groups = net_cfg.get("gru_groups").unwrap().parse::<usize>()?;
    // Real and imaginary part are in channel dimension
    let emb = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(1, s, emb_n_hidden));
    let c0 = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(1, layer_width, s, nb_df));
    dbg!(&emb.shape, &c0.shape);
    let mut df = tract_onnx::onnx()
        .model_for_path(dfmodule)?
        .with_input_fact(0, emb)?
        .with_input_fact(1, c0)?;
    // Maybe constantize gru state needed for tract pulse
    let hdf = Tensor::zero::<f32>(&[df_n_layers, 1, df_n_hidden / gru_groups])?;
    if constantize_input(&mut df, "hdf", hdf.into()).is_err() {
        eprintln!("No gru state found in onnx file. Skipping constantization.");
    };
    df = df.with_input_names(&["emb", "c0"])?.with_output_names(&["coefs", "alpha"])?;
    df.analyse(true)?;
    let df = df.into_typed()?;
    let df = df.declutter()?;
    let pulsed = PulsedModel::new(&df, 1)?;
    let delay = pulsed.output_fact(0)?.delay;
    println!("with delay: {}", delay);
    Ok(pulsed)
}

fn init_dfop_delayspec(dfopinit: &Path, df_cfg: &ini::Properties) -> Result<PulsedModel> {
    let s = tract_pulse::fact::stream_dim();
    println!("Init df OP delay");

    let n_freq = df_cfg.get("fft_size").unwrap().parse::<usize>()? / 2 + 1;

    let spec = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(s, n_freq, 2));
    let mut dfopinit = tract_onnx::onnx().model_for_path(dfopinit)?.with_input_fact(0, spec)?;

    dfopinit = dfopinit.with_input_names(&["spec"])?.with_output_names(&["spec_d"])?;
    dfopinit.analyse(true)?;
    let dfopinit = dfopinit.into_typed()?;
    let dfopinit = dfopinit.declutter()?;
    dbg!(&dfopinit);
    let pulsed = PulsedModel::new(&dfopinit, 1)?;
    let delay = pulsed.output_fact(0)?.delay;
    println!("Init dfop delay with delay: {}", delay);
    Ok(pulsed)
}

fn init_dfop_step(
    dfopstep: &Path,
    model_cfg: &ini::Properties,
    df_cfg: &ini::Properties,
) -> Result<TypedModel> {
    println!("Init df OP step");

    let nb_df = df_cfg.get("nb_df").unwrap().parse::<usize>()?;
    let df_order = model_cfg.get("df_order").unwrap().parse::<usize>()?;
    let n_freq = df_cfg.get("fft_size").unwrap().parse::<usize>()? / 2 + 1;

    let spec = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(n_freq, 2));
    let coefs = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(df_order, nb_df, 2));
    let alpha = InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(1));
    let spec_buf_in =
        InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(df_order, n_freq, 2));
    dbg!(&spec.shape, &coefs.shape, &alpha.shape, &spec_buf_in.shape);
    let mut dfop = tract_onnx::onnx()
        .model_for_path(dfopstep)?
        .with_input_fact(0, spec)?
        .with_input_fact(1, coefs)?
        .with_input_fact(2, alpha)?
        .with_input_fact(3, spec_buf_in)?;
    dfop = dfop
        .with_input_names(&["spec", "coefs", "alpha", "spec_buf_in"])?
        .with_output_names(&["spec_f", "spec_buf"])?;
    let dfop = dfop.into_typed()?;
    let dfop = dfop.into_optimized()?;
    Ok(dfop)
}

fn main() -> Result<()> {
    let base_dir = args().nth(1).expect("path to base dir expected");
    let base_dir = Path::new(&base_dir);
    if !base_dir.is_dir() {
        return Err(anyhow!("Base dir not found"));
    }
    let config = base_dir.join("config.ini");
    let exp_dir = base_dir.join("export");
    if !config.is_file() {
        return Err(anyhow!("Config not found"));
    }
    let config = Ini::load_from_file(config)?;
    let model_cfg = config.section(Some("deepfilternet")).unwrap();
    let df_cfg = config.section(Some("df")).unwrap();
    let mut enc = SimpleState::new(
        init_encoder(&exp_dir.join("enc.onnx"), model_cfg, df_cfg)?
            .into_typed()?
            .into_optimized()?
            .into_runnable()?,
    )?;
    let mut dec = SimpleState::new(
        init_decoder(&exp_dir.join("dec.onnx"), model_cfg, df_cfg)?.into_runnable()?,
    )?;
    let mut df_net = SimpleState::new(
        init_dfmodule(&exp_dir.join("dfnet.onnx"), model_cfg, df_cfg)?
            .into_typed()?
            .into_optimized()?
            .into_runnable()?,
    )?;
    let mut spec_delay = SimpleState::new(
        init_dfop_delayspec(&exp_dir.join("dfop_delayspec.onnx"), df_cfg)?
            .into_typed()?
            .into_optimized()?
            .into_runnable()?,
    )?;
    let mut df_step = SimpleState::new(
        init_dfop_step(&exp_dir.join("dfop_step.onnx"), model_cfg, df_cfg)?.into_runnable()?,
    )?;

    let sr = df_cfg.get("sr").unwrap().parse::<usize>()?;
    let hop_size = df_cfg.get("hop_size").unwrap().parse::<usize>()?;
    let fft_size = df_cfg.get("fft_size").unwrap().parse::<usize>()?;
    let min_nb_erb_freqs = df_cfg.get("min_nb_erb_freqs").unwrap().parse::<usize>()?;
    let nb_erb = df_cfg.get("nb_erb").unwrap().parse::<usize>()?;
    let nb_df = df_cfg.get("nb_df").unwrap().parse::<usize>()?;
    let nb_freq_bins = fft_size / 2 + 1;
    let df_order = model_cfg.get("df_order").unwrap().parse::<usize>()?;
    let alpha = df_cfg.get("norm_alpha").unwrap().parse::<f32>()?;
    let min_db = df_cfg.get("min_db").unwrap().parse::<f32>()?;
    let clamp_min = 10f32.powf(min_db / 20.0);
    assert!(clamp_min < 1.0);
    dbg!(clamp_min);

    let reader = ReadWav::new("assets/noisy_snr0_mono.wav")?;
    assert_eq!(sr, reader.sr);
    let noisy = reader.samples_arr2()?;
    let mut enh: Array2<f32> = ArrayD::default(noisy.shape()).into_dimensionality()?;

    // Init buffers
    let mut spec_buf = tensor0(0f32).broadcast_scalar_to_shape(&[nb_freq_bins, 2])?;
    let mut erb_buf = tensor0(0f32).broadcast_scalar_to_shape(&[1, 1, 1, nb_erb])?;
    let mut cplx_buf = tensor0(0f32).broadcast_scalar_to_shape(&[1, 1, nb_df, 2])?;
    let mut m_zeros = vec![clamp_min; nb_erb];
    let mut rolling_spec_buf =
        tensor0(0f32).broadcast_scalar_to_shape(&[df_order, nb_freq_bins, 2])?;

    let ch = noisy.len_of(Axis(0));
    let mut states = Vec::with_capacity(ch);
    for _ in 0..ch {
        let mut state = DFState::new(sr as usize, fft_size, hop_size, nb_erb, min_nb_erb_freqs);
        state.init_norm_states(nb_df);
        states.push(state)
    }
    let t0 = Instant::now();
    // loop over input stream
    let mut i = 0;
    for (noisy_ch, mut enh_ch) in noisy
        .axis_chunks_iter(Axis(1), hop_size)
        .zip(enh.axis_chunks_iter_mut(Axis(1), hop_size))
    {
        for ((noisy_frame, mut enh_frame), state) in noisy_ch
            .axis_iter(Axis(0))
            .zip(enh_ch.axis_iter_mut(Axis(0)))
            .zip(states.iter_mut())
        {
            //dbg!(noisy_frame.shape(), enh_frame.shape());
            let mut input = vec![0f32; hop_size];
            for (inp, &nsy) in input.iter_mut().zip(noisy_frame) {
                *inp = nsy
            }
            if noisy_frame.len() < hop_size {
                break; // for now
            }
            // Compute spectrogram and dnn features
            let spec = convert_to_mut_complex(spec_buf.as_slice_mut()?);
            state.analysis(&input, spec);
            state.erb_feat(&spec, alpha, erb_buf.as_slice_mut()?);
            state.cplx_feat_clone(
                &spec[0..nb_df],
                alpha,
                convert_to_mut_complex(cplx_buf.as_slice_mut()?),
            );

            // Run dnn
            let mut enc_emb = enc.run(tvec!(
                erb_buf.clone(),
                cplx_buf.clone().permute_axes(&[0, 3, 1, 2])?
            ))?;

            let &lsnr = enc_emb.pop().unwrap().to_scalar::<f32>()?;
            let c0 = enc_emb.pop().unwrap();
            let emb = enc_emb.pop().unwrap();
            if i == 0 || lsnr < -15. {
                state.apply_mask(spec, &mut m_zeros, false);
            } else {
                //dbg!(emb.shape());
                let dec_input = tvec!(
                    emb.clone().into_tensor(),
                    enc_emb.pop().unwrap().into_tensor(), // e3
                    enc_emb.pop().unwrap().into_tensor(), // e2
                    enc_emb.pop().unwrap().into_tensor(), // e1
                    enc_emb.pop().unwrap().into_tensor(), // e0
                );
                let mut m = dec.run(dec_input)?;
                let mut m = m.pop().unwrap().into_tensor().into_shape(&[nb_erb])?;
                state.apply_mask(spec, m.as_slice_mut()?, true);
            }
            spec_buf = spec_delay
                .run(tvec!(spec_buf.clone().into_shape(&[1, nb_freq_bins, 2])?))?
                .pop()
                .unwrap()
                .into_tensor();
            if lsnr < 30.0 {
                // Run Deep Filter Module
                let mut df = df_net.run(tvec!(emb.into_tensor(), c0.into_tensor()))?;
                let alpha = df.pop().unwrap();
                let coefs = df.pop().unwrap();
                //dbg!(spec_buf.shape());
                //println!(
                //    "{}, {}",
                //    spec_buf.as_slice()?.iter().sum::<f32>(),
                //    alpha.to_scalar::<f32>()?
                //);
                //if i < 15 {
                //    dbg!(spec_buf.as_slice()?.iter().sum::<f32>());
                //    dbg!(spec_d.as_slice()?.iter().sum::<f32>());
                //    dbg!(spec_buf.as_slice()?.iter().sum::<f32>());
                //    dbg!(alpha.to_scalar::<f32>()?);
                //}
                let mut out = df_step.run(tvec!(
                    spec_buf.into_shape(&[nb_freq_bins, 2])?,
                    coefs.into_tensor().into_shape(&[df_order, nb_df, 2])?,
                    //alpha.into_tensor().into_shape(&[1])?,
                    tensor0(1f32).broadcast_scalar_to_shape(&[1])?,
                    rolling_spec_buf,
                ))?;
                rolling_spec_buf = out.pop().unwrap().into_tensor();
                spec_buf = out.pop().unwrap().into_tensor();
                //println!("{}", spec_buf.as_slice()?.iter().sum::<f32>());
            }
            state.synthesis(
                convert_to_mut_complex(spec_buf.as_slice_mut()?),
                enh_frame.as_slice_mut().unwrap(),
            );
        }
        i += 1;
    }
    let duration_ms = t0.elapsed().as_millis();
    let audio_len_ms = noisy.len_of(Axis(1)) as f32 / sr as f32 * 1000.;
    println!(
        "Enhanced file in {} ms (RT factor: {})",
        duration_ms,
        audio_len_ms / duration_ms as f32
    );
    write_wav_arr2("out/enh.wav", enh.view(), sr as u32)?;

    Ok(())
}

pub fn convert_to_mut_complex(buffer: &mut [f32]) -> &mut [Complex32] {
    unsafe {
        let ptr = buffer.as_ptr() as *mut Complex32;
        let len = buffer.len();
        std::slice::from_raw_parts_mut(ptr, len / 2)
    }
}

pub fn convert_to_mut_real(buffer: &mut [Complex32]) -> &mut [f32] {
    unsafe {
        let ptr = buffer.as_ptr() as *mut f32;
        let len = buffer.len();
        std::slice::from_raw_parts_mut(ptr, len * 2)
    }
}
