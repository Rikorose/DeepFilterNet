use ndarray::prelude::*;
use rand::Rng;
use thiserror::Error;

use crate::transforms::*;
use crate::util::*;

pub type Result<T> = std::result::Result<T, DfMixUtilsError>;

#[derive(Error, Debug)]
pub enum DfMixUtilsError {
    #[error("Ndarray Shape Error")]
    NdarrayShapeError(#[from] ndarray::ShapeError),
    #[error("DF Utils Error")]
    UtilsError(#[from] UtilsError),
}

pub fn combine_noises(
    ch: usize,
    len: usize,
    noises: &mut [Array2<f32>],
    noise_gains: Option<&[f32]>,
) -> Result<Array2<f32>> {
    let mut rng = thread_rng()?;
    // Adjust length of noises to clean length
    for ns in noises.iter_mut() {
        loop {
            if len.checked_sub(ns.len_of(Axis(1))).is_some() {
                // TODO: Remove this clone if ndarray supports repeat
                ns.append(Axis(1), ns.clone().view())?;
            } else {
                break;
            }
        }
        let too_large = ns.len_of(Axis(1)).checked_sub(len);
        if let Some(too_large) = too_large {
            let start: usize = rng.gen_range(0..too_large);
            ns.slice_collapse(s![.., start..start + len]);
        }
    }
    // Adjust number of noise channels to clean channels
    for ns in noises.iter_mut() {
        while ns.len_of(Axis(0)) > ch {
            ns.remove_index(Axis(0), rng.gen_range(0..ns.len_of(Axis(0))))
        }
        while ns.len_of(Axis(0)) < ch {
            let r = rng.gen_range(0..ns.len_of(Axis(0)));
            let slc = ns.slice(s![r..r + 1, ..]).to_owned();
            ns.append(Axis(0), slc.view())?;
        }
    }
    // Apply gain to noises
    if let Some(ns_gains) = noise_gains {
        for (ns, &g) in noises.iter_mut().zip(ns_gains) {
            *ns *= 10f32.powf(g / 20.);
        }
    }
    // Average noises
    let noise = Array2::zeros((ch, len));
    let noise = noises.iter().fold(noise, |acc, x| acc + x) / ch as f32;
    Ok(noise)
}

pub fn mix_audio_signal(
    clean: Array2<f32>,
    clean_rev: Option<Array2<f32>>,
    mut noise: Array2<f32>,
    snr_db: f32,
    gain_db: f32,
    atten_db: Option<f32>,
) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
    // Apply gain to speech
    let g = 10f32.powf(gain_db / 20.);
    let mut clean_out = &clean * g;
    let clean_mix = clean_rev.map(|c| &c * g).unwrap_or_else(|| clean_out.clone());
    // For energy calculation use clean speech to also consider direct-to-reverberant ratio
    let k = mix_f(clean.view(), noise.view(), snr_db);
    if let Some(atten_db) = atten_db {
        // Create a mixture with a higher SNR as target signal
        let k_target = mix_f(clean.view(), noise.view(), snr_db + atten_db);
        for (c, &n) in clean_out.iter_mut().zip(noise.iter()) {
            *c += n * k_target;
        }
    }
    // Create mixture at given SNR
    noise *= k;
    let mut mixture = clean_mix + &noise;
    // Guard against clipping
    let max = &([&clean_out, &noise, &mixture].iter().map(|x| find_max_abs(x.iter())))
        .collect::<std::result::Result<Vec<f32>, crate::util::UtilsError>>()?;
    let max = find_max(max)?;
    if (max - 1.) > 1e-10 {
        let f = 1. / (max + 1e-10);
        clean_out *= f;
        noise *= f;
        mixture *= f;
    }
    Ok((clean_out, noise, mixture))
}
