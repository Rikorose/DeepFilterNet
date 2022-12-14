/// This module is based on sile/dagc but also supports a simple stereo version
use ndarray::{ArrayViewMut2, Axis};

#[derive(Debug)]
pub struct Agc {
    pub desired_output_rms: f32,
    pub distortion_factor: f32,
    pub gain: f32,
}

impl Agc {
    pub fn new(desired_output_rms: f32, distortion_factor: f32) -> Self {
        assert!(desired_output_rms > 0.);
        assert!((0f32..1f32).contains(&distortion_factor));
        Self {
            desired_output_rms,
            distortion_factor,
            gain: 1.,
        }
    }

    /// Process a chunk of samples
    pub fn process(&mut self, mut samples: ArrayViewMut2<f32>, snr: Option<f32>) {
        let frozen = snr.unwrap_or_default() < 8.;
        if frozen {
            samples.map_inplace(|s| *s = *s * self.gain);
        } else {
            for mut s in samples.axis_iter_mut(Axis(1)) {
                s.map_inplace(|s| *s = *s * self.gain);
                let y = s.mean().unwrap().powi(2) / self.desired_output_rms;
                let z = 1.0 + (self.distortion_factor * (1.0 - y));
                self.gain *= z;
            }
        }
    }
}
