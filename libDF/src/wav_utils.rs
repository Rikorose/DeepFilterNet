use std::result::Result;
use std::{
    fs::File,
    io::{BufReader, Read},
};

use hound::{WavReader, WavWriter};
#[cfg(feature = "dataset")]
use ndarray::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WavUtilsError {
    #[error("Hound Error")]
    HoundError(#[from] hound::Error),
    #[error("Hound Error Detail")]
    HoundErrorDetail { source: hound::Error, msg: String },
    #[error("Ndarray Shape Error")]
    NdarrayShapeError(#[from] ndarray::ShapeError),
}

pub struct ReadWav {
    reader: WavReader<BufReader<File>>,
    pub channels: usize,
    pub sr: usize,
    pub len: usize,
}

impl ReadWav {
    pub fn new(path: &str) -> Result<Self, WavUtilsError>
    where
        Self: Sized,
    {
        let reader = match WavReader::open(path) {
            Err(e) => {
                return Err(WavUtilsError::HoundErrorDetail {
                    source: e,
                    msg: format!("Could not find audio file {}", path),
                })
            }
            Ok(r) => r,
        };
        let channels = reader.spec().channels as usize;
        let sr = reader.spec().sample_rate as usize;
        let len = reader.len() as usize / channels;
        Ok(ReadWav {
            reader,
            channels,
            sr,
            len,
        })
    }
    pub fn iter(&mut self) -> impl Iterator<Item = f32> + '_ {
        read_wav_raw(&mut self.reader)
    }
    pub fn samples_vec(mut self) -> Result<Vec<Vec<f32>>, WavUtilsError> {
        let mut out = vec![Vec::<f32>::new(); self.channels];
        let mut samples = self.iter();
        'outer: loop {
            for out_ch in out.iter_mut() {
                match samples.next() {
                    None => break 'outer,
                    Some(x) => out_ch.push(x),
                }
            }
        }
        Ok(out)
    }
    #[cfg(feature = "dataset")]
    pub fn samples_arr2(mut self) -> Result<Array2<f32>, WavUtilsError> {
        Ok(
            Array2::from_shape_vec((self.len, self.channels), self.iter().collect())?
                .t()
                .to_owned(),
        )
    }
}

fn read_wav_raw<R: Read>(reader: &mut WavReader<R>) -> impl Iterator<Item = f32> + '_ {
    reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32767.0)
}

pub fn read_wav(path: &str) -> Result<(Vec<Vec<f32>>, u32), WavUtilsError> {
    let mut reader = WavReader::open(path)?;
    let ch = reader.spec().channels as usize;
    let sr = reader.spec().sample_rate;
    let mut out = vec![Vec::<f32>::new(); ch];
    let mut samples = read_wav_raw(&mut reader);
    'outer: loop {
        for out_ch in out.iter_mut() {
            match samples.next() {
                None => break 'outer,
                Some(x) => out_ch.push(x),
            }
        }
    }
    Ok((out, sr))
}

pub fn write_wav_iter<'a, I>(path: &str, iter: I, sr: u32, ch: u16) -> Result<(), WavUtilsError>
where
    I: IntoIterator<Item = &'a f32>,
{
    let spec = hound::WavSpec {
        channels: ch,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;

    for &sample in iter.into_iter() {
        writer.write_sample((sample * i16::MAX as f32) as i16)?;
    }
    Ok(())
}

pub fn write_wav(path: &str, x: &[Vec<f32>], sr: u32) -> Result<(), WavUtilsError> {
    let spec = hound::WavSpec {
        channels: x.len() as u16,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;

    for t in 0..x[0].len() {
        for ch in x.iter() {
            writer.write_sample((ch[t] * i16::MAX as f32) as i16)?;
        }
    }
    Ok(writer.finalize()?)
}

#[cfg(feature = "dataset")]
pub fn write_wav_arr2(path: &str, x: ArrayView2<f32>, sr: u32) -> Result<(), WavUtilsError> {
    let spec = hound::WavSpec {
        channels: x.len_of(Axis(0)) as u16,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for xt in x.axis_iter(Axis(1)) {
        for s in xt.iter() {
            writer.write_sample((s * i16::MAX as f32) as i16)?;
        }
    }
    Ok(writer.finalize()?)
}
