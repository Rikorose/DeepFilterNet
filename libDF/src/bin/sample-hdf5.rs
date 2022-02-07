use std::env::args;

use anyhow::Result;
use df::dataset::Hdf5Dataset;

fn main() -> Result<()> {
    let mut args = args();
    dbg!(&args);
    let p = args.nth(2).expect("HDF5 dataset path expected");
    dbg!(&p);
    let n = args.nth(3).unwrap_or("1".to_string()).parse::<usize>()?;
    let ds = Hdf5Dataset::new(&p)?;
    let k = "LJSpeech-1.1_wavs_LJ044-0172.wav";
    let data = ds.read("LJSpeech-1.1_wavs_LJ044-0172.wav")?;
    dbg!(k, data.shape());
    for (k, _i) in ds.keys()?.iter().zip(0..n) {
        let data = ds.read(k)?;
        dbg!(k, data.shape());
    }
    Ok(())
}
