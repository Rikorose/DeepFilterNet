use std::env::args;

use anyhow::Result;
use df::dataset::Hdf5Dataset;
use df::wav_utils::write_wav_arr2;
use rand::seq::SliceRandom;

fn main() -> Result<()> {
    let args = args().collect::<Vec<String>>();
    let p = args.get(1).expect("HDF5 dataset path expected");
    let k = match args.get(2) {
        Some(k) => k.to_string(),
        None => ds.keys()?.choose(&mut rand::thread_rng()).unwrap().to_string(),
    };
    let data = ds.read(&k)?;
    dbg!(&k, data.shape());
    let name = "out/".to_owned() + &k;
    println!("Sampled {}", k);
    write_wav_arr2(&name, data.view(), ds.sr.unwrap() as u32).unwrap();
    Ok(())
}
