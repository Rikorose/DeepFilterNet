use std::env::args;
use std::process::exit;

use anyhow::Result;
use df::dataset::Hdf5Dataset;
use df::wav_utils::write_wav_arr2;
use rand::seq::SliceRandom;

fn main() -> Result<()> {
    let args = args().collect::<Vec<String>>();
    let p = match args.get(1) {
        Some(p) => p,
        None => {
            eprintln!("HDF5 dataset path expected");
            exit(1);
        }
    };
    let ds = Hdf5Dataset::new(p)?;
    let k = match args.get(2) {
        Some(k) => k.to_string(),
        None => ds.keys()?.choose(&mut rand::thread_rng()).unwrap().to_string(),
    };
    let data = ds.read(&k).unwrap();
    let out_dir = args.get(3).cloned().unwrap_or_else(|| "out".to_owned());
    let name = format!("{out_dir}/{k}");
    println!("{name}");
    write_wav_arr2(&name, data.view(), ds.sr.unwrap_or(24000) as u32).unwrap();
    Ok(())
}
