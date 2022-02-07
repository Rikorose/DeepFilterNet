use std::env::args;

use anyhow::Result;
use df::{
    augmentations::seed_from_u64,
    dataset::{Dataset, DatasetBuilder, DatasetConfigJson, Split},
};

fn main() -> Result<()> {
    let args = args().collect::<Vec<String>>();
    let cfg = args.get(1).expect("HDF5 config file expected");
    let dir = args.get(2).expect("HDF5 dataset dir expected");
    // let out = args.nth(3).unwrap_or_else(|| "out".to_string());
    seed_from_u64(42);
    let cfg = DatasetConfigJson::open(cfg).unwrap();
    let mut ds_builder = DatasetBuilder::new(dir, 48000).df_params(960, None, None, None, None);
    ds_builder = ds_builder.p_sample_full_speech(1.0);
    let mut train_ds = ds_builder.dataset(cfg.split_config(Split::Train)).build_fft_dataset()?;
    train_ds.generate_keys()?;

    for idx in 0..train_ds.len() {
        let sample = train_ds.get_sample(idx, Some(0)).unwrap();
        dbg!(sample);
    }
    Ok(())
}
