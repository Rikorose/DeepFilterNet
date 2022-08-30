use std::path::{Path, PathBuf};

use crate::dataset::*;

/// Generate path of json cache file containing the HDF5 keys.
pub fn cache_path(cfg_path: &str) -> PathBuf {
    let mut p = Path::new(cfg_path).to_path_buf();
    let cache_file_name = p.file_stem().unwrap().to_str().unwrap().to_owned();
    p.set_file_name(".cache_".to_owned() + &cache_file_name);
    p.set_extension("cfg");
    p
}
/// Load HDF5 keys into DatasetConfigJson using the json cache file.
/// This function validates the topicality by hashing the modified timestamp and file size.
pub fn load_hdf5_key_cache(cfg_path: &str, cfg: &mut DatasetConfigJson) {
    let cache_path = cache_path(cfg_path);
    if !cache_path.is_file() {
        return;
    }
    log::info!(
        "Loading HDF5 key cache from {}",
        cache_path.to_str().unwrap_or_default()
    );
    match DatasetConfigCacheJson::open(cache_path.to_str().unwrap()) {
        Err(e) => log::warn!("Could not load dataset keys cache: {}", e),
        Ok(cache) => {
            cfg.set_keys(Split::Train, cache.keys()).expect("Could not set cached keys");
            cfg.set_keys(Split::Valid, cache.keys()).expect("Could not set cached keys");
            cfg.set_keys(Split::Test, cache.keys()).expect("Could not set cached keys");
        }
    }
}
/// Write all combined (train/valid/test) Hdf5Cfgs to a json file. This may be used for the next
/// TdDataset initialization.
pub fn write_hdf5_key_cache(cfg_path: &str, cfg: &DatasetConfigJson) {
    let cache_path = cache_path(cfg_path);
    let mut cache = Vec::new();
    cache.extend(cfg.train.iter().filter_map(|x| x.keys_unchecked().cloned()));
    cache.extend(cfg.valid.iter().filter_map(|x| x.keys_unchecked().cloned()));
    cache.extend(cfg.test.iter().filter_map(|x| x.keys_unchecked().cloned()));
    let cache = DatasetConfigCacheJson::new(cache);
    log::trace!("Writing HDF5 json key cache to {}", cache_path.display());
    cache.write(cache_path.to_str().unwrap()).expect("Failed to write cache.");
}
/// Fetch latest HDF5 keys and update the corresponding Hdf5Cfgs.
///
/// This method is supposed to be called after TdDataset initialization so that the HDF5 keys are
/// stored within each Hdf5Cfg. Thus, we can serialize Hdf5Cfg to json and reuse the cached keys for
/// the next initialization.
pub fn fetch_hdf5_keys_from_ds(ds_dir: &str, cfgs: &mut [Hdf5Cfg], ds: &FftDataset) {
    for hdf5cfg in cfgs.iter_mut() {
        let ds_path = ds_dir.to_owned() + "/" + hdf5cfg.filename();
        let cfg = match ds.get_hdf5cfg(hdf5cfg.filename()) {
            Some(cfg) => cfg,
            None => {
                log::warn!("Could not get hdf5cfg for filename {}", hdf5cfg.filename());
                continue;
            }
        };
        let hash = cfg
            .hash()
            .unwrap_or_else(|| cfg.hash_from_ds_path(&ds_path).expect("Could not calculate hash"));
        if let Some(ds_keys) = cfg.load_keys(hash).expect("Could not load Hdf5Keys.") {
            hdf5cfg.set_keys(ds_keys.clone()).expect("Could not update keys");
        }
    }
}
