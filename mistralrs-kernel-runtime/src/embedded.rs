use crate::universal::KernelCatalog;
use anyhow::{anyhow, Result};
use std::sync::{Arc, OnceLock};

include!(concat!(env!("OUT_DIR"), "/dkr_typed_embedded.rs"));

pub fn catalog() -> Result<Arc<KernelCatalog>> {
    static CATALOG: OnceLock<Result<Arc<KernelCatalog>, String>> = OnceLock::new();
    CATALOG
        .get_or_init(|| {
            let result = load().map(Arc::new).map_err(|error| error.to_string());
            if let Ok(catalog) = &result {
                if std::env::var_os("CANDLE_DKR_TRACE").is_some() {
                    eprintln!("MISTRAL_DKR_CATALOG_EMBEDDED manifests={}", catalog.len());
                }
            }
            result
        })
        .as_ref()
        .map(Arc::clone)
        .map_err(|error| anyhow!("{error}"))
}
