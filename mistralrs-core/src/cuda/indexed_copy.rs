#[cfg(feature = "dkr-indexed-copy")]
#[path = "indexed_copy_dynamic.rs"]
mod provider;

#[cfg(not(feature = "dkr-indexed-copy"))]
#[path = "indexed_copy_legacy.rs"]
mod provider;

pub(crate) use provider::copy_rows;
