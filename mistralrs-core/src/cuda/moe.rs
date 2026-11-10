#[cfg(feature = "dkr-moe")]
#[path = "moe_dynamic.rs"]
mod provider;

#[cfg(not(feature = "dkr-moe"))]
#[path = "moe_legacy.rs"]
mod provider;

pub use provider::moe_gemm;
