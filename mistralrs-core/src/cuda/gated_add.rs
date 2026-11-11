#[cfg(feature = "dkr-gated-add")]
#[path = "gated_add_dynamic.rs"]
mod provider;

#[cfg(not(feature = "dkr-gated-add"))]
#[path = "gated_add_legacy.rs"]
mod provider;

pub(crate) use provider::try_sigmoid_gated_add;
