//! Binary-image policy, shared in spelling with Candle's native DKR route.
//! Fixed on first use: switching an image mode requires a process restart.
use anyhow::{bail, Result};
use std::sync::OnceLock;

pub const IMAGE_MODE_ENV: &str = "CANDLE_DKR_IMAGE_MODE";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageMode {
    Auto,
    Cubin,
    Ptx,
}

impl ImageMode {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value {
            None | Some("auto") => Ok(Self::Auto),
            Some("cubin") => Ok(Self::Cubin),
            Some("ptx") => Ok(Self::Ptx),
            _ => bail!("CANDLE_DKR_IMAGE_MODE must be auto, cubin or ptx"),
        }
    }

    pub fn current() -> Result<Self> {
        static MODE: OnceLock<std::result::Result<ImageMode, String>> = OnceLock::new();
        MODE.get_or_init(|| {
            let setting = std::env::var(IMAGE_MODE_ENV)
                .map(Some)
                .or_else(|error| match error {
                    std::env::VarError::NotPresent => Ok(None),
                    other => Err(other),
                })
                .map_err(|_| format!("{IMAGE_MODE_ENV} must be UTF-8"))?;
            Self::parse(setting.as_deref()).map_err(|error| error.to_string())
        })
        .as_ref()
        .copied()
        .map_err(|error| anyhow::anyhow!("{error}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_policy_is_strict_and_has_explicit_rollback() {
        assert_eq!(ImageMode::parse(None).unwrap(), ImageMode::Auto);
        assert_eq!(ImageMode::parse(Some("auto")).unwrap(), ImageMode::Auto);
        assert_eq!(ImageMode::parse(Some("cubin")).unwrap(), ImageMode::Cubin);
        assert_eq!(ImageMode::parse(Some("ptx")).unwrap(), ImageMode::Ptx);
        for bad in ["", "PTX", "fallback", "sm61", "auto "] {
            assert!(ImageMode::parse(Some(bad)).is_err());
        }
    }
}
