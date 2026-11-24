use anyhow::Result;
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf};

#[allow(dead_code)]
#[path = "src/universal.rs"]
mod universal;
use universal::{KernelCatalog, KernelManifest, Parameter};

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn manifest_source(m: &KernelManifest) -> String {
    let arguments = m
        .arguments
        .iter()
        .map(|arg| match arg {
            Parameter::Tensor {
                name,
                dtype,
                rank,
                access,
                representation,
            } => {
                let representation = representation
                    .as_ref()
                    .map(|value| format!("Some({value:?}.into())"))
                    .unwrap_or("None".into());
                format!(
                    "Parameter::Tensor {{ name:{name:?}.into(), dtype:ElementType::{dtype:?}, \
                     rank:{rank}, access:Access::{access:?}, representation:{representation} }}"
                )
            }
            Parameter::Scalar { name, dtype } => format!(
                "Parameter::Scalar {{ name:{name:?}.into(), dtype:ElementType::{dtype:?} }}"
            ),
        })
        .collect::<Vec<_>>()
        .join(",");
    let cubin_artifact = m.cubin_artifact.as_ref()
        .map(|path| format!("Some(std::path::PathBuf::from({:?}))", path.to_str().unwrap()))
        .unwrap_or_else(|| "None".into());
    let cubin_sha256 = m.cubin_sha256.as_ref()
        .map(|sha| format!("Some({sha:?}.into())"))
        .unwrap_or_else(|| "None".into());
    format!(
        "KernelManifest {{ schema:{:?}.into(), abi:{:?}.into(), operation:{:?}.into(), \
         implementation_id:{:?}.into(), artifact:{:?}.into(), sha256:{:?}.into(), \
         cubin_artifact:{cubin_artifact}, cubin_sha256:{cubin_sha256}, entry:{:?}.into(), \
         target_cc:{}, min_cc:{}, max_cc:{}, block:{:?}, shared_mem_bytes:{}, arguments:vec![{arguments}] }}",
        m.schema, m.abi, m.operation, m.implementation_id, m.artifact, m.sha256, m.entry,
        m.target_cc, m.min_cc, m.max_cc, m.block, m.shared_mem_bytes
    )
}

fn main() -> Result<()> {
    const EMBED_ENV: &str = "DKR_EMBED_TYPED_DIR";
    println!("cargo:rerun-if-env-changed={EMBED_ENV}");
    println!("cargo:rerun-if-changed=src/universal.rs");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut rows = Vec::new();
    let mut root_literal = "None".into();
    if let Some(root) = env::var_os(EMBED_ENV) {
        let root = PathBuf::from(root).canonicalize()?;
        root_literal = format!("Some({:?})", root.to_str().unwrap());
        let catalog = KernelCatalog::discover(&root)?;
        let mut modules = catalog.verified_modules().collect::<Vec<_>>();
        modules.sort_by_key(|m| {
            (
                &m.manifest.operation,
                &m.manifest.implementation_id,
                m.manifest.target_cc,
            )
        });
        let mut pending = vec![root.clone()];
        while let Some(dir) = pending.pop() {
            println!("cargo:rerun-if-changed={}", dir.display());
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    pending.push(entry.path());
                }
            }
        }
        let mut written = std::collections::BTreeSet::new();
        for module in modules {
            let m = &module.manifest;
            let copy = out.join(format!("dkr-typed-{}.ptx", m.sha256.to_ascii_lowercase()));
            if written.insert(copy.clone()) { fs::write(&copy, module.ptx.as_bytes())?; }
            let cubin = if let Some(image) = &module.cubin {
                let hash = m.cubin_sha256.as_ref().expect("validated paired CUBIN hash");
                let image_copy = out.join(format!("dkr-typed-{}.cubin", hash.to_ascii_lowercase()));
                if written.insert(image_copy.clone()) { fs::write(&image_copy, image.as_ref())?; }
                format!("Some(SharedImage::Embedded(&include_bytes!({image_copy:?})[..]))")
            } else {
                "None".into()
            };
            rows.push(format!(
                "VerifiedKernel {{ manifest:{}, ptx:SharedImage::Embedded(include_str!({copy:?})), cubin:{cubin}, \
                 manifest_sha256:{:?}.into() }}",
                manifest_source(m),
                module.manifest_sha256
            ));
        }
    }
    let source = if rows.is_empty() {
        "Err(anyhow!(\"typed artifacts were not embedded\"))".into()
    } else {
        format!("use crate::universal::{{Access,ElementType,KernelManifest,Parameter,VerifiedKernel,SharedImage}}; \
                 KernelCatalog::from_build_verified(vec![{}])", rows.join(",\n"))
    };
    fs::write(out.join("dkr_typed_embedded.rs"), format!(
        "pub const ROOT: Option<&str> = {root_literal};\nfn load() -> Result<KernelCatalog> {{ {source} }}\n"
    ))?;
    Ok(())
}
