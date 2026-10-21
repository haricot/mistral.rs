use std::{
    env,
    path::{Path, PathBuf},
};

const CUDA_ROOT_VARS: [&str; 4] = [
    "CUDA_PATH",
    "CUDA_HOME",
    "CUDA_ROOT",
    "CUDA_TOOLKIT_ROOT_DIR",
];

fn contains_cupti(dir: &Path) -> bool {
    dir.join("libcupti.so").exists()
        || dir.join("libcupti.so.12").exists()
        || dir.join("libcupti_static.a").exists()
}

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_FLOW_TUNER_CUPTI");
    for var in CUDA_ROOT_VARS {
        println!("cargo:rerun-if-env-changed={var}");
    }

    if env::var_os("CARGO_FEATURE_FLOW_TUNER_CUPTI").is_none() {
        return;
    }

    let roots: Vec<PathBuf> = CUDA_ROOT_VARS
        .iter()
        .filter_map(|var| env::var_os(var).map(PathBuf::from))
        .collect();

    let mut searched = Vec::new();
    for root in &roots {
        for relative in [
            "extras/CUPTI/lib64",
            "extras/CUPTI/lib",
            "targets/x86_64-linux/lib",
            "lib64",
            "lib",
        ] {
            let candidate = root.join(relative);
            searched.push(candidate.clone());
            if !contains_cupti(&candidate) {
                continue;
            }

            println!("cargo:rustc-link-search=native={}", candidate.display());
            println!(
                "cargo:rustc-link-arg-bin=mistralrs=-Wl,-rpath,{}",
                candidate.display()
            );
            println!(
                "cargo:warning=Flow Tuner CUPTI library directory: {}",
                candidate.display()
            );
            return;
        }
    }

    panic!(
        "flow-tuner-cupti is enabled but libcupti was not found. \
         Set CUDA_PATH to the CUDA 12.9 toolkit root. Searched: {searched:?}"
    );
}
