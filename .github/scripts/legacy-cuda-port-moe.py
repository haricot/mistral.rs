#!/usr/bin/env python3
"""Semantic port of one precisely identified historical MoE commit.

Runs inside the stopped rebase at 251963b3. Keep maintained upstream files,
reintroduce the legacy grouped SIMT fallback, and preserve a physical MoE
test. Any unfamiliar conflicted file, symbol layout or commit fails closed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

COMMIT = "251963b32a10195fad344a182fb37e935b14ca1c"
conflicts = {
    "mistralrs-core/src/cuda/ffi.rs",
    "mistralrs-core/src/cuda/moe.rs",
    "mistralrs-core/src/cuda/moe_gemm.cu",
    "mistralrs-core/src/cuda/moe_gemm_wmma.cu",
    "mistralrs-core/src/cuda/moe_gemv.cu",
}

def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def replace_once(src: str, old: str, new: str, label: str) -> str:
    require(src.count(old) == 1, f"{label}: anchor appeared {src.count(old)} times")
    return src.replace(old, new, 1)

require(git("rev-parse", "REBASE_HEAD") == COMMIT, "Unexpected rebase commit")
unmerged = set(git("diff", "--name-only", "--diff-filter=U").splitlines())
require(unmerged == conflicts, f"Unexpected conflict set: {sorted(unmerged)}")
for path in sorted(conflicts):
    Path(path).write_text(subprocess.check_output(["git", "show", f":2:{path}"], text=True))

hfma_path = Path("mistralrs-core/src/cuda/moe_gemm_hfma2.cu")
require(hfma_path.is_file() and "mistralrs_moe_gemm_hfma2" in hfma_path.read_text(),
        "Historical SIMT kernel was not brought forward intact")

# The upstream FFI has no old transposed-weights ABI. Preserve that contract.
ffi_path = Path("mistralrs-core/src/cuda/ffi.rs")
ffi = ffi_path.read_text()
require("pub fn moe_gemm_hfma2(" not in ffi, "MoE fallback already exists")
ffi_marker = "    // MoE GEMV for decode phase (optimized for small batch sizes M <= 8)\n"
ffi_declaration = """
    // Pascal grouped SIMT fallback; the implementation is in moe_gemm_hfma2.cu.
    #[link_name = "mistralrs_moe_gemm_hfma2"]
    pub fn moe_gemm_hfma2(
        input: *const c_void,
        weights: *const c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut c_void,
        expert_offsets: *mut i32,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32,
        stream: i64,
    );

"""
ffi = replace_once(ffi, ffi_marker, ffi_declaration + ffi_marker, "MoE FFI")
ffi_path.write_text(ffi)

# Do not compile architectures that cannot run the WMMA grouped kernel.
build_path = Path("mistralrs-core/build.rs")
build = build_path.read_text()
build = replace_once(
    build,
    '    println!("cargo::rustc-check-cfg=cfg(has_flashinfer_gdn_sm90_kernel)");',
    '    println!("cargo::rustc-check-cfg=cfg(has_flashinfer_gdn_sm90_kernel)");\n'
    '    println!("cargo::rustc-check-cfg=cfg(has_moe_wmma)");',
    "Core build.rs custom cfg",
)
cap_anchor = '        let compute_cap = builder.get_compute_cap().unwrap_or(80);\n'
cap_gate = """
        if compute_cap >= 70 {
            println!("cargo:rustc-cfg=has_moe_wmma");
        } else {
            // Generic BF16 GEMM uses unsupported native BF16 half2 on Pascal.
            // Pascal goes through moe_gemm_hfma2.cu instead. The GEMV kernel
            // remains available for small decode batches.
            builder = builder.exclude(&["moe_gemm_wmma.cu", "moe_gemm.cu"]);
        }
"""
build = replace_once(build, cap_anchor, cap_anchor + cap_gate, "WMMA compile capability")
build_path.write_text(build)

# Explicitly select SIMT on SM61, preserving the maintained fast WMMA path for
# >=SM70 and native GEMV decode for small batches.
moe_path = Path("mistralrs-core/src/cuda/moe.rs")
moe = moe_path.read_text()
moe = replace_once(
    moe,
    "        let stream = dev.cuda_stream().cu_stream() as i64;",
    "        let stream = input.stream().cu_stream() as i64;",
    "MoE stream identity",
)
kernel_selection = """        let moe_func = if is_prefill {
            crate::cuda::ffi::moe_gemm_wmma
        } else if size_m_i32 <= GEMV_THRESHOLD {
            crate::cuda::ffi::moe_gemv
        } else {
            crate::cuda::ffi::moe_gemm
        };"""
replacement = """        // Pascal lacks WMMA. Both prefill and large decode require the SIMT
        // fallback, including BF16 multiplication in float accumulators.
        #[cfg(not(has_moe_wmma))]
        if is_prefill || size_m_i32 > GEMV_THRESHOLD {
            return launch_pascal_simt_moe::<T>(
                dev,
                input.slice(input_offset..).device_ptr(input.stream()).0 as *const c_void,
                weights.slice(weights_offset..).device_ptr(weights.stream()).0 as *const c_void,
                sorted_token_ids.slice(sti_offset..)
                    .device_ptr(sorted_token_ids.stream()).0 as *const i32,
                experts_ids.slice(ei_offset..).device_ptr(experts_ids.stream()).0 as *const i32,
                topk_weights_ptr,
                num_experts_i32, topk_i32, size_m_i32,
                size_n_i32, size_k_i32, data_type, stream,
            );
        }

        #[cfg(has_moe_wmma)]
        let moe_func = if is_prefill {
            crate::cuda::ffi::moe_gemm_wmma
        } else if size_m_i32 <= GEMV_THRESHOLD {
            crate::cuda::ffi::moe_gemv
        } else {
            crate::cuda::ffi::moe_gemm
        };
        #[cfg(not(has_moe_wmma))]
        let moe_func = crate::cuda::ffi::moe_gemv;"""
moe = replace_once(moe, kernel_selection, replacement, "MoE kernel selection")

fallback = """
#[cfg(feature = "cuda")]
fn launch_pascal_simt_moe<
    T: candle_core::cuda_backend::CudaDType
        + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
>(
    dev: &candle_core::cuda_backend::CudaDevice,
    input: *const core::ffi::c_void,
    weights: *const core::ffi::c_void,
    sorted_token_ids: *const i32,
    expert_ids: *const i32,
    topk_weights: *const f32,
    num_experts: i32,
    topk: i32,
    size_m: i32,
    size_n: i32,
    size_k: i32,
    dtype: i32,
    stream: i64,
) -> Result<Tensor> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

    let expert_offsets = unsafe { dev.alloc::<i32>((num_experts + 1) as usize) }?;
    let output = unsafe { dev.alloc::<T>((size_m * size_n) as usize) }?;
    unsafe {
        crate::cuda::ffi::moe_gemm_hfma2(
            input,
            weights,
            sorted_token_ids,
            expert_ids,
            topk_weights,
            output.device_ptr(output.stream()).0 as *mut core::ffi::c_void,
            expert_offsets.device_ptr(expert_offsets.stream()).0 as *mut i32,
            num_experts,
            topk,
            size_m,
            size_n,
            size_k,
            dtype,
            stream,
        );
    }
    let output = candle_core::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(Tensor::from((
        candle_core::Storage::Cuda(output),
        (size_m as usize, size_n as usize),
    )))
}

#[cfg(all(test, feature = "cuda"))]
mod pascal_moe_tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn legacy_sm61_moe_f16_bf16_prefill() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16] {
            let input = Tensor::ones((1, 16), dtype, &device)?;
            let weights = Tensor::ones((1, 4, 16), dtype, &device)?;
            let sorted = Tensor::new(&[0u32], &device)?;
            let experts = Tensor::new(&[0u32], &device)?;
            let output = moe_gemm(
                &input, &weights, &None, &sorted, &experts, 1, true
            )?;
            device.synchronize()?;
            let output = output.to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?.to_vec2::<f32>()?;
            assert_eq!(output.len(), 1);
            for val in &output[0] {
                assert!((*val - 16.0).abs() < 0.5, "{dtype:?}: {val}");
            }
        }
        Ok(())
    }
}
"""
require("fn launch_pascal_simt_moe" not in moe, "Already-ported Rust MoE implementation")
moe_path.write_text(moe + "\n" + fallback)

for path in sorted(conflicts):
    subprocess.run(["git", "add", path], check=True)
subprocess.run(["git", "add", "mistralrs-core/build.rs"], check=True)
subprocess.run(["git", "diff", "--check", "--cached"], check=True)
print("PORT_MOE_R2_READY: preserved upstream and added Pascal SIMT fallback + F16/BF16 MoE test")
