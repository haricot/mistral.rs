#!/usr/bin/env python3
"""Conservative source-level coverage audit of CUDA compilation units."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

QUANT_GROUPS = {
    "afq": "quantized_linear",
    "bitsandbytes": "quantized_linear",
    "blockwise_fp8": "fp8_linear_and_norm",
    "cutlass_moe": "moe",
    "embedding": "embedding",
    "gemv": "dense_linear",
    "gguf": "quantized_linear",
    "gguf_affine_packed": "quantized_linear",
    "gptq": "quantized_linear",
    "hqq": "quantized_linear",
    "indexed_moe": "moe",
    "lora": "lora",
    "marlin": "quantized_linear",
    "memory": "memory_transfer",
    "mmq_gguf": "quantized_linear",
    "mmvq_gguf": "quantized_linear",
    "moe": "moe",
    "moe_grouped": "moe",
    "mxfp4": "fp4_linear",
    "mxfp8": "fp8_linear_and_norm",
    "nvfp4_cutlass": "fp4_linear",
    "nvfp4_decode": "fp4_linear",
    "nvfp4_w4a16": "fp4_linear",
    "ops": "fused_tensor_and_convolution",
    "rotary": "rotary_position",
    "routed_gemv": "moe",
    "scalar_fp8": "fp8_linear_and_norm",
}

CORE_GROUPS = {
    "attention_prep": "attention",
    "dflash_context": "attention",
    "dflash_selector": "attention",
    "dynamic_conv": "convolution",
    "gated_add": "fused_tensor_and_convolution",
    "gdn": "state_space",
    "gdn_mma": "state_space",
    "graph": "graph_and_sync",
    "hyper_connection": "fused_tensor_and_convolution",
    "indexed_copy": "memory_transfer",
    "input_packing": "layout_and_packing",
    "moe_gemm": "moe",
    "moe_gemm_hfma2": "moe",
    "moe_gemm_wmma": "moe",
    "moe_gemv": "moe",
    "qwen4_exp": "fused_tensor_and_convolution",
    "rounded_rope": "rotary_position",
    "sort": "sampling_and_sort",
    "speculative_rejection": "sampling_and_sort",
    "ssm": "state_space",
}

CANDLE_ROOT_GROUPS = {
    "affine": "elementwise",
    "binary": "elementwise",
    "cast": "cast",
    "conv": "convolution",
    "fill": "memory_transfer",
    "indexing": "indexing",
    "mmvq_gguf": "quantized_linear",
    "quantized": "quantized_linear",
    "reduce": "reduction",
    "sort": "sampling_and_sort",
    "ternary": "elementwise",
    "unary": "elementwise",
}

BACKEND_METHOD_GROUPS = {
    "try_clone": "memory_transfer",
    "dtype": "host_metadata",
    "device": "host_metadata",
    "to_cpu_storage": "device_transfer",
    "affine": "elementwise",
    "powf": "elementwise",
    "elu": "elementwise",
    "reduce_op": "reduction",
    "cmp": "elementwise",
    "to_dtype": "cast",
    "unary_impl": "elementwise",
    "binary_impl": "elementwise",
    "where_cond": "elementwise",
    "conv1d": "convolution",
    "conv_transpose1d": "convolution",
    "conv2d": "convolution",
    "conv_transpose2d": "convolution",
    "avg_pool2d": "pooling",
    "max_pool2d": "pooling",
    "upsample_nearest1d": "resampling",
    "upsample_nearest2d": "resampling",
    "upsample_bilinear2d": "resampling",
    "gather": "indexing",
    "scatter_set": "indexing",
    "scatter_add_set": "indexing",
    "index_select": "indexing",
    "index_add": "indexing",
    "matmul": "dense_linear",
    "copy_strided_src": "memory_transfer",
    "copy2d": "memory_transfer",
    "const_set": "memory_transfer",
    "new": "device_management",
    "location": "host_metadata",
    "same_device": "host_metadata",
    "zeros_impl": "memory_transfer",
    "alloc_uninit": "device_management",
    "storage_from_slice": "device_transfer",
    "storage_from_cpu_storage": "device_transfer",
    "storage_from_cpu_storage_owned": "device_transfer",
    "rand_uniform": "random_number_generation",
    "rand_normal": "random_number_generation",
    "set_seed": "device_management",
    "get_current_seed": "device_management",
    "synchronize": "device_management",
}


def mistral_family(path: Path) -> str | None:
    parts = path.parts
    if len(parts) >= 4 and parts[:2] == ("mistralrs-quant", "kernels"):
        return QUANT_GROUPS.get(parts[2])
    if len(parts) == 4 and parts[:3] == ("mistralrs-core", "src", "cuda"):
        return CORE_GROUPS.get(path.stem)
    if parts[:3] == ("mistralrs-paged-attn", "src", "cuda"):
        if any(token in path.name for token in ("cache", "kvscale", "copy_blocks")):
            return "kv_cache"
        return "attention"
    if parts[:2] == ("mistralrs-flash-attn", "kernels"):
        return "attention_sm80_only"
    if parts[:2] == ("legacy-provider-poc", "f32"):
        return "proof_fixture_not_production"
    if parts[:2] in (
        ("mistralrs-quant", "third_party"),
        ("mistralrs-core", "third_party"),
        ("mistralrs-paged-attn", "third_party"),
    ):
        return "vendor_specialized"
    return None


def candle_family(path: Path) -> str | None:
    if path.parts[0] == "mmq_gguf":
        return "quantized_linear"
    if path.parts[0] == "moe":
        return "moe"
    if len(path.parts) == 1:
        return CANDLE_ROOT_GROUPS.get(path.stem)
    return None


def audit(candle_dir: Path | None) -> dict:
    records = []
    backend_methods = []
    for base in (
        "mistralrs-core", "mistralrs-quant", "mistralrs-paged-attn",
        "mistralrs-vision", "mistralrs-flash-attn", "legacy-provider-poc",
    ):
        for file in sorted((ROOT / base).rglob("*.cu")):
            path = file.relative_to(ROOT)
            records.append({"source": str(path), "family": mistral_family(path)})
    if candle_dir is not None:
        candle_root = candle_dir.resolve()
        kernel_root = candle_root / "candle-kernels" / "src"
        if not kernel_root.is_dir():
            raise ValueError(f"Candle kernel source not found: {kernel_root}")
        for file in sorted(kernel_root.rglob("*.cu")):
            path = file.relative_to(kernel_root)
            records.append({"source": f"candle-kernels/src/{path}", "family": candle_family(path)})
        backend_source = candle_root / "candle-core" / "src" / "backend.rs"
        method_names = re.findall(
            r"^\s*(?:unsafe\s+)?fn\s+([a-z][a-z_0-9]*)\s*(?:<[^>]+>)?\s*\(",
            backend_source.read_text(),
            re.MULTILINE,
        )
        backend_methods = [
            {"name": name, "family": BACKEND_METHOD_GROUPS.get(name)}
            for name in method_names
        ]

    missing = [record["source"] for record in records if record["family"] is None]
    unmapped_methods = [row["name"] for row in backend_methods if row["family"] is None]
    return {
        "schema": "DKR-R1-CUDA-SOURCE-INVENTORY-V1",
        "scope": "All checked-in CUDA .cu units plus Candle BackendStorage/BackendDevice trait methods; vendor libraries/model-level dispatch require separate gates",
        "candle_checked": candle_dir is not None,
        "files": len(records),
        "families": dict(sorted(Counter(row["family"] for row in records if row["family"]).items())),
        "unclassified": missing,
        "backend_methods_checked": bool(backend_methods),
        "backend_methods": len(backend_methods),
        "backend_method_families": dict(sorted(Counter(row["family"] for row in backend_methods if row["family"]).items())),
        "unclassified_backend_methods": unmapped_methods,
        "status": "PASS_SOURCE_AND_BACKEND_CLASSIFICATION" if records and not missing and not unmapped_methods else "REVIEW_UNCLASSIFIED_OPERATION",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candle-dir", type=Path, help="checkout pinned to cuda_legacy revision")
    args = parser.parse_args()
    result = audit(args.candle_dir)
    print(json.dumps(result, indent=2))
    if result["unclassified"] or result["unclassified_backend_methods"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
