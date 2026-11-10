#!/usr/bin/env python3
"""Conservative source-level coverage audit of CUDA compilation units."""

import argparse
import json
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
    for base in (
        "mistralrs-core", "mistralrs-quant", "mistralrs-paged-attn",
        "mistralrs-vision", "mistralrs-flash-attn", "legacy-provider-poc",
    ):
        for file in sorted((ROOT / base).rglob("*.cu")):
            path = file.relative_to(ROOT)
            records.append({"source": str(path), "family": mistral_family(path)})
    if candle_dir is not None:
        kernel_root = candle_dir.resolve() / "candle-kernels" / "src"
        if not kernel_root.is_dir():
            raise ValueError(f"Candle kernel source not found: {kernel_root}")
        for file in sorted(kernel_root.rglob("*.cu")):
            path = file.relative_to(kernel_root)
            records.append({"source": f"candle-kernels/src/{path}", "family": candle_family(path)})

    missing = [record["source"] for record in records if record["family"] is None]
    return {
        "schema": "DKR-R1-CUDA-SOURCE-INVENTORY-V1",
        "scope": "CUDA .cu compilation units only; Rust ops/vendor libraries require separate coverage gates",
        "candle_checked": candle_dir is not None,
        "files": len(records),
        "families": dict(sorted(Counter(row["family"] for row in records if row["family"]).items())),
        "unclassified": missing,
        "status": "PASS_SOURCE_CLASSIFICATION" if records and not missing else "REVIEW_UNCLASSIFIED_SOURCE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candle-dir", type=Path, help="checkout pinned to cuda_legacy revision")
    args = parser.parse_args()
    result = audit(args.candle_dir)
    print(json.dumps(result, indent=2))
    if result["unclassified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
