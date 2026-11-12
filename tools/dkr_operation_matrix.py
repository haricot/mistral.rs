#!/usr/bin/env python3
"""Conservative, non-compiling DKR operation coverage report.

A classified .cu file is *not* a migrated kernel. A committed adapter is not
an SM61-qualified kernel. This report keeps those states separate.
"""

import argparse
import json
from pathlib import Path

from dkr_r1_inventory import (
    BACKEND_METHOD_GROUPS,
    CANDLE_ROOT_GROUPS,
    CORE_GROUPS,
    QUANT_GROUPS,
    audit,
)

ROUTES = {
    "dense_linear": "device_or_vendor",
    "quantized_linear": "device_or_vendor",
    "fp8_linear_and_norm": "device_or_vendor",
    "fp4_linear": "device_or_vendor",
    "moe": "device_or_vendor",
    "attention": "device_or_vendor",
    "kv_cache": "device_or_vendor",
    "convolution": "device_or_vendor",
    "state_space": "device_or_vendor",
    "fused_tensor_and_convolution": "device_or_vendor",
    "elementwise": "device_candidate",
    "reduction": "device_candidate",
    "rotary_position": "device_candidate",
    "embedding": "device_candidate",
    "layout_and_packing": "device_candidate",
    "cast": "device_candidate",
    "indexing": "device_candidate",
    "sampling_and_sort": "device_candidate",
    "lora": "device_or_vendor",
    "pooling": "device_candidate",
    "resampling": "device_candidate",
    "random_number_generation": "device_candidate",
    "memory_transfer": "host_and_device",
    "device_transfer": "host_and_device",
    "graph_and_sync": "host_runtime_deferred",
    "device_management": "host_runtime",
    "host_metadata": "host_runtime",
    "vendor_specialized": "vendor_adapter",
    "attention_sm80_only": "unsupported_sm61",
    "proof_fixture_not_production": "historical_fixture",
}

# Each label below refers to a specific external typed adapter and PTX
# package checked into this branch; NONE is GPU-qualified by this matrix.
STAGED_OPERATIONS = {
    "dense_linear": ["linear.gemv.plain.b128.v1", "linear.gemv.bias.b128.v1", "linear.gemm.plain.t16.v1", "linear.gemm.bias.t16.v1"],
    "kv_cache": ["kv_cache.scale_update.atomic_max.v1", "kv_cache.gather.vllm.contiguous.v1"],
    "fp8_linear_and_norm": ["quant.fp8_e4m3.encode.v1", "quant.fp8_e4m3.decode.v1"],
    "fp4_linear": ["quant.fp4.dequant.nvfp4.v1", "quant.fp4.dequant.mxfp4.v1"],
    "quantized_decode_gemv": ["quant.fp8.blockwise.gemv.b128.v1", "quant.mxfp4.gemv.plain.b128.v1", "quant.mxfp4.gemv.bias.b128.v1"],
    "moe": ["moe.grouped_gemm.v1"],
    "reduction": [
        "moe.weighted_reduce.v1",
        "attention.softmax_sinks.v1",
        "tensor.add_rms_norm.v1",
        "tensor.rms_residual.v1",
        "tensor.rms_residual_then_rms.v1",
    ],
    "fused_tensor_and_convolution": [
        "tensor.sigmoid_gated_add.v1",
        "attention.causal_mask_f32.inplace.v1",
    ],
    "sampling_and_sort": [
        "sampling.logits.penalties_sparse.f32.v1",
        "sampling.logits.bias_sparse.f32.v1",
    ],
    "memory_transfer": ["tensor.indexed_row_copy.v1", "layout.pad_decode_u32.v1"],
    "state_space": ["state_space.ssm_scan_f32.v1"],
    "rotary_position": ["rotary.rounded_rms_rope.v1"],
}

# Deliberately explicit: the operation entries above do not certify
# source-unit completeness or Candle backend trait coverage.
KNOWN_UNMIGRATED = {
    "dense_linear": "GEMM/GEMV F32/F16/BF16 and cuBLAS provider ABI",
    "quantized_linear": "GGUF/MMQ/MMVQ, GPTQ, HQQ, Marlin, AFQ and other weight formats",
    "fp8_linear_and_norm": "packed scale descriptors and SM61 SIMT/DP4A paths",
    "fp4_linear": "NVFP4/MXFP4 packed metadata and SM61 SIMT paths",
    "moe": "router/top-k/routed expert paths, resident expert cache",
    "attention": "SDPA, paged attention, MLA, SM61 alternatives to SM80 kernels",
    "kv_cache": "copy, reshape, gather and quantized KV-scale paths",
    "convolution": "Conv1D/Conv2D, dynamic and LFM3 convolution",
    "state_space": "GDN variants, fused recurrent and hyperconnection paths",
    "reduction": "LayerNorm, RMSNorm strided 4D, general reductions/softmax",
    "fused_tensor_and_convolution": "remaining fused activation/conv kernels",
    "elementwise": "unary, binary, ternary, affine and cast families",
    "rotary_position": "RoPE and normalized fused rotary",
    "embedding": "lookup/gather and layout-specialized embedding",
    "layout_and_packing": "multi-buffer input packing and transforms",
    "indexing": "gather, scatter, index-select and index-add",
    "sampling_and_sort": "top-k, sort, top-1, speculative rejection",
    "lora": "dynamic/fused/routed LoRA",
    "pooling": "average/max pool",
    "resampling": "nearest/bilinear interpolation",
    "random_number_generation": "uniform/normal RNG with seed and stream contract",
}

def matrix(candle_dir: Path | None) -> dict:
    inventory = audit(candle_dir)
    known = (
        set(QUANT_GROUPS.values())
        | set(CORE_GROUPS.values())
        | set(CANDLE_ROOT_GROUPS.values())
        | set(BACKEND_METHOD_GROUPS.values())
        | {
            "attention", "attention_sm80_only", "kv_cache",
            "proof_fixture_not_production", "vendor_specialized",
        }
    )
    missing = sorted(known - ROUTES.keys())
    extra = sorted(ROUTES.keys() - known)
    families = []
    for name in sorted(known):
        staged = STAGED_OPERATIONS.get(name, [])
        families.append({
            "family": name,
            "route": ROUTES.get(name),
            "source_units_classified": inventory["families"].get(name, 0),
            "candle_backend_methods_classified": [
                method for method, family in BACKEND_METHOD_GROUPS.items()
                if family == name
            ],
            "staged_external_operations": staged,
            "source_family_complete": False,
            "sm61_qualified": False,
            "remaining_scope": KNOWN_UNMIGRATED.get(name, "external adapter or host contract audit"),
            "state": "partially_staged" if staged else "host_or_vendor"
            if ROUTES.get(name, "").startswith(("host", "vendor"))
            else "source_classified_only",
        })

    classification_errors = (
        missing + extra + inventory["unclassified"]
        + inventory["unclassified_backend_methods"]
    )
    return {
        "schema": "DKR-OPERATION-COVERAGE-V2",
        "source_inventory": inventory,
        "candle_revision_pin": "d99f6d77e507688bd9e4b654fbe20ac9e18238fc",
        "candle_checked": candle_dir is not None,
        "packages_staged": 13,
        "families": families,
        "unmapped_families": missing,
        "extraneous_families": extra,
        "gpu_qualified": False,
        "cuda_graphs": "deferred",
        "status": "FAIL_UNACCOUNTED" if classification_errors else (
            "INVENTORY_COMPLETE_MIGRATION_INCOMPLETE"
            if candle_dir is not None else
            "PARTIAL_CANDLE_CHECKOUT_REQUIRED_MIGRATION_INCOMPLETE"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candle-dir", type=Path, help="pinned cuda_legacy Candle checkout")
    parser.add_argument("--output", type=Path, help="write JSON to explicit file")
    args = parser.parse_args()
    result = matrix(args.candle_dir)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
    if result["status"] == "FAIL_UNACCOUNTED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
