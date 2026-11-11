#!/usr/bin/env python3
"""Account for CUDA source families and Candle backend operations without building Rust."""

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


# A source classification is never evidence of an executable DKR provider.
# "device_candidate" means external code and a typed adapter still need qualification.
ROUTES = {
    "dense_linear": "device_candidate",
    "quantized_linear": "device_candidate",
    "fp8_linear_and_norm": "device_candidate",
    "fp4_linear": "device_candidate",
    "moe": "device_candidate",
    "attention": "device_candidate",
    "kv_cache": "device_candidate",
    "convolution": "device_candidate",
    "state_space": "device_candidate",
    "fused_tensor_and_convolution": "device_candidate",
    "elementwise": "device_candidate",
    "reduction": "device_candidate",
    "rotary_position": "device_candidate",
    "embedding": "device_candidate",
    "layout_and_packing": "device_candidate",
    "cast": "device_candidate",
    "indexing": "device_candidate",
    "sampling_and_sort": "device_candidate",
    "lora": "device_candidate",
    "pooling": "device_candidate",
    "resampling": "device_candidate",
    "random_number_generation": "device_candidate",
    "memory_transfer": "host_and_device",
    "device_transfer": "host_and_device",
    "graph_and_sync": "host_runtime",
    "device_management": "host_runtime",
    "host_metadata": "host_runtime",
    "vendor_specialized": "vendor_provider",
    "attention_sm80_only": "unsupported_sm61",
    "proof_fixture_not_production": "historical_fixture",
}

SM61_EXTERNAL_CANDIDATES = {
    "elementwise": ["copy", "affine", "add", "sub", "mul", "div", "exp", "tanh", "silu", "gelu"],
    "reduction": ["reduce_sum", "reduce_max", "softmax", "rms_norm", "rms_norm_scaled", "residual_rms_norm", "residual_rms_norm_scaled"],
    "fused_tensor_and_convolution": ["causal_mask"],
    "sampling_and_sort": ["logits_penalty_sparse", "logits_bias_sparse"],
    "memory_transfer": ["copy"],
    "indexing": ["gather_rows"],
    "rotary_position": ["rope"],
}


def matrix(candle_dir: Path | None) -> dict:
    inventory = audit(candle_dir)
    known_families = (
        set(QUANT_GROUPS.values())
        | set(CORE_GROUPS.values())
        | set(CANDLE_ROOT_GROUPS.values())
        | set(BACKEND_METHOD_GROUPS.values())
        | {"attention", "attention_sm80_only", "kv_cache", "proof_fixture_not_production", "vendor_specialized"}
    )
    unmapped_families = sorted(known_families - ROUTES.keys())
    extraneous_families = sorted(ROUTES.keys() - known_families)

    families = []
    for name in sorted(known_families):
        families.append({
            "family": name,
            "route": ROUTES.get(name),
            "source_units": inventory["families"].get(name, 0),
            "backend_methods": sorted(k for k, v in BACKEND_METHOD_GROUPS.items() if v == name),
            "sm61_source_candidates": SM61_EXTERNAL_CANDIDATES.get(name, []),
            "dkr_execution": "not_qualified",
        })

    gaps = (
        unmapped_families
        + extraneous_families
        + inventory["unclassified"]
        + inventory["unclassified_backend_methods"]
    )
    return {
        "schema": "DKR-R2-ALL-OPERATIONS-MATRIX-V1",
        "candle_pin": "d99f6d77e507688bd9e4b654fbe20ac9e18238fc",
        "source_inventory": inventory,
        "known_backend_method_count": len(BACKEND_METHOD_GROUPS),
        "unmapped_families": unmapped_families,
        "extraneous_families": extraneous_families,
        "families": families,
        "status": (
            "FAIL_UNACCOUNTED" if gaps else
            "PASS_ACCOUNTED_NOT_GPU_QUALIFIED" if candle_dir is not None else
            "PARTIAL_CANDLE_NOT_CHECKED"
        ),
        "gpu_qualified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candle-dir", type=Path, help="exact pinned cuda_legacy Candle checkout")
    parser.add_argument("--output", type=Path, help="write JSON to an explicit path")
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
