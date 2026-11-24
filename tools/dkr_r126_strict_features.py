#!/usr/bin/env python3
"""Audit resolved Cargo metadata for the strict native + typed catalogue candidate."""
import json
import sys

try:
    data = json.load(sys.stdin)
    ids = {p["id"]: p["name"] for p in data["packages"]}
    names = ("candle-core", "candle-nn", "candle-kernels",
             "mistralrs-core", "mistralrs-quant", "mistralrs-paged-attn",
             "mistralrs-kernel-runtime")
    graph = {}
    for name in names:
        nodes = [set(n["features"]) for n in data["resolve"]["nodes"] if ids[n["id"]] == name]
        if len(nodes) != 1:
            raise ValueError(f"expected unique resolved package {name}")
        graph[name] = nodes[0]
        forbidden = {f for f in nodes[0] if f.startswith("cuda-legacy")
                     or f in ("cuda-dkr-s31", "dkr-sm61-experimental", "cutile")}
        if forbidden:
            raise ValueError(f"{name} has forbidden features {sorted(forbidden)}")
    required = {
        "candle-core": {"cuda-dkr-qualified-default", "cuda-dkr-embedded", "cuda-dkr-typed-image"},
        "candle-nn": {"cuda-dkr-qualified-default", "cuda-dkr-embedded"},
        "candle-kernels": {"cuda-dkr-embedded"},
        "mistralrs-core": {"dkr-sm61-cubin-qualified", "dkr-typed-operations", "dkr-fp4-dequant"},
        "mistralrs-kernel-runtime": {"shared-registry"},
    }
    for name, wanted in required.items():
        missing = wanted - graph[name]
        if missing:
            raise ValueError(f"{name} lacks {sorted(missing)}")
    if "--paged-fp8-io" in sys.argv[1:]:
        for name in ("mistralrs-core", "mistralrs-paged-attn"):
            missing = {"dkr-paged-catalog-only", "dkr-kv-fp8-io"} - graph[name]
            if missing:
                raise ValueError(f"{name} lacks {sorted(missing)}")
        print("DKR_R126_S9B2_FEATURES=PASS paged_catalog_only=true fp8_io=true")
    if "--prefill-sinks" in sys.argv[1:]:
        for name in ("mistralrs-core", "mistralrs-paged-attn"):
            missing = {"dkr-paged-catalog-only", "dkr-paged-options", "dkr-prefill-sinks"} - graph[name]
            if missing:
                raise ValueError(f"{name} lacks {sorted(missing)}")
        print("DKR_R126_S9B3_FEATURES=PASS paged_options=true prefill_sinks=true")
    if "--mla-cache-io" in sys.argv[1:]:
        for name in ("mistralrs-core", "mistralrs-paged-attn"):
            missing = {"dkr-paged-catalog-only", "dkr-mla-cache-io"} - graph[name]
            if missing:
                raise ValueError(f"{name} lacks {sorted(missing)}")
        print("DKR_R126_S9B4A_FEATURES=PASS mla_cache_io=true")
    if "--flashinfer-hnd-io" in sys.argv[1:]:
        for name in ("mistralrs-core", "mistralrs-paged-attn"):
            missing = {"dkr-paged-catalog-only", "dkr-flashinfer-hnd-io"} - graph[name]
            if missing:
                raise ValueError(f"{name} lacks {sorted(missing)}")
        print("DKR_R126_S9B4B_FEATURES=PASS flashinfer_hnd_io=true")
    if "--mla-decode" in sys.argv[1:]:
        for name in ("mistralrs-core", "mistralrs-paged-attn"):
            missing = {"dkr-paged-catalog-only", "dkr-mla-decode"} - graph[name]
            if missing:
                raise ValueError(f"{name} lacks {sorted(missing)}")
        print("DKR_R126_S9B4C_FEATURES=PASS mla_decode=true")
    print("DKR_R126_STRICT_FEATURES=PASS candle_embedded=true typed=true legacy_flags=false")
    print("DKR_R126_STATIC_PRODUCER_ELIMINATION=NOT_QUALIFIED mistral_build_scripts=remain")
except (OSError, ValueError, KeyError, TypeError) as exc:
    print(f"DKR_R126_STRICT_FEATURES=FAIL reason={exc}", file=sys.stderr)
    sys.exit(1)
