---
title: Configure model topology
description: Per-layer placement and quantization via a YAML topology file.
sidebar:
  order: 7
---

Topology is a per-layer placement and quantization mechanism. A YAML file specifies, per layer range, the device and quantization to use.

Most cases do not need topology. Defaults work for typical hardware; `mistralrs tune` covers common optimization.

## Config

A YAML file keyed by `start-end` layer-range selectors:

```yaml
runtime:
  cpu_moe: true
  profile: true
  gguf_cpu_moe_expert_cache: 0
  gguf_cpu_moe_q4_1_expert_cache: 128
  gguf_cpu_moe_q4k_expert_cache: 256
  gguf_cpu_moe_parallel_topk: true

0-16:
  device: cuda[0]
  isq: q4k
16-32:
  device: cuda[1]
  isq: q4k
32-40:
  device: cpu
  isq: q8_0
```

Layers outside any range use defaults. `device` is a CUDA (`cuda[N]`), Metal (`metal[N]`), or CPU (`cpu`) specifier. `isq` accepts any ISQ type name recognized by `--isq`.
The optional `runtime` block enables model runtime knobs such as CPU MoE, profiling, and GGUF CPU MoE expert caches. Keep the generic `gguf_cpu_moe_expert_cache` low because it caches dequantized fallback experts; prefer dtype-specific caches such as `gguf_cpu_moe_q4_1_expert_cache` and `gguf_cpu_moe_q4k_expert_cache`.

Pass with `--topology`:

```bash
mistralrs serve --topology topology.yaml -m <model>
```

## Notes

Embedding layers, LM head, and pre/post-norm are not individually addressable; they follow the first or last transformer layer's placement.

For an introduction to per-layer quantization tradeoffs, see [the explanation page on quantization tradeoffs](/mistral.rs/explanation/quantization-tradeoffs/).
