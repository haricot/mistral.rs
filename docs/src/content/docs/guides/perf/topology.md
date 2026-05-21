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

## Submodels

Supported multimodal models can use a `submodels` block to select which modality-specific paths are loaded. For example, Qwen3.5 MoE can skip the vision tower for text-only runs:

```yaml
submodels:
  vision:
    load: false

0-40:
  device: cuda[0]
```

`enabled: false` is accepted as an alias for `load: false`. Image or video inputs fail with a clear error while the vision submodel is disabled.

Pass with `--topology`:

```bash
mistralrs serve --topology topology.yaml -m <model>
```

## Notes

Embedding layers, LM head, and pre/post-norm are not individually addressable; they follow the first or last transformer layer's placement.

For an introduction to per-layer quantization tradeoffs, see [the explanation page on quantization tradeoffs](/mistral.rs/explanation/quantization-tradeoffs/).
