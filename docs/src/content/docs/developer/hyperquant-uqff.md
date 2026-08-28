---
title: HyperQuant UQFF schema
description: Development contract for the HQZ4 weight codec and its UQFF representation.
---

HQZ4 is the first HyperQuant profile. It is a deterministic 4-bit weight codec intended to support
both a portable reference path and a fused INT8 DP4A path. UQFF 1.3 stores this codec with a stable
format tag. Command-line quantization and accelerator kernels remain separate integration stages.

## Scope

HQZ4 v1 supports rank-2 linear weights with shape `[output, input]`. It uses:

- a power-of-two group size along the input dimension;
- one randomized Hadamard transform per input group;
- signs derived from a 64-bit seed and the input-group index;
- one non-negative F32 scale per output row and input group in schema 2;
- signed scalar levels in `[-7, 7]`;
- two low-nibble-first codes per byte.

The sign pattern is shared by every output row. This is required because one transformed activation
must be reusable by all rows of a projection.

For normalized Hadamard matrix `H` and diagonal sign matrix `D`, define `R = H D`. The stored
quantized weights approximate `W R^T`; the runtime transforms the activation as `R x`. The product
therefore approximates `W x` without reconstructing the complete original-space weight matrix.

## UQFF tensors

The UQFF 1.3 representation is:

| Suffix | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `weight.format` | U8 | scalar | `7`, the HyperQuant serde discriminator |
| `weight.schema` | U32 | scalar | `2` for current HQZ4; readers also accept schema 1 |
| `weight.layout` | U8 | scalar | `0`, row-major signed nibbles |
| `weight.transform` | U8 | scalar | `0`, shared randomized Hadamard transform |
| `weight.bits` | U8 | scalar | `4` |
| `weight.group_size` | U32 | scalar | Input-group width |
| `weight.shape` | U32 | `[2]` | Logical `[output, input]` shape |
| `weight.seed_lo` | U32 | scalar | Low half of the RHT seed |
| `weight.seed_hi` | U32 | scalar | High half of the RHT seed |
| `weight` | U8 | `[output, input / 2]` | Packed signed codes |
| `weight.scales` | F32 | `[output, input / group_size]` | Group scales (`F16` in legacy schema 1) |
| `bias` | source dtype | optional | Linear bias |

The format stores algorithmic facts, not backend promises. In particular, it does not serialize a
`dp4a_ready` flag. A backend must validate the compute capability, layout, group alignment, and
supported query geometry itself.

An input-dimension shard preserves the global input-group index when deriving signs. The runtime
computes `global_group_offset = input_start / group_size`; it must not restart the sign sequence at
zero for every rank.

## Execution profiles

The same artifact has two implemented execution profiles:

- A16 reference: deterministic dequantization followed by dense CPU matrix multiplication with
  floating-point activations.
- A8/W4 DP4A on SM61+: transform and dynamically quantize activations to signed bytes, unpack W4
  to signed bytes, accumulate with DP4A into INT32, then apply activation and weight scales.

Single-token decode uses one warp per output. Multi-token prefill uses a 4x4 output tile: each warp
owns one weight row and reuses its packed W4 loads across four activation rows. Compatible Q/K/V,
gate/up, and GDN projections share one transformed A8 activation. Compatibility includes the
activation block shape, transform, seed, and global group offset; a mismatch falls back to separate
projection calls.

With the default group width of 128 and schema-2 F32 scales, the linear-layer payload is
`4 + 32 / 128 = 4.25` bits per weight, excluding per-layer metadata and optional bias tensors.

## Relation to GGUF quantization

UQFF and GGUF are containers; HQZ4 must be compared with a GGML tensor quantization type, not with
GGUF itself. By payload density, HQZ4 at its default group width is closest to `IQ4_XS` (4.25 bits
per weight). It is not byte-compatible or algorithmically equivalent: `IQ4_XS` is a nonlinear
codebook quantization, while HQZ4 is a signed linear W4 quantization after a shared randomized
Hadamard transform. `Q4_0` and `Q4_K` use 4.5 bits per weight and are useful conventional W4
baselines. `Q4_K_M` is a model-level mixed recipe rather than one tensor encoding.

## Excluded from v1

Rice or ANS entropy coding, variable-width bit stripping, vector lattices, learned rotations, and
mixed per-layer bit allocation are separate profiles. They must not change the HQZ4 byte contract.
Input-dimension sharding is valid only at group-aligned boundaries.
