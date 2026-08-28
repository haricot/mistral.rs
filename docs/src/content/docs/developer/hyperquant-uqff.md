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
- one non-negative F16 scale per output row and input group;
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
| `weight.schema` | U32 | scalar | `1` for HQZ4 v1 |
| `weight.layout` | U8 | scalar | `0`, row-major signed nibbles |
| `weight.transform` | U8 | scalar | `0`, shared randomized Hadamard transform |
| `weight.bits` | U8 | scalar | `4` |
| `weight.group_size` | U32 | scalar | Input-group width |
| `weight.shape` | U32 | `[2]` | Logical `[output, input]` shape |
| `weight.seed_lo` | U32 | scalar | Low half of the RHT seed |
| `weight.seed_hi` | U32 | scalar | High half of the RHT seed |
| `weight` | U8 | `[output, input / 2]` | Packed signed codes |
| `weight.scales` | F16 | `[output, input / group_size]` | Group scales |
| `bias` | source dtype | optional | Linear bias |

The format stores algorithmic facts, not backend promises. In particular, it does not serialize a
`dp4a_ready` flag. A backend must validate the compute capability, layout, group alignment, and
supported query geometry itself.

An input-dimension shard preserves the global input-group index when deriving signs. The runtime
computes `global_group_offset = input_start / group_size`; it must not restart the sign sequence at
zero for every rank.

## Execution profiles

The same artifact has two execution profiles. The first is implemented; the second is planned:

- A16 reference: deterministic dequantization followed by dense CPU matrix multiplication with
  floating-point activations.
- A8 DP4A (planned): transform and dynamically quantize activations to INT8, unpack W4 to INT8,
  accumulate
  with DP4A into INT32, then apply scales and produce the requested floating-point output.

The SM61 backend must support single-token decode and multi-token matrices with query lengths up to
9 so LFM2.5 DSpark verification does not fall back to a full dequantization path.

## Excluded from v1

Rice or ANS entropy coding, variable-width bit stripping, vector lattices, learned rotations, and
mixed per-layer bit allocation are separate profiles. They must not change the HQZ4 byte contract.
Input-dimension sharding is valid only at group-aligned boundaries.
