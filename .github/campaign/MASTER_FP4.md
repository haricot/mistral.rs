# Master and FP4 campaign preparation

The preparation pins `master` and both FP4 source refs in the snapshot. Legacy merges the generic packed NVFP4 dispatch from `fix/allow_old_card-nvfp4-legacy`; its source blob must match `fp4_mistralrs_standalone`. The Bolt/DFlash component also merges the complete standalone FP4 source. All four existing standalone components remain independent except for their declared Legacy prerequisite. Extended composes those exact candidates.

FP4 has no independent physical qualification prerequisite. Each final campaign runs an analytic W4A16 reference on CPU and the packed Pascal E7 implementation with F16/BF16 activations on SM61. The embedding test checks packed MXFP4 gather against Candle's dequantized reference. These are kernel and format tests; they do not qualify a complete FP4 model. The legacy dispatch rejects W4A4 and rank-three expert weights explicitly.

The merge resolutions retain upstream FP8/portable NVFP4 implementations and the opt-in Pascal fallbacks. The API migration preserves Gemma 4 text-only GGUF loading through the new architecture registry, the CPU per-layer embedding path, and DSpark embedding calls. LFM3 D1 is included from master and type-checked; no D1 model execution is claimed.

The CUDA SDK identity inputs are unchanged. Source changes rebuild the affected candidates, while the digest-pinned compiler image is reused. Prepared-ref identities include conflict resolutions and source compatibility migrations.

A push to either `cuda-reuse-legacy.once` or `cuda-reuse-extended.once` on `ci/extended-release-r1-single-build` starts a complete qualification campaign. The Extended campaign also compiles the final optimized executable and assembles its runtime image. This review branch cannot promote source branches or publish the image. A source or preparation correction requires a new marker push, never a GPU-only retry with artifacts from an older attempt.

GPU archives are extracted and removed one component at a time; compressed archives, JUnit reports and hashes remain available. ASD isolation, immutable candidates, SHA256 verification, full LFM3 evidence and numerical acceptance requirements are unchanged. VL/GGUF thresholds are not relaxed by this preparation.
