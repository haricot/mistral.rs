# Master and FP4 campaign preparation

The selected base is `095991a7d680a6994e6f1554156ebef5d09db049`.
The snapshot rejects a moved master. The four physically validated Candle
dependencies retain their immutable fork commits and actual version 0.11.0.
The master's zip 9.1.3 is pinned to its official v9.1.3 commit in a separate
Git source, so Candle's registry zip 9.0.3 can coexist without removing the new
skill archive header-consistency check.

The preparation pins `master` and both FP4 source refs in the snapshot. Legacy merges the generic packed NVFP4 dispatch from `fix/allow_old_card-nvfp4-legacy`; its source blob must match `fp4_mistralrs_standalone`. The Bolt/DFlash component also merges the complete standalone FP4 source. All four existing standalone components remain independent except for their declared Legacy prerequisite. Extended composes those exact candidates.

FP4 has no independent physical qualification prerequisite. Each final campaign runs an analytic W4A16 reference on CPU and the packed Pascal E7 implementation with F16/BF16 activations on SM61. The embedding test checks packed MXFP4 gather against Candle's dequantized reference. These are kernel and format tests; they do not qualify a complete FP4 model. The legacy dispatch rejects W4A4 and rank-three expert weights explicitly.

The merge resolutions retain upstream FP8/portable NVFP4 implementations and the opt-in Pascal fallbacks. The API migration preserves Gemma 4 text-only GGUF loading through the new architecture registry, the CPU per-layer embedding path, and DSpark embedding calls. LFM3 D1 is included from master and type-checked; no D1 model execution is claimed.

The CUDA SDK identity inputs are unchanged. Source changes rebuild the affected candidates, while the digest-pinned compiler image is reused. Prepared-ref identities include conflict resolutions and source compatibility migrations.

A push to either `cuda-reuse-legacy.once` or `cuda-reuse-extended.once` on `ci/extended-release-r1-single-build` starts a complete qualification campaign. The Extended campaign also compiles the final optimized executable and assembles its runtime image. This review branch cannot promote source branches or publish the image. A source or preparation correction requires a new marker push, never a GPU-only retry with artifacts from an older attempt.

GPU archives are extracted and removed one component at a time; compressed archives, JUnit reports and hashes remain available. ASD isolation, immutable candidates, SHA256 verification, full LFM3 evidence and numerical acceptance requirements are unchanged. VL/GGUF thresholds are not relaxed by this preparation.

The scalar FP8 capability is separate from the SM80 MMA capability. `grouped_fp8.cu` is excluded for targets below SM80; its Rust module and dispatch sites use `has_fp8_mma_kernels`. The per-tensor CPU fallback and scalar/blockwise FP8 kernels remain enabled under their existing capabilities.

The FP8 roundtrip test explicitly uses CPU in the hosted CPU profile. A separate `cuda_sm61_roundtrip_f8e4m3` test runs the same body on the physical GPU. Enabling the CUDA feature is not interpreted as proof of a device on a hosted runner; the CPU CUDA-link stub remains an error-returning loader aid.

FP8 KV admission accepts SM61 only when the paged-attention build actually
compiled its explicit legacy FP8 path for SM61. Ordinary builds still require
SM80+, and missing kernels, CPU, MLA and unsupported legacy architectures retain
their existing rejection. CPU tests exercise the capability matrix; the physical
profile also checks the real device gate.

The software SM61 FP8 cache selects the standard paged-attention backend before
cache sizing and allocation in both text and multimodal pipelines. The same
metadata then controls cache layout, execution plans and decode. FlashInfer HND
is retained for FP16; native FP8 builds are unchanged. A physical regression
uses the production CacheEngine, the hybrid layer mask and GQA decode with an
FP8 cache, in addition to the synthetic benchmark. The real-model Text benchmark
must also complete with both FP16 and FP8; a passing synthetic cache test cannot
substitute for it.

Every physical component writes and reads an FP8 paged cache, using nontrivial
slot mappings and unequal K/V scales. Exactly representable fixtures check all
stored values and compare decode against a CPU softmax reference and FP16.
The 0.01 attention tolerance is the existing upstream paged-attention fixture
tolerance; it does not change any full-model numeric policy. The benchmark uses
128, 2048 and 8192 tokens, four warmups and nine alternating FP16/FP8 samples of
eight calls each, with synchronization around host timing. It records raw write
and decode times, medians and allocated cache bytes. API/launch overhead is
included. No speedup threshold is required. These are synthetic cache tests,
separate from the real-model measurements in the Extended release job.
