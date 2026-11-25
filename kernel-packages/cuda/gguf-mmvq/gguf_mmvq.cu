// Standalone typed GGUF MMVQ device images. The existing quantized dot
// products, reduction and prefetch bodies remain the single source of truth.
// Attribution: mistralrs-quant/third_party/ggml_mmvq/README.md (see vecdot.cuh).
#if !defined(DKR_GGUF_MMVQ_IMAGE) || DKR_GGUF_MMVQ_IMAGE < 1 || DKR_GGUF_MMVQ_IMAGE > 11
#error "select one bounded GGUF MMVQ image (1..11)"
#endif
#include "../../../mistralrs-quant/kernels/mmvq_gguf/mmvq_gguf.cu"
