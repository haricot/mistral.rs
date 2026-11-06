# FP4 standalone v2

Base: master@095991a7d680a6994e6f1554156ebef5d09db049.

This branch starts from the current upstream-compatible NVFP4/MXFP4 implementation,
not the old FP4 branch, which also incorporated Bolt/DFlash and legacy changes.
No application source, dependency or feature is changed in this baseline.

The CPU workflow checks the quantization crate and executes its existing NVFP4
unit and checkpoint tests. It does not fetch model fixtures or use a GPU.
A passing CPU workflow is not a physical CUDA or full-model qualification.

The subsequent legacy_v3 composition has four independent inputs: the updated
BF16, FP8 and FP4 standalone branches and the existing legacy branch. Legacy is
not a prerequisite or ancestor added to these standalone branches.
The old Pascal-specific FP4 adapter must be reconciled in that composition;
this baseline alone does not claim NVFP4 execution on SM61.
