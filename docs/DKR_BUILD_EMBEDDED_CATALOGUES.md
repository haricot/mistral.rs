# Build-embedded qualified DKR catalogues

The SM61 candidate can embed the exact qualified native Candle and typed DKR PTX
artifacts. The build scripts reuse the filesystem validators, then generate Rust
tables and include the verified bytes in the executable. Manifest parsing,
artifact hashing and entry inventory discovery happen at build time. CUDA context
creation, driver module loading and launch plan preparation still happen at runtime.

Set `DKR_EMBED_NATIVE_DIR` and `DKR_EMBED_TYPED_DIR` during compilation to select the
two input directories. Native embedding requires Candle's `cuda-dkr-s31` feature.
With embedded tables, the default runtime source is `embedded`.
`MISTRALRS_DKR_ARTIFACT_SOURCE=filesystem` selects the original validating loader
for comparison. Explicit `CANDLE_DKR_S31_DIR` and `DKR_KERNEL_PACKAGES` paths must
match their embedded build inputs when using the embedded source. With these
variables unset, the compiled input identifiers are used without accessing disk.

## Sealed SM61 startup comparison

From the candidate checkout, prepare one executable using the existing qualified
archive and CUDA build cache:

```bash
python3 tools/dkr_r123_s1a_shared_catalog.py --embedded --build \
  --original "$DKR_S6_ORIGINAL_REPO" --s0-report "$DKR_S0_REPORT"
```

Then run the comparison with the same arguments, omitting `--build`. It measures
fresh processes in static, filesystem, embedded, embedded, filesystem, static
order, followed by one telemetry process per mode. Filesystem and embedded modes
use the same candidate executable and a verified copy of all 212 typed manifests
and 25 native families. During embedded executions those copies are temporarily
renamed and restored, including after subprocess failures. Original archives stay
at their original paths.

The report is `embedded-catalogue-comparison.json`. Completion requires exact
numerical parity against the sealed reference, identical external routes and
submission counts between both candidate modes, and the expected catalogue source
telemetry. Timing results report drift and do not automatically approve a
performance promotion.
