# UQFF draft standalone (experimental)

## Provenance
Created from `master@28ed6daca455b4ae063896014d1ea698b912c539`, with the `nemotron_bolt_dflash_standalone@1319e272149f41e7ce1648b2b52514877403fb44` prerequisite applied. The original `quantize -m <DFlashDraftModel>` command still uses the ordinary model loader and does **not** export this architecture.

## Implemented
- DFlash2-specific preflight plan, preserving target tap indices, hidden geometry, layers, checkpoint NVFP4 flag and `has_own_lm_head`.
- Q8_0-only plan validation.
- Non-destructive CLI command to display that plan in JSON:
```bash
cargo run -p mistralrs-cli -- uqff draft-plan -m /path/to/local-dflash-checkpoint
```
- Fixture-based tests cover Bolt's model geometry, own-head and invalid architecture.

## Not yet implemented
- A true standalone Q8_0 DFlash UQFF writer, streaming quantized tensors without instantiating the full dense draft on GPU.
- Preservation of UQFF tensor metadata for NVFP4 input tensors and untouched norms/heads.
- DFlash-specific UQFF reloader, integrity hashes and equivalence of generated proposals.
- Atomic write/rollback, shard-sized memory cap and full memory accounting.

This branch must **not** claim that a file was generated or that `quantize --isq Q8_0` now accepts DFlash. The command here is intentionally only `draft-plan`.

## Local qualification (operator-run)
```bash
cargo fmt --all -- --check
cargo test -p mistralrs-core bolt_draft_preflight_preserves_taps_and_head_contract
cargo test -p mistralrs-core non_dflash_model_is_not_a_draft
```

The inherited `mistral-integration/sources.json` is provenance of the Bolt prerequisite, not a valid promotion declaration for UQFF draft.
