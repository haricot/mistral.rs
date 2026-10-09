#!/usr/bin/env bash
# Rebase the exact PR head onto the official upstream master, then stage only
# a reproducible Candle cuda_legacy candidate. Never mutate allow_old_card.
set -euo pipefail
report="$RUNNER_TEMP/legacy-rebase"
mkdir -p "$report"
git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
git remote add upstream https://github.com/EricLBuehler/mistral.rs.git
git fetch --no-tags origin +refs/heads/allow_old_card:refs/remotes/origin/allow_old_card
git fetch --no-tags upstream +refs/heads/master:refs/remotes/upstream/master
old="$(git rev-parse refs/remotes/origin/allow_old_card)"
master="$(git rev-parse refs/remotes/upstream/master)"
candle="$(git ls-remote https://github.com/haricot/candle.git refs/heads/cuda_legacy | cut -f1)"
for sha in "$old" "$master" "$candle"; do
  [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo "::error::Source SHA missing"; exit 2; }
done
for v in "$EXPECTED_PR" "$EXPECTED_UPSTREAM" "$EXPECTED_CANDLE"; do
  [[ -z "$v" || "$v" =~ ^[0-9a-f]{40}$ ]] || {
    echo "::error::Expected SHA must be a full SHA"; exit 2;
  }
done
test -z "$EXPECTED_PR" || test "$EXPECTED_PR" = "$old"
test -z "$EXPECTED_UPSTREAM" || test "$EXPECTED_UPSTREAM" = "$master"
test -z "$EXPECTED_CANDLE" || test "$EXPECTED_CANDLE" = "$candle"
test "$(git ls-remote upstream refs/heads/master | cut -f1)" = "$master"
test "$(git ls-remote origin refs/heads/allow_old_card | cut -f1)" = "$old"
git clone --quiet --no-tags --branch cuda_legacy --single-branch \
  https://github.com/haricot/candle.git "$RUNNER_TEMP/candle-cuda-legacy"
test "$(git -C "$RUNNER_TEMP/candle-cuda-legacy" rev-parse HEAD)" = "$candle"
jq -e '
  .schema_version==1 and .kind=="legacy-cuda-five" and
  .integration_target=="legacy-cuda" and
  ([.features[].feature] | sort)==
    (["bf16_candle","fp8_candle","fp4_candle",
      "cudnn_fallback_candle","moe_simt_f16_candle"] | sort)
' "$RUNNER_TEMP/candle-cuda-legacy/candle-integration/legacy-cuda.json" > /dev/null
jq -n --arg old "$old" --arg master "$master" --arg candle "$candle" \
  '{schema_version:1,status:"PENDING",original_pr_sha:$old,
    upstream_repository:"EricLBuehler/mistral.rs",
    upstream_master_sha:$master,candle_repository:"haricot/candle",
    candle_branch:"cuda_legacy",candle_sha:$candle,
    physical_gpu_validated:false}' > "$report/provenance.json"
base="$(git merge-base "$old" "$master")"
echo "base=$base old_pr=$old upstream=$master" > "$report/rebase.log"
cp .github/scripts/legacy-cuda-port-moe.py "$RUNNER_TEMP/legacy-cuda-port-moe.py"
cp .github/scripts/legacy-cuda-port-quant.py "$RUNNER_TEMP/legacy-cuda-port-quant.py"
cp .github/scripts/legacy-cuda-port-paged.py "$RUNNER_TEMP/legacy-cuda-port-paged.py"
cp .github/scripts/legacy-cuda-port-fp8.py "$RUNNER_TEMP/legacy-cuda-port-fp8.py"
git switch --detach "$old"
git switch -c "rebased-allow-old-card-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
# Preserve the upstream build script and semantically replay ONLY the
# independently reviewed legacy BF16 flag from historical commit 37007f62.
# Never select "ours" wholesale for arbitrary conflicts: that can silently
# drop the PR's CUDA kernels.
resolve_first_bf16_conflict() {
  local current conflicted path="mistralrs-core/build.rs"
  current="$(git rev-parse REBASE_HEAD 2>/dev/null)" || return 1
  conflicted="$(git diff --name-only --diff-filter=U)"
  [[ "$current" == 37007f62b22dce109ddad5920dc128e46cbb0a47 ]] || return 1
  [[ "$conflicted" == "$path" ]] || return 1
  git show ":2:$path" > "$path" || return 1
  python3 - "$path" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
rerun = '        println!("cargo:rerun-if-changed=build.rs");\n'
if text.count(rerun) != 1:
    raise SystemExit("Unexpected current upstream build.rs rerun hook")
text = text.replace(
    rerun,
    rerun + '        println!("cargo:rerun-if-env-changed=ALLOW_LEGACY");\n',
    1,
)
pin = '        let compute_cap = builder.get_compute_cap().unwrap_or(80);\n'
if text.count(pin) != 1:
    raise SystemExit("Unexpected modern CUDA capability handling")
legacy = '''
        let allow_legacy = std::env::var("ALLOW_LEGACY").unwrap_or_default();
        let allow_legacy_bf16 = allow_legacy == "all"
            || allow_legacy
                .split(',')
                .map(str::trim)
                .any(|value| value == "bf16");
'''
text = text.replace(pin, pin + legacy + '\n', 1)
old = '''        if compute_cap < 80 {
            builder = builder.arg("-DNO_BF16_KERNEL");
        }
'''
new = '''        if compute_cap < 80 && !allow_legacy_bf16 {
            builder = builder.arg("-DNO_BF16_KERNEL");
        }
        if allow_legacy_bf16 {
            builder = builder.arg("-DALLOW_LEGACY_BF16");
        }
'''
if text.count(old) != 1:
    raise SystemExit("Unexpected SM80/BF16 kernel gate; not safe to auto-resolve")
text = text.replace(old, new, 1)
path.write_text(text)
PY
  git diff --check && git add "$path"
  echo "Resolved exact commit $current / $path using current upstream implementation" \
    >> "$report/rebase.log"
}

# This pin-only historical commit targets Candle 0.10.2 and becomes obsolete
# when this workflow pins all four modern 0.11.0 workspace dependencies to the
# validated cuda_legacy SHA. We can skip it ONLY when it conflicts, and only
# after verifying its exact commit SHA and the two dependency-only files.
skip_obsolete_0102_pin_conflict() {
  local current conflicted changed
  current="$(git rev-parse REBASE_HEAD 2>/dev/null)" || return 1
  [[ "$current" == a56ae7eeeed6469265c5b1659f2c4251f3074302 ]] || return 1
  changed="$(git diff-tree --no-commit-id --name-only -r "$current" | sort)"
  [[ "$changed" == "$(printf 'Cargo.lock\nCargo.toml')" ]] || return 1
  conflicted="$(git diff --name-only --diff-filter=U)"
  [[ -n "$conflicted" ]] || return 1
  while IFS= read -r item; do
    [[ "$item" == "Cargo.toml" || "$item" == "Cargo.lock" ]] || return 1
  done <<< "$conflicted"
  echo "Skipping historical 0.10.2-only Candle pin $current; replaced with 0.11.0 cuda_legacy after rebase" \
    >> "$report/rebase.log"
}

rebased=false
if git rebase --onto "$master" "$base" >> "$report/rebase.log" 2>&1; then
  rebased=true
else
  # Each resolver verifies the precise commit, conflicted files, and upstream
  # code shape. Stop immediately on any new, unreviewed conflict.
  for ((attempt=0; attempt<6; attempt++)); do
    if resolve_first_bf16_conflict; then
      echo "::notice::Ported opt-in BF16 gate onto current upstream build.rs"
      if GIT_EDITOR=true git rebase --continue >> "$report/rebase.log" 2>&1; then
        rebased=true
        break
      fi
    elif [[ "$(git rev-parse REBASE_HEAD 2>/dev/null || true)" == 251963b32a10195fad344a182fb37e935b14ca1c ]] &&
         python3 "$RUNNER_TEMP/legacy-cuda-port-moe.py" >> "$report/rebase.log" 2>&1; then
      echo "::notice::Semantically ported historical CUDA MoE via maintained ABI and Pascal SIMT"
      if GIT_EDITOR=true git rebase --continue >> "$report/rebase.log" 2>&1; then
        rebased=true
        break
      fi
    elif [[ "$(git rev-parse REBASE_HEAD 2>/dev/null || true)" == 30f21d8f79f2209f8a6fa74edb57a350e9ad017c ]] &&
         python3 "$RUNNER_TEMP/legacy-cuda-port-quant.py" >> "$report/rebase.log" 2>&1; then
      echo "::notice::Ported quant BF16 dispatch; AFQ modern source preserved"
      if GIT_EDITOR=true git rebase --continue >> "$report/rebase.log" 2>&1; then
        rebased=true
        break
      fi
    elif [[ "$(git rev-parse REBASE_HEAD 2>/dev/null || true)" == adc54c5bdbd69b0e738241b78357919af9b0c6e0 ]] &&
         python3 "$RUNNER_TEMP/legacy-cuda-port-paged.py" >> "$report/rebase.log" 2>&1; then
      echo "::notice::Ported paged BF16 helpers and FP8 scale guards onto modern sources"
      if GIT_EDITOR=true git rebase --continue >> "$report/rebase.log" 2>&1; then
        rebased=true
        break
      fi
    elif skip_obsolete_0102_pin_conflict; then
      echo "::notice::Replacing obsolete Candle 0.10.2-only pin with current cuda_legacy"
      if git rebase --skip >> "$report/rebase.log" 2>&1; then
        rebased=true
        break
      fi
    elif [[ "$(git rev-parse REBASE_HEAD 2>/dev/null || true)" == bd7e8d43bc680f8e3a5d4d291a076dffe1e8d3b3 ]] &&
         python3 "$RUNNER_TEMP/legacy-cuda-port-fp8.py" >> "$report/rebase.log" 2>&1; then
      echo "::notice::Ported legacy FP8 to modern quant and paged-attention builds"
      if GIT_EDITOR=true git rebase --continue >> "$report/rebase.log" 2>&1; then
        rebased=true
        break
      fi
    else
      break
    fi
  done
fi
if [[ "$rebased" != true ]]; then
  git diff --name-only --diff-filter=U > "$report/conflicts.txt" || true
  git ls-files -u > "$report/unmerged-index.txt" || true
  git status --short > "$report/status.txt" || true
  git rev-parse REBASE_HEAD > "$report/failed-commit.sha" 2>/dev/null || true
  echo "::group::Unresolved CUDA PR rebase conflict"
  cat "$report/failed-commit.sha" "$report/conflicts.txt" 2>/dev/null || true
  tail -n 55 "$report/rebase.log"
  echo "::endgroup::"
  git rebase --abort || true
  jq '.status="REBASE_CONFLICT"' "$report/provenance.json" > "$report/temp.json"
  mv "$report/temp.json" "$report/provenance.json"
  echo "::error::Unreviewed conflict; PR branch unchanged. Inspect the archived conflict proof."
  exit 1
fi
git merge-base --is-ancestor "$master" HEAD || exit 3
# The original PR was written before the current Cargo feature graph existed.
# Forward-wire explicit Candle 0.11 BF16/FP8/FP4 features rather than compiling
# a no-op "legacy" build. Leave all unrelated features untouched.
python3 - <<'PY'
from pathlib import Path
wiring = {
    "mistralrs-cli": {
        "cuda-legacy-bf16": ["cuda", "mistralrs-core/cuda-legacy-bf16", "mistralrs-server-core/cuda-legacy-bf16"],
        "cuda-legacy-fp8": ["cuda", "mistralrs-core/cuda-legacy-fp8", "mistralrs-server-core/cuda-legacy-fp8"],
        "cuda-legacy-fp4": ["cuda", "mistralrs-core/cuda-legacy-fp4", "mistralrs-server-core/cuda-legacy-fp4"],
    },
    "mistralrs-server-core": {
        "cuda-legacy-bf16": ["cuda", "mistralrs-core/cuda-legacy-bf16"],
        "cuda-legacy-fp8": ["cuda", "mistralrs-core/cuda-legacy-fp8"],
        "cuda-legacy-fp4": ["cuda", "mistralrs-core/cuda-legacy-fp4"],
    },
    "mistralrs-core": {
        "cuda-legacy-bf16": ["cuda", "candle-core/cuda-legacy-bf16", "mistralrs-quant/cuda-legacy-bf16", "mistralrs-paged-attn/cuda-legacy-bf16"],
        "cuda-legacy-fp8": ["cuda", "candle-core/cuda-legacy-fp8", "mistralrs-quant/cuda-legacy-fp8", "mistralrs-paged-attn/cuda-legacy-fp8"],
        "cuda-legacy-fp4": ["cuda", "candle-core/cuda-legacy-fp4", "mistralrs-quant/cuda-legacy-fp4", "mistralrs-paged-attn/cuda-legacy-fp4"],
    },
    "mistralrs-quant": {
        "cuda-legacy-bf16": ["cuda", "candle-core/cuda-legacy-bf16"],
        "cuda-legacy-fp8": ["cuda", "candle-core/cuda-legacy-fp8"],
        "cuda-legacy-fp4": ["cuda", "candle-core/cuda-legacy-fp4"],
    },
    "mistralrs-paged-attn": {
        "cuda-legacy-bf16": ["cuda", "candle-core/cuda-legacy-bf16"],
        "cuda-legacy-fp8": ["cuda", "candle-core/cuda-legacy-fp8"],
        "cuda-legacy-fp4": ["cuda", "candle-core/cuda-legacy-fp4"],
    },
}
for crate, flags in wiring.items():
    path = Path(crate) / "Cargo.toml"
    text = path.read_text()
    anchor = "[features]\n"
    if text.count(anchor) != 1:
        raise SystemExit(f"{crate}: expected exactly one [features] table")
    active = text.split("[features]\n", 1)[1].split("\n[", 1)[0]
    lines = []
    for feature, dependencies in flags.items():
        existing = [line for line in active.splitlines() if line.startswith(feature + " =")]
        if existing:
            # Already-ported sources must agree with the expected feature graph.
            if len(existing) != 1 or not all(f'"{d}"' in existing[0] for d in dependencies):
                raise SystemExit(f"{crate}: incompatible existing feature {feature}")
            continue
        rendered = ", ".join(f'"{d}"' for d in dependencies)
        lines.append(f"{feature} = [{rendered}]")
    if lines:
        text = text.replace(anchor, anchor + "\n".join(lines) + "\n", 1)
        path.write_text(text)
PY
CANDLE_SHA="$candle" python3 - <<'PY'
import os
import re
from pathlib import Path
candle = os.environ["CANDLE_SHA"]
manifest = Path("Cargo.toml")
text = manifest.read_text()
for name in ("candle-core", "candle-nn", "candle-flash-attn-v3", "candle-metal-kernels"):
    pat = re.compile(rf"(?m)^{re.escape(name)}[ \t]*=[ \t]*\{{[^\n]*\}}[ \t]*$")
    value = f'{name} = {{ git = "https://github.com/haricot/candle.git", version = "0.11.0", rev = "{candle}" }}'
    text, hits = pat.subn(value, text)
    if hits != 1:
        raise SystemExit(f"{name}: expected one modern 0.11 dependency, found {hits}")
# The 2026-01 PR patch pointed at Candle 0.10.2, which must not leak into
# the 0.11.0 dependency graph. Remove only the known exact historical pin.
import re as _re
legacy_section = _re.compile(
    r"(?ms)^\[patch\.crates-io\]\n"
    r"((?:candle-(?:core|nn|flash-attn-v3|flash-attn|metal-kernels|kernels) = "
    r"\{ git = \"https://github.com/haricot/candle\.git\", rev = \"e1f418a\" \}\n){6})"
)
if "[patch.crates-io]" in text:
    # We can only discard exactly the original pin-only patch block; keep
    # any unrelated overrides or customizations for manual review.
    text, hits = legacy_section.subn("", text)
    if hits != 1:
        raise SystemExit("Unexpected legacy [patch.crates-io] block; refusing implicit removal")
    text = text.replace("\n\n\n[profile.release-with-debug]", "\n\n[profile.release-with-debug]")
manifest.write_text(text)
PY
mkdir -p mistralrs-core/tests
cat > mistralrs-core/tests/legacy_sm61_runtime.rs <<'RS'
#![cfg(feature = "cuda")]
use candle_core::{DType, Device, Result, Tensor};

#[test]
fn legacy_sm61_f32_bf16_fp8_roundtrip() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let input = Tensor::new(&[1f32, 2., 3., 4.], &device)?;
    let doubled = (&input + &input)?;
    assert_eq!(
        doubled.to_device(&Device::Cpu)?.to_vec1::<f32>()?,
        vec![2., 4., 6., 8.]
    );
    for dtype in [DType::BF16, DType::F8E4M3] {
        let tensor = input.to_dtype(dtype)?;
        let restored = tensor.to_dtype(DType::F32)?;
        device.synchronize()?;
        let values = restored.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        for (value, expected) in values.iter().zip([1., 2., 3., 4.]) {
            assert!((value - expected).abs() < 0.1, "{dtype:?}: {value} != {expected}");
        }
    }
    Ok(())
}
RS
git diff --check
rustup toolchain install stable --profile minimal --component rustfmt
rustfmt --edition 2021 mistralrs-core/tests/legacy_sm61_runtime.rs
rustfmt --edition 2021 mistralrs-core/src/cuda/moe.rs
rustfmt --edition 2021 mistralrs-quant/src/unquantized/mod.rs
rustfmt --edition 2021 mistralrs-paged-attn/src/cuda/backend/paged_attention.rs
rustfmt --edition 2021 mistralrs-quant/build.rs
rustfmt --edition 2021 mistralrs-paged-attn/build.rs
rustfmt --edition 2021 mistralrs-core/build.rs
cargo +stable update --workspace > "$report/cargo-update.log" 2>&1 || {
  tail -n 100 "$report/cargo-update.log"; exit 1;
}
cargo +stable metadata --locked --format-version 1 > "$report/metadata.json"
grep -Fq "$candle" Cargo.lock
lock="$(sha256sum Cargo.lock | cut -d' ' -f1)"
git add Cargo.toml Cargo.lock \
  mistralrs-cli/Cargo.toml mistralrs-core/Cargo.toml \
  mistralrs-server-core/Cargo.toml mistralrs-quant/Cargo.toml \
  mistralrs-paged-attn/Cargo.toml \
  mistralrs-core/src/cuda/moe.rs \
  mistralrs-core/build.rs \
  mistralrs-quant/build.rs \
  mistralrs-quant/src/unquantized/mod.rs \
  mistralrs-paged-attn/build.rs \
  mistralrs-paged-attn/src/cuda/backend/paged_attention.rs \
  mistralrs-core/tests/legacy_sm61_runtime.rs
git commit -m "build(cuda): pin Candle cuda_legacy and add Pascal runtime check"
candidate="$(git rev-parse HEAD)"
# GitHub Actions' default GitHub App token cannot push a branch introducing
# upstream workflow files without Workflows:write. Transport the exact Git
# objects as an immutable Git bundle instead. No remote branch is mutated
# until final promotion, after CPU/CUDA/physical SM61 proof.
git bundle create "$report/candidate.bundle" HEAD "^$master"
git bundle verify "$report/candidate.bundle"
bundle_sha="$(sha256sum "$report/candidate.bundle" | cut -d' ' -f1)"
ref="artifact://mistral-legacy-prep-$GITHUB_RUN_ID/candidate.bundle"
test "$(git ls-remote origin refs/heads/allow_old_card | cut -f1)" = "$old"
test "$(git ls-remote upstream refs/heads/master | cut -f1)" = "$master"
jq --arg candidate "$candidate" --arg lock "$lock" --arg ref "$ref" \
   --arg bundle "$bundle_sha" \
  '.status="CANDIDATE_STAGED" | .candidate_sha=$candidate |
   .lock_sha256=$lock | .candidate_ref=$ref |
   .bundle_sha256=$bundle' \
  "$report/provenance.json" > "$report/temp.json"
mv "$report/temp.json" "$report/provenance.json"
cp Cargo.lock "$report/Cargo.lock"
echo "candidate=$candidate" >> "$GITHUB_OUTPUT"
echo "candidate_ref=$ref" >> "$GITHUB_OUTPUT"
echo "bundle_sha256=$bundle_sha" >> "$GITHUB_OUTPUT"
echo "pr_sha=$old" >> "$GITHUB_OUTPUT"
echo "upstream_sha=$master" >> "$GITHUB_OUTPUT"
echo "candle_sha=$candle" >> "$GITHUB_OUTPUT"
echo "lock_sha256=$lock" >> "$GITHUB_OUTPUT"
echo "Candidate $candidate transported in Git bundle $bundle_sha; PR unchanged" >> "$GITHUB_STEP_SUMMARY"
