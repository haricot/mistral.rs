#!/usr/bin/env bash
# Restore an immutable rebased CUDA candidate from the prepare-job artifact.
# No GitHub workflow write permission is required for CPU/GPU proof jobs.
set -euo pipefail
: "${CANDIDATE:?candidate commit required}"
: "${MASTER:?upstream commit required}"
: "${LOCK:?lock hash required}"
: "${BUNDLE_SHA:?bundle SHA256 required}"
: "${CANDIDATE_BUNDLE:?path to candidate.bundle required}"

[[ "$CANDIDATE" =~ ^[0-9a-f]{40}$ ]]
[[ "$MASTER" =~ ^[0-9a-f]{40}$ ]]
[[ "$LOCK" =~ ^[0-9a-f]{64}$ ]]
[[ "$BUNDLE_SHA" =~ ^[0-9a-f]{64}$ ]]
test -s "$CANDIDATE_BUNDLE"
test "$(sha256sum "$CANDIDATE_BUNDLE" | cut -d' ' -f1)" = "$BUNDLE_SHA"
git remote add upstream https://github.com/EricLBuehler/mistral.rs.git
git fetch --no-tags upstream +refs/heads/master:refs/remotes/upstream/master
test "$(git rev-parse refs/remotes/upstream/master)" = "$MASTER" || {
  echo "::error::Upstream master moved since candidate staging; stop and rebase afresh"
  exit 3
}
git bundle verify "$CANDIDATE_BUNDLE"
git fetch --no-tags "$CANDIDATE_BUNDLE" HEAD
git switch --detach "$CANDIDATE"
test "$(git rev-parse HEAD)" = "$CANDIDATE"
git merge-base --is-ancestor "$MASTER" HEAD
test "$(sha256sum Cargo.lock | cut -d' ' -f1)" = "$LOCK"
echo "::notice::Verified candidate $CANDIDATE from bundle $BUNDLE_SHA"
