#!/usr/bin/env python3
"""Fail-closed 3-way transplant of DSpark-only edits onto upstream LFM3.

Used ONLY for the two manually audited 2026-10-28 conflicts. It applies
feature changes from git index stage 2 relative to stage 1 onto stage 3.
Never overwrites upstream chunks or expands the accepted path set.
"""
from __future__ import annotations
import difflib
import subprocess
from pathlib import Path

PROFILES = {
    "lfm25_dspark": frozenset((
        "mistralrs-cli/src/args/mod.rs",
        "mistralrs-core/src/paged_attention/layers/paged_attention.rs",
    )),
    "nemotron_bolt_dflash": frozenset((
        "mistralrs-core/src/gguf/mod.rs",
    )),
}
ALLOWED = frozenset().union(*PROFILES.values())

def command(repo, *args):
    p = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"DSpark functional repair: git {args} failed: {p.stderr}")
    return p.stdout

def read_index(repo, stage, path):
    if path not in ALLOWED:
        raise RuntimeError(f"DSpark functional repair: unapproved source {path}")
    return command(repo, "show", f":{stage}:{path}").splitlines(keepends=True)

def contiguous_positions(lines, fragment):
    if not fragment:
        return []
    first = fragment[0]
    length = len(fragment)
    return [i for i, line in enumerate(lines)
            if line == first and lines[i:i+length] == fragment]

def context_scores(lines, base, i1, i2, position, old_length, limit=18):
    left = 0
    for offset in range(1, min(limit, i1, position) + 1):
        if lines[position-offset] != base[i1-offset]:
            break
        left += 1
    right = 0
    for offset in range(min(limit, len(base)-i2, len(lines)-position-old_length)):
        if lines[position+old_length+offset] != base[i2+offset]:
            break
        right += 1
    return left, right

def find_location(lines, base, i1, i2, removed, path):
    if removed:
        positions = contiguous_positions(lines, removed)
    else:
        # Insertions are allowed only by matching existing source context
        # on either side. Missing/ambiguous anchors fail rather than guess.
        positions = list(range(len(lines)+1))
    ranked = []
    for position in positions:
        left, right = context_scores(lines, base, i1, i2, position, len(removed))
        if left + right >= (2 if removed else 4):
            ranked.append((left+right, min(left, right), position, left, right))
    if not ranked:
        raise RuntimeError(f"DSpark functional repair: changed or missing anchor {path}@{i1+1}")
    best = max((a,b) for a,b,*_ in ranked)
    finalists = [row for row in ranked if row[:2] == best]
    if len(finalists) != 1:
        raise RuntimeError(f"DSpark functional repair: ambiguous anchor {path}@{i1+1}")
    return finalists[0][2]

def transplant(base, feature, upstream, path):
    if path not in ALLOWED:
        raise RuntimeError(f"DSpark functional repair: path not whitelisted: {path}")
    result = upstream[:]
    opcodes = difflib.SequenceMatcher(None, base, feature, autojunk=False).get_opcodes()
    audit=[]
    # Descending source offsets prevent an early insertion shifting later
    # locations while the incoming master is being reconciled.
    for tag, i1, i2, j1, j2 in reversed(opcodes):
        if tag == "equal":
            continue
        removed, added = base[i1:i2], feature[j1:j2]
        if (path == "mistralrs-cli/src/args/mod.rs"
                and removed == ["    /// MTP assistant model id or path.\n"]
                and added == ["    /// MTP, DFlash, or DSpark assistant model id or path.\n"]
                and upstream.count("    /// Separate MTP draft model, as a Hugging Face ID or local path.\n") == 2):
            audit.append((i1+1, "upstream-comment-retained"))
            continue
        if tag == "delete" or len(removed) > 3:
            raise RuntimeError(f"DSpark functional repair: destructive delta {path}@{i1+1}")
        # Do not deduplicate by searching the *whole file*: MTP config
        # contains identical changes in distinct Run/Serve/Bench blocks.
        # First resolve the precise old-context location.
        try:
            location = find_location(result, base, i1, i2, removed, path)
        except RuntimeError:
            if not removed:
                raise
            # Upstream may already contain this exact replacement. Accept
            # it only when its surrounding base context matches this hunk.
            existing = contiguous_positions(result, added)
            if len(existing) != 1:
                raise
            left, right = context_scores(result, base, i1, i2,
                                         existing[0], len(added))
            if left + right < 3:
                raise
            audit.append((i1+1, "already-present"))
            continue
        if added and result[location:location+len(added)] == added:
            audit.append((i1+1, "already-present"))
            continue
        result[location:location+len(removed)] = added
        audit.append((i1+1, "applied"))
    result_text = "".join(result)
    for marker in ("<<<<<<< ", "=======", ">>>>>>> "):
        if any(line.startswith(marker) for line in result_text.splitlines()):
            raise RuntimeError(f"DSpark functional repair: unresolved marker in {path}")
    return result_text, list(reversed(audit))

def resolve_known_conflicts(repo, conflicts, profile):
    if profile not in PROFILES:
        raise RuntimeError("unknown LFM3 reconciliation profile")
    selected = sorted(conflicts & PROFILES[profile])
    for path in selected:
        base = read_index(repo, 1, path)
        feature = read_index(repo, 2, path)
        upstream = read_index(repo, 3, path)
        merged, audit = transplant(base, feature, upstream, path)
        destination = Path(repo) / path
        destination.write_text(merged)
        command(repo, "add", path)
        print(f"LFMDSPARK_LFM3_RECONCILED={path} edits={len(audit)} decisions={audit}", flush=True)


def migrate_bolt_apis(repo):
    repo = Path(repo)
    gguf = repo / "mistralrs-core/src/pipeline/gguf.rs"
    registry = repo / "mistralrs-core/src/gguf/arch_registry.rs"
    if not gguf.exists():
        return
    text = gguf.read_text()
    start = '        if architecture.eq_ignore_ascii_case("gemma4") {\n'
    if start in text:
        end = '        let metadata_keys = archive\n'
        if text.count(start) != 1 or end not in text:
            raise RuntimeError("unrecognized Gemma4 text GGUF route")
        a = text.index(start); b = text.index(end, a)
        if "SynthesizedMultimodalSource" not in text[a:b]:
            raise RuntimeError("unrecognized Gemma4 synthesis API")
        gguf.write_text(text[:a] + text[b:])
        text = registry.read_text()
        old = 'GgufArchSpec::new(&["gemma4"])\n        .text(GgufTextRoute::ProjectorRequired)'
        if text.count(old) != 1:
            raise RuntimeError("unrecognized Gemma4 architecture registry")
        text = text.replace(old, old.replace('ProjectorRequired', 'Synthesized(synthesize_gemma4_text)'), 1)
        text += '''
fn synthesize_gemma4_text(inputs: &GgufSynthesisInputs<'_>) -> Result<SynthesizedGguf> {
    let archive = inputs.archive;
    let names = archive.tensors().keys().cloned().collect::<Vec<_>>();
    let config = super::gemma4_config::prepare_gemma4_text_config(
        inputs.external_config, archive.metadata(), &names,
    )?;
    Ok(SynthesizedGguf {
        config,
        bindings: build_gemma4_bindings(archive)?,
        loader_type: MultimodalLoaderType::Gemma4,
        processor_config: None,
        preprocessor_config: None,
        rope_pairing: RopePairing::HalfSplit,
    })
}

#[cfg(test)]
mod campaign_gemma4_text_route_tests {
    use super::*;
    #[test]
    fn gemma4_keeps_text_synthesis_and_multimodal_resolution() {
        assert!(matches!(arch_spec("gemma4").text, GgufTextRoute::Synthesized(_)));
        assert!(matches!(arch_spec("GEMMA4").multimodal, GgufMultimodalRoute::Resolve(_)));
        assert!(matches!(arch_spec("qwen3vl").text, GgufTextRoute::ProjectorRequired));
    }
}
'''
        registry.write_text(text)
    path = repo / "mistralrs-core/src/models/lfm2/speculative.rs"
    if path.exists():
        text = path.read_text()
        old = '.embedding_forward(&ids, self.dtype)?'
        if old in text:
            if text.count(old) != 1:
                raise RuntimeError("unrecognized DSpark embedding API")
            path.write_text(text.replace(old, '.embedding_forward(&ids, self.dtype.into())?', 1))
