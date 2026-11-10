import copy
import json
import tempfile
import unittest
from pathlib import Path

from fixture import create_model, HIDDEN, INTERMEDIATE, MODEL_SHAPES
from run import compare_models, check_provider, profile, reuse_existing, sha


class ReviewTests(unittest.TestCase):
    def model(self):
        return dict(prompt_tokens=[1, 2, 3], logits=[[0.1, 0.2, 1.0]], response=dict(choices=[dict(
            message=dict(content="proof proof proof proof"), finish_reason="length",
            logprobs=dict(content=[dict(token="proof", logprob=-0.1)] * 4))]))

    def test_rejects_wrong_logits_tokens_and_incomplete_generation(self):
        reference = self.model()
        for change in ["logits", "tokens", "length", "nan"]:
            actual = copy.deepcopy(reference)
            if change == "logits":
                actual["logits"][0][0] = 10.0
            elif change == "tokens":
                actual["prompt_tokens"] = [1, 3, 2]
            elif change == "nan":
                actual["logits"][0][0] = float("nan")
            else:
                actual["response"]["choices"][0]["logprobs"]["content"].pop()
            with self.subTest(change=change), self.assertRaises(AssertionError):
                compare_models(actual, reference)

    def test_rejects_prefill_only_provider_evidence(self):
        decisions = [dict(decision_id=f"prefill-{n}-{k}", shape=[3, n, k], submissions=1, artifact_sha256="abc") for n, k in MODEL_SHAPES]
        evidence = dict(purpose="validation_only", fallback_used=False, resolved_modules=1,
                        prepared_layers=8, decisions=decisions,
                        devices=[dict(compute_capability=61)])
        selected = dict(target_cc=61, decisions=[dict(artifact_sha256="abc")])
        with self.assertRaises(AssertionError):
            check_provider(dict(provider=evidence, provider_after_load=dict(prepared_layers=8)), selected, True)

    def test_rejects_provider_submissions_only_during_model_load(self):
        decisions = [dict(decision_id=f"{m}-{n}-{k}", shape=[m,n,k], submissions=2,
                          artifact_sha256="abc") for m in [1,3] for n,k in MODEL_SHAPES]
        evidence = dict(purpose="validation_only", fallback_used=False, resolved_modules=1,
                        prepared_layers=8, decisions=decisions,
                        devices=[dict(compute_capability=61)])
        report = dict(provider=evidence, provider_after_load=copy.deepcopy(evidence))
        selected = dict(target_cc=61, decisions=[dict(artifact_sha256="abc")])
        with self.assertRaises(AssertionError):
            check_provider(report, selected, True)

    def test_packed_gate_up_execution_requires_fused_not_individual_projection(self):
        active = [(HIDDEN, HIDDEN), (HIDDEN, INTERMEDIATE),
                  (2 * INTERMEDIATE, HIDDEN)]
        selected = dict(target_cc=61, decisions=[
            dict(shape=[m, n, k], artifact_sha256="abc")
            for m in (1, 3) for n, k in MODEL_SHAPES
        ])
        decisions = [
            dict(decision_id=f"{m}-{n}-{k}", shape=[m, n, k],
                 submissions=1, artifact_sha256="abc")
            for m in (1, 3) for n, k in active
        ]
        evidence = dict(purpose="validation_only", fallback_used=False,
                        resolved_modules=1, prepared_layers=10, decisions=decisions,
                        devices=[dict(compute_capability=61)])
        report = dict(provider=evidence,
                      provider_after_load=dict(prepared_layers=10, decisions=[]))
        check_provider(report, selected, True)
        for omitted in ((1, 2 * INTERMEDIATE, HIDDEN),
                        (3, 2 * INTERMEDIATE, HIDDEN)):
            with self.subTest(omitted=omitted):
                broken = copy.deepcopy(report)
                broken["provider"]["decisions"] = [
                    d for d in decisions if tuple(d["shape"]) != omitted
                ]
                with self.assertRaisesRegex(AssertionError, "required provider shapes"):
                    check_provider(broken, selected, True)

    def test_packed_projection_has_exact_prefill_and_decode_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest = directory / "module.json"
            manifest.write_text(json.dumps({
                "implementation_id": "test", "sha256": "0" * 64, "target_cc": 61,
            }))
            selected = profile(manifest, directory / "A.profile.json", "A")
            shapes = {tuple(row["shape"]) for row in selected["decisions"]}
            self.assertIn((1, 4 * HIDDEN, HIDDEN), shapes)
            self.assertIn((3, 4 * HIDDEN, HIDDEN), shapes)
            self.assertEqual(len(shapes), len(selected["decisions"]))


    def prepare_reuse(self, root):
        source = root / "previous"
        output = root / "replay"
        source.mkdir()
        output.mkdir()
        binary = source / "target/debug/examples/provider_f32_inference"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"previously-built-sm61-binary")
        binary_hash = sha(binary)
        for label in ("cpu-model", "baseline-model", "A-operator"):
            (source / f"{label}.json").write_text(json.dumps({
                "status": "PASS", "binary_sha256": binary_hash,
            }))
        for name, entry in (("a", "linear_simple"), ("b", "linear_tiled")):
            folder = source / "packages" / name
            folder.mkdir(parents=True)
            ptx = folder / "kernel.ptx"
            ptx.write_text(f".version 8.0\\n.target sm_61\\n.visible .entry {entry}() {{}}\\n")
            (folder / "module.json").write_text(json.dumps({
                "schema": 1, "abi": "mistral.linear.f32.v1",
                "target_cc": 61, "min_cc": 61, "max_cc": 61,
                "artifact": "kernel.ptx", "entry": entry, "dtype": "f32",
                "sha256": sha(ptx),
            }))
        return source, output, binary

    def test_reuses_previous_binary_and_ptx_without_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output, original = self.prepare_reuse(Path(tmp))
            binary, manifests, proof = reuse_existing(source, output)
            self.assertEqual(binary, original)
            self.assertEqual(sha(binary), sha(original))
            self.assertEqual([p.parent.name for p in manifests], ["a", "b"])
            self.assertEqual(proof["source"], str(source))
            for manifest in manifests:
                self.assertEqual(sha(manifest.parent / "kernel.ptx"),
                                 json.loads(manifest.read_text())["sha256"])
            self.assertTrue((source / "packages/a/kernel.ptx").exists())

    def test_reuse_rejects_corrupted_manifest_and_binary_checkpoint(self):
        for change in ("manifest", "binary", "dirty_output"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                source, output, binary = self.prepare_reuse(Path(tmp))
                if change == "manifest":
                    path = source / "packages/a/kernel.ptx"
                    path.write_text("corrupted")
                elif change == "binary":
                    binary.write_bytes(b"recompiled")
                else:
                    (output / "existing.txt").write_text("do not overwrite")
                with self.assertRaises((RuntimeError, FileExistsError)):
                    reuse_existing(source, output)

    def test_local_checkpoint_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = create_model(Path(tmp) / "a"), create_model(Path(tmp) / "b")
            for path in a.iterdir():
                self.assertEqual(path.read_bytes(), (b / path.name).read_bytes())


if __name__ == "__main__":
    unittest.main()

