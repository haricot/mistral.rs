import copy
import tempfile
import unittest
from pathlib import Path

from fixture import create_model, MODEL_SHAPES
from run import compare_models, check_provider


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
        decisions = [dict(shape=[3, n, k], submissions=1, artifact_sha256="abc") for n, k in MODEL_SHAPES]
        evidence = dict(purpose="validation_only", fallback_used=False, resolved_modules=1,
                        prepared_layers=8, decisions=decisions,
                        devices=[dict(compute_capability=61)])
        selected = dict(target_cc=61, decisions=[dict(artifact_sha256="abc")])
        with self.assertRaises(AssertionError):
            check_provider(dict(provider=evidence, provider_after_load=dict(prepared_layers=8)), selected, True)

    def test_local_checkpoint_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = create_model(Path(tmp) / "a"), create_model(Path(tmp) / "b")
            for path in a.iterdir():
                self.assertEqual(path.read_bytes(), (b / path.name).read_bytes())


if __name__ == "__main__":
    unittest.main()
