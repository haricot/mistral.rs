"""Contract tests with a synthetic 714-tensor inventory (no checkpoint needed)."""
import unittest

from voxtral4_gguf_contract import expected_inventory, validate_header_contract

META = {
    "general.architecture": "voxtral4b",
    "voxtral4b.audio.n_layers": 32,
    "voxtral4b.llm.n_layers": 26,
    "voxtral4b.audio.d_model": 1280,
    "voxtral4b.audio.n_heads": 32,
    "voxtral4b.audio.head_dim": 64,
    "voxtral4b.llm.d_model": 3072,
    "voxtral4b.llm.n_heads": 32,
    "voxtral4b.llm.n_kv_heads": 8,
    "voxtral4b.llm.head_dim": 128,
    "voxtral4b.llm.vocab_size": 131072,
    "voxtral4b.proj.frame_stack": 4,
    "voxtral4b.proj.in_dim": 5120,
    "voxtral4b.proj.out_dim": 3072,
    "tokenizer.tekken.n_vocab": 150000,
    "tokenizer.tekken.n_specials": 1000,
}

SHAPES = {
    "audio.conv.1.weight": (1280, 128, 3),
    "audio.conv.2.weight": (1280, 1280, 3),
    "audio.ln_post.weight": (1280,),
    "audio.mel_filters": (201, 128),
    "audio.mel_window": (400,),
    "output_norm.weight": (3072,),
    "proj1.weight": (3072, 5120),
    "proj2.weight": (3072, 3072),
    "token_embd.weight": (131072, 3072),
    "blk.0.ada_norm_down.weight": (32, 3072),
    "blk.25.ada_norm_up.weight": (3072, 32),
    "audio.blk.0.attn_q.weight": (2048, 1280),
}


def synthetic_inventory():
    weights, names = expected_inventory(32, 26)
    assert len(weights) == 711 and len(names) == 714
    info = {n: (SHAPES.get(n, (1,)), 0) for n in names}
    for n in ("audio.conv.1.weight", "audio.conv.2.weight"):
        info[n] = (info[n][0], 1)
    remaining = sorted(set(names) - {"audio.conv.1.weight", "audio.conv.2.weight"})
    for n in remaining[:409]:
        info[n] = (info[n][0], 12)
    return info


class Voxtral4ContractTests(unittest.TestCase):
    def test_known_release_contract(self):
        got = validate_header_contract(META, synthetic_inventory())
        self.assertEqual(got["status"], "B1_HEADER_PASS_RUNTIME_NOT_QUALIFIED", got)
        self.assertEqual(got["weights_expected"], 711)
        self.assertEqual(got["all_tensors_expected"], 714)
        self.assertEqual(got["tensor_type_counts"], {"0": 303, "1": 2, "12": 409})
        self.assertEqual(got["tokenizer_vs_embedding_vocab"]["token_id_mapping"], "NOT_VALIDATED")

    def test_missing_one_hidden_block_weight_is_caught(self):
        info = synthetic_inventory()
        del info["audio.blk.12.attn_k.weight"]
        got = validate_header_contract(META, info)
        self.assertEqual(got["status"], "B1_FAIL")
        self.assertIn("audio.blk.12.attn_k.weight", got["missing_tensors"])

    def test_unexpected_tensor_is_caught(self):
        info = synthetic_inventory()
        info["unexpected.weight"] = ((12, 12), 12)
        self.assertEqual(validate_header_contract(META, info)["status"], "B1_FAIL")

    def test_wrong_shape_is_caught(self):
        info = synthetic_inventory()
        info["proj1.weight"] = ((3072, 1280), 12)
        result = validate_header_contract(META, info)
        self.assertEqual(result["status"], "B1_FAIL")
        self.assertTrue(any("proj1.weight" in reason for reason in result["issues"]))

    def test_wrong_quantization_distribution_is_caught(self):
        info = synthetic_inventory()
        name = "audio.blk.0.attn_q.weight"
        info[name] = (info[name][0], 0)
        self.assertEqual(validate_header_contract(META, info)["status"], "B1_FAIL")

    def test_model_revision_not_silently_accepted(self):
        other = dict(META, **{"voxtral4b.llm.n_layers": 28})
        self.assertEqual(validate_header_contract(other, synthetic_inventory())["status"], "B1_FAIL")


if __name__ == "__main__":
    unittest.main()
