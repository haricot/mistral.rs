"""Offline unit tests for the CrispASR capped Tekken rank domain."""
import struct
import unittest

from voxtral4_gguf_token_ids import audit_token_domain


def encode_pieces(pieces):
    raw = b"".join(struct.pack("<H", len(p)) + p for p in pieces)
    return b"".join(struct.pack("<f", float(v)) for v in raw)


class TokenIdDomainTests(unittest.TestCase):
    def test_capped_domain_excludes_inert_tail_and_keeps_specials(self):
        pieces = [bytes([i]) for i in range(256)] + [f"word-{i}".encode() for i in range(30)]
        names = [f"<sp{i}>" for i in range(10)]
        result = audit_token_domain(encode_pieces(pieces), names, len(pieces), 276, 4)
        self.assertEqual(result["active_bpe"], 266)
        self.assertEqual(result["inactive_bpe"], 20)
        self.assertEqual(result["max_active_bpe_id"], 275)
        self.assertEqual(result["first_inactive_bpe_id"], 276)
        self.assertEqual(result["special_audio_token_name"], "<sp4>")
        self.assertEqual(result["active_single_byte_alphabet"], 256)

    def test_all_entries_fit(self):
        pieces = [bytes([i]) for i in range(256)]
        report = audit_token_domain(encode_pieces(pieces), ["a"], 256, 300, 0)
        self.assertEqual(report["inactive_bpe"], 0)
        self.assertIsNone(report["first_inactive_bpe_id"])
        self.assertEqual(report["max_active_bpe_id"], 256)

    def test_rejects_bad_embeddings_and_special_ids(self):
        pieces = [bytes([i]) for i in range(256)]
        for rows, names, audio_id in [(1, ["a"], 0), (300, ["a"], 1), (0, ["a"], 0)]:
            with self.subTest(rows=rows, audio=audio_id):
                with self.assertRaises(ValueError):
                    audit_token_domain(encode_pieces(pieces), names, 256, rows, audio_id)

    def test_rejects_corrupt_f32_or_blob(self):
        pieces = [bytes([i]) for i in range(256)]
        blob = encode_pieces(pieces)
        for bad in [blob[:-3], blob + struct.pack("<f", 1.3), blob + b"\x00"]:
            with self.subTest(length=len(bad)):
                with self.assertRaises(ValueError):
                    audit_token_domain(bad, ["a"], 256, 300, 0)

    def test_byte_alphabet_required(self):
        pieces = [b"a"] * 256
        with self.assertRaisesRegex(ValueError, "256 byte tokens"):
            audit_token_domain(encode_pieces(pieces), ["a"], 256, 300, 0)

    def test_duplicate_special_names_rejected(self):
        pieces = [bytes([i]) for i in range(256)]
        with self.assertRaises(ValueError):
            audit_token_domain(encode_pieces(pieces), ["a", "a"], 256, 300, 0)


if __name__ == "__main__":
    unittest.main()
