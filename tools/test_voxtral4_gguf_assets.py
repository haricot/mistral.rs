"""Model-free B2-A regressions for lossless GGUF Tekken byte decoding."""
import struct
import unittest

from voxtral4_gguf_assets import decode_tekken


def encode_fixture(tokens):
    original = b"".join(struct.pack("<H", len(token)) + token for token in tokens)
    packed = struct.pack("<" + "f" * len(original), *original)
    return packed, original


class Voxtral4AssetsTests(unittest.TestCase):
    def test_exact_round_trip_with_binary_non_utf8_token(self):
        raw, packed = encode_fixture([b"", b"\x00\xff", "Bonjour".encode()])
        report = decode_tekken(raw, 3)
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["packed_bytes"], len(packed))
        self.assertEqual(report["max_token_bytes"], 7)
        self.assertEqual(report["empty_tokens"], 1)
        self.assertTrue(report["payload_byte_values_exact"])
        self.assertEqual(report["packed_sha256"], report["parsed_sha256"])

    def test_float_payload_must_represent_exact_bytes(self):
        for bad in [256.0, -1.0, 17.125, float("nan"), float("inf")]:
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    decode_tekken(struct.pack("<f", bad), 1)

    def test_truncated_token_is_rejected(self):
        packed, _ = encode_fixture([b"Bonjour", b"le monde"])
        with self.assertRaises(ValueError):
            decode_tekken(packed[:-4], 2)

    def test_excess_payload_is_rejected(self):
        packed, _ = encode_fixture([b"x"])
        with self.assertRaises(ValueError):
            decode_tekken(packed + struct.pack("<f", 3.0), 1)

    def test_bad_vocab_counts_are_rejected(self):
        packed, _ = encode_fixture([b"x"])
        for count in [0, -1, 200001, 2.5]:
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    decode_tekken(packed, count)


if __name__ == "__main__":
    unittest.main()
