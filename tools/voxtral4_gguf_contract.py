"""Fail-closed, header-only Voxtral4b GGUF inventory contract.

The converter's full 711 named weight bindings + 3 signal/tokenizer tensors
are checked without reading tensor data or claiming numerical/inference parity.
"""
from collections import Counter

TOP = {
    "token_embd.weight", "output_norm.weight",
    "audio.conv.1.weight", "audio.conv.1.bias",
    "audio.conv.2.weight", "audio.conv.2.bias",
    "audio.ln_post.weight", "proj1.weight", "proj2.weight",
}
AUDIO_COMPONENTS = (
    "attn_norm.weight", "ffn_norm.weight",
    "attn_q.weight", "attn_q.bias", "attn_k.weight",
    "attn_v.weight", "attn_v.bias",
    "attn_out.weight", "attn_out.bias",
    "ffn_gate.weight", "ffn_down.weight", "ffn_down.bias",
    "ffn_up.weight",
)
DECODER_COMPONENTS = (
    "attn_norm.weight", "ffn_norm.weight",
    "attn_q.weight", "attn_k.weight", "attn_v.weight",
    "attn_output.weight", "ffn_gate.weight", "ffn_down.weight",
    "ffn_up.weight", "ada_norm_down.weight", "ada_norm_up.weight",
)
AUXILIARY = {
    "audio.mel_filters", "audio.mel_window",
    "tokenizer.tekken.vocab_tensor",
}
# Known source: cstr/voxtral-mini-4b-realtime-q4_k.gguf user header.
# Indices are the GGUF tensor type IDs (0 F32, 1 F16, 12 Q4_K).
KNOWN_DTYPE_COUNTS = {"0": 303, "1": 2, "12": 409}


def expected_inventory(audio_layers, decoder_layers):
    if not isinstance(audio_layers, int) or not 1 <= audio_layers <= 128:
        raise ValueError("invalid audio layer count")
    if not isinstance(decoder_layers, int) or not 1 <= decoder_layers <= 128:
        raise ValueError("invalid decoder layer count")
    weight_names = set(TOP)
    for i in range(audio_layers):
        weight_names.update(f"audio.blk.{i}.{x}" for x in AUDIO_COMPONENTS)
    for i in range(decoder_layers):
        weight_names.update(f"blk.{i}.{x}" for x in DECODER_COMPONENTS)
    return weight_names, weight_names | AUXILIARY


def validate_header_contract(metadata, tensor_info):
    """Return evidence: no tensor bodies read. Never claim runtime PASS."""
    arch = metadata.get("general.architecture")
    namespace = "voxtral4b" if arch == "voxtral4b" else (
        "voxtral4" if arch == "voxtral4" else None
    )
    problems = []
    if namespace is None:
        return {
            "status": "B1_FAIL", "issues": ["unsupported Voxtral architecture"],
            "weights_expected": 0, "missing_tensors": [], "extra_tensors": [],
        }
    # The observed release is 32 audio + 26 decoder layers. Do not silently
    # accept another model revision as qualified by this sample's evidence.
    a = metadata.get(f"{namespace}.audio.n_layers")
    d = metadata.get(f"{namespace}.llm.n_layers")
    if a != 32 or d != 26:
        problems.append(f"unexpected Voxtral4b layer geometry: audio={a}, decoder={d}")
        return {"status": "B1_FAIL", "issues": problems,
                "weights_expected": 711, "missing_tensors": [], "extra_tensors": []}
    weight_names, complete_names = expected_inventory(a, d)
    names = set(tensor_info)
    missing = sorted(complete_names - names)
    extra = sorted(names - complete_names)
    if missing:
        problems.append(f"{len(missing)} missing tensor names")
    if extra:
        problems.append(f"{len(extra)} unexpected tensor names")
    for k, value in (
        (f"{namespace}.audio.d_model", 1280),
        (f"{namespace}.audio.n_heads", 32),
        (f"{namespace}.audio.head_dim", 64),
        (f"{namespace}.llm.d_model", 3072),
        (f"{namespace}.llm.n_heads", 32),
        (f"{namespace}.llm.n_kv_heads", 8),
        (f"{namespace}.llm.head_dim", 128),
        (f"{namespace}.llm.vocab_size", 131072),
        (f"{namespace}.proj.frame_stack", 4),
        (f"{namespace}.proj.in_dim", 5120),
        (f"{namespace}.proj.out_dim", 3072),
        ("tokenizer.tekken.n_vocab", 150000),
        ("tokenizer.tekken.n_specials", 1000),
    ):
        if metadata.get(k) != value:
            problems.append(f"{k}={metadata.get(k)} expected={value}")
    shape_expectations = {
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
    for name, expected_shape in shape_expectations.items():
        if name in tensor_info and tuple(tensor_info[name][0]) != expected_shape:
            problems.append(
                f"{name} has shape {tensor_info[name][0]}, expected {expected_shape}"
            )
    kinds = Counter(str(kind) for _, kind in tensor_info.values())
    if dict(kinds) != KNOWN_DTYPE_COUNTS:
        problems.append(f"unexpected GGUF tensor type counts: {dict(kinds)}")
    return {
        "status": "B1_HEADER_PASS_RUNTIME_NOT_QUALIFIED" if not problems else "B1_FAIL",
        "issues": problems,
        "weights_expected": len(weight_names),
        "auxiliary_expected": len(AUXILIARY),
        "all_tensors_expected": len(complete_names),
        "actual_tensors": len(names),
        "missing_tensors": missing,
        "extra_tensors": extra,
        "tensor_type_counts": dict(sorted(kinds.items())),
        "tokenizer_vs_embedding_vocab": {
            "tekken_vocab": metadata.get("tokenizer.tekken.n_vocab"),
            "embedding_rows": metadata.get(f"{namespace}.llm.vocab_size"),
            "token_id_mapping": "NOT_VALIDATED",
        },
        "quantized_forward": "NOT_VALIDATED",
        "audio_forward": "NOT_VALIDATED",
    }
