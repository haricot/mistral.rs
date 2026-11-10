import json
from pathlib import Path
import struct

HIDDEN = 32
INTERMEDIATE = 64
VOCAB = 32
# The Llama loader also prepares a packed F32 projection with 4 * HIDDEN rows.
# It must have an exact ASD decision: configured providers cannot fall back.
MODEL_SHAPES = [
    (HIDDEN, HIDDEN),
    (INTERMEDIATE, HIDDEN),
    (HIDDEN, INTERMEDIATE),
    (4 * HIDDEN, HIDDEN),
]
OPERATOR_SHAPE = (17, 33)
ROW_COUNTS = [1, 3, 5, 6]


def create_model(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(architectures=["LlamaForCausalLM"], model_type="llama", hidden_act="silu",
                  hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, vocab_size=VOCAB,
                  num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                  rms_norm_eps=1e-5, rope_theta=10000.0, max_position_embeddings=64,
                  tie_word_embeddings=False, torch_dtype="float32", bos_token_id=0,
                  eos_token_id=31, pad_token_id=31)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokens = ["<s>", "hello", "world", "mechanism", "proof"] + [f"word{i}" for i in range(5, 31)] + ["</s>"]
    tokenizer = dict(version="1.0", truncation=None, padding=None, added_tokens=[dict(id=i, content=t, single_word=False,
                         lstrip=False, rstrip=False, normalized=False, special=True)
                         for i,t in [(0,"<s>"),(31,"</s>")]],
                     normalizer=None, pre_tokenizer={"type": "Whitespace"}, post_processor=None,
                     decoder=dict(type="ByteLevel", add_prefix_space=False, trim_offsets=True, use_regex=True), model={"type": "WordLevel", "vocab": {t:i for i,t in enumerate(tokens)},
                                          "unk_token": "<s>"})
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer) + "\n")
    (directory / "tokenizer_config.json").write_text(json.dumps(dict(
        bos_token="<s>", eos_token="</s>", unk_token="<s>", pad_token="</s>",
        chat_template="{% for message in messages %}{{ message['content'] }} {% endfor %}")) + "\n")
    tensors = {"model.embed_tokens.weight": [VOCAB, HIDDEN], "model.norm.weight": [HIDDEN],
               "lm_head.weight": [VOCAB, HIDDEN],
               "model.layers.0.input_layernorm.weight": [HIDDEN],
               "model.layers.0.post_attention_layernorm.weight": [HIDDEN]}
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        tensors[f"model.layers.0.self_attn.{name}.weight"] = [HIDDEN, HIDDEN]
    for name in ["gate_proj", "up_proj"]:
        tensors[f"model.layers.0.mlp.{name}.weight"] = [INTERMEDIATE, HIDDEN]
    tensors["model.layers.0.mlp.down_proj.weight"] = [HIDDEN, INTERMEDIATE]
    header, payload = {}, bytearray()
    for ordinal, (name, shape) in enumerate(tensors.items()):
        count = 1
        for dim in shape:
            count *= dim
        if len(shape) == 1:
            values = [1.0] * count
        elif name == "model.embed_tokens.weight":
            values = [0.2 + (i * 7 % 31) / 256 for i in range(count)]
        elif name == "lm_head.weight":
            values = [0.25 if i // HIDDEN == 4 else (-0.25 if i // HIDDEN == 31 else
                      ((i * 11 % 17) - 8) / 2048) for i in range(count)]
        else:
            values = [((i * 7 + ordinal * 3) % 29 - 14) / 2048 for i in range(count)]
        start = len(payload)
        payload.extend(struct.pack(f"<{count}f", *values))
        header[name] = dict(dtype="F32", shape=shape, data_offsets=[start, len(payload)])
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return directory
