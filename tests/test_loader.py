"""The checkpoint loader, against `transformers` itself.

Most of this file is marked `oracle` because it needs the real Qwen3-0.6B
weights. The mapping logic is pure string manipulation though, so that part is
tested without any download.
"""

from __future__ import annotations

import json

import pytest
import torch

from mini_vllm.model.loader import (
    GLOBAL_NAMES,
    LAYER_NAMES,
    TIED_LM_HEAD,
    ModelConfig,
    expected_names,
    expected_shape,
    load_weights,
    map_name,
    resolve_model_path,
)

# The Qwen3-0.6B architecture table. Hardcoded so a silently different checkpoint
# is caught rather than absorbed.
QWEN3_06B = {
    "num_hidden_layers": 28,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000,
    "tie_word_embeddings": True,
}


# ------------------------------------------------------ mapping, no weights


def test_maps_every_global_and_layer_name():
    assert map_name("model.embed_tokens.weight") == "embedding"
    assert map_name("model.norm.weight") == "final_norm"
    assert map_name("model.layers.7.self_attn.q_proj.weight") == "layers.7.attn.wq"
    assert map_name("model.layers.27.mlp.down_proj.weight") == "layers.27.mlp.down"
    assert map_name("model.layers.0.input_layernorm.weight") == "layers.0.attn_norm"
    assert map_name("model.layers.0.post_attention_layernorm.weight") == "layers.0.mlp_norm"


def test_tied_lm_head_is_dropped_deliberately():
    """None means "recognized and intentionally not stored", not "unknown"."""
    assert map_name(TIED_LM_HEAD) is None


def test_unknown_name_raises_rather_than_being_ignored():
    """Silently skipping an unrecognized weight is how you ship a broken model."""
    with pytest.raises(KeyError, match="unmapped checkpoint weight"):
        map_name("model.layers.0.self_attn.rotary_emb.inv_freq")
    with pytest.raises(KeyError, match="unmapped checkpoint weight"):
        map_name("something.entirely.made.up")


def test_expected_names_covers_every_layer():
    config = ModelConfig(**QWEN3_06B, max_position_embeddings=40960)
    names = expected_names(config)

    assert len(names) == len(GLOBAL_NAMES) + 28 * len(LAYER_NAMES)
    assert "layers.27.attn.k_norm" in names
    assert "layers.28.attn.wq" not in names


def test_config_derives_gqa_and_projection_sizes():
    config = ModelConfig(**QWEN3_06B, max_position_embeddings=40960)

    assert config.group_size == 2
    # The trap: the attention projection is twice the hidden size.
    assert config.q_projection_size == 2048
    assert config.kv_projection_size == 1024
    assert config.q_projection_size != config.hidden_size


def test_config_rejects_indivisible_head_counts():
    bad = QWEN3_06B | {"num_key_value_heads": 5}
    with pytest.raises(ValueError, match="must be a multiple"):
        ModelConfig(**bad, max_position_embeddings=40960)


def test_expected_shapes_follow_the_design_table():
    config = ModelConfig(**QWEN3_06B, max_position_embeddings=40960)

    assert expected_shape("embedding", config) == (151936, 1024)
    assert expected_shape("layers.0.attn.wq", config) == (2048, 1024)
    assert expected_shape("layers.0.attn.wk", config) == (1024, 1024)
    assert expected_shape("layers.0.attn.wo", config) == (1024, 2048)
    assert expected_shape("layers.0.attn.q_norm", config) == (128,)
    assert expected_shape("layers.0.mlp.gate", config) == (3072, 1024)
    assert expected_shape("layers.0.mlp.down", config) == (1024, 3072)


# ---------------------------------------------------- against the real weights


@pytest.fixture(scope="module")
def model_path():
    path = resolve_model_path()
    if not (path / "model.safetensors").is_file():
        pytest.skip("Qwen3-0.6B weights are not downloaded")
    return path


@pytest.fixture(scope="module")
def loaded(model_path):
    return load_weights(model_path)


@pytest.mark.oracle
def test_config_matches_the_design_document(model_path):
    raw = json.loads((model_path / "config.json").read_text())
    config = ModelConfig.from_dict(raw)

    for field, value in QWEN3_06B.items():
        assert getattr(config, field) == value, f"{field} differs from QWEN3_06B"
    assert config.dtype is torch.bfloat16


@pytest.mark.oracle
def test_mapping_is_total_in_both_directions(loaded, model_path):
    """Nothing in the checkpoint is ignored, nothing in the model is unfilled.

    `load_weights` already enforces this, so reaching here is most of the test;
    the assertions below pin the counts so a future checkpoint that adds or drops
    a tensor is noticed.
    """
    weights, config = loaded

    assert weights.keys() == expected_names(config)
    assert len(weights) == 2 + 28 * 11

    from safetensors import safe_open

    with safe_open(model_path / "model.safetensors", framework="pt") as handle:
        checkpoint_keys = set(handle.keys())

    # Every checkpoint key is either mapped or explicitly dropped.
    mapped = {key for key in checkpoint_keys if map_name(key) is not None}
    assert len(checkpoint_keys) == 311
    assert mapped == checkpoint_keys - {TIED_LM_HEAD}


@pytest.mark.oracle
def test_every_tensor_is_bitwise_equal_to_transformers(loaded):
    """The strongest available check: not close, *equal*.

    Loading is a pure data move, so anything short of bitwise equality means a
    transpose, a dtype cast, or a wrong key.
    """
    weights, _config = loaded

    from transformers import AutoModelForCausalLM

    reference = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16
    )
    reference_state = reference.state_dict()

    for hf_name, hf_tensor in reference_state.items():
        ours = map_name(hf_name)
        if ours is None:
            continue
        assert torch.equal(weights[ours], hf_tensor), f"{hf_name} -> {ours} is not bitwise equal"


@pytest.mark.oracle
def test_tied_lm_head_really_is_the_embedding(model_path):
    """Justifies dropping `lm_head.weight` instead of storing it.

    If these ever differed, reading logits off the embedding would be wrong and
    the tie flag would be lying.
    """
    from safetensors import safe_open

    with safe_open(model_path / "model.safetensors", framework="pt") as handle:
        embedding = handle.get_tensor("model.embed_tokens.weight")
        lm_head = handle.get_tensor(TIED_LM_HEAD)

    assert torch.equal(embedding, lm_head)


@pytest.mark.oracle
def test_loaded_dtype_and_shapes(loaded):
    weights, config = loaded

    for name, tensor in weights.items():
        assert tensor.dtype is torch.bfloat16, f"{name} is {tensor.dtype}"
        assert tuple(tensor.shape) == expected_shape(name, config), name


@pytest.mark.oracle
def test_a_wrong_config_is_rejected(model_path):
    """A config that disagrees with the checkpoint must fail at load time."""
    wrong = ModelConfig(**(QWEN3_06B | {"hidden_size": 999}), max_position_embeddings=40960)
    with pytest.raises(ValueError, match="expected shape"):
        load_weights(model_path, wrong)
