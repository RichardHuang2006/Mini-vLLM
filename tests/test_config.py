"""Step 0.2 -- config validation and KV cache sizing arithmetic.

The byte math here is checked against numbers worked out by hand in PLAN.md.
It is worth over-testing: a wrong constant produces no error, just an engine
that holds the wrong number of sequences.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from nanovllm.config import Config, KVGeometry, kv_geometry, resolve_dtype

KIB = 1024


def hf_stub(**fields) -> SimpleNamespace:
    """Stand-in for a HuggingFace config; only the read fields matter."""
    return SimpleNamespace(**fields)


# Real values, fetched from the published config.json of each model.
QWEN25_05B = hf_stub(
    num_hidden_layers=24,
    hidden_size=896,
    num_attention_heads=14,
    num_key_value_heads=2,
    head_dim=None,  # absent in the published config; must be derived as 64
)
QWEN25_15B = hf_stub(
    num_hidden_layers=28,
    hidden_size=1536,
    num_attention_heads=12,
    num_key_value_heads=2,
    head_dim=None,  # derived: 128
)
QWEN3_06B = hf_stub(
    num_hidden_layers=28,
    hidden_size=1024,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,  # explicit, and NOT hidden_size // num_heads (which is 64)
)


# --------------------------------------------------------------------------
# dtype resolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("float8_e4m3fn", torch.float8_e4m3fn),
        (torch.float32, torch.float32),
    ],
)
def test_resolve_dtype(value, expected):
    assert resolve_dtype(value) is expected


@pytest.mark.parametrize("value", ["float17", "", "nn"])
def test_resolve_dtype_rejects_junk(value):
    """`nn` is the interesting one: it exists on torch, but is a module."""
    with pytest.raises(ValueError, match="unknown dtype"):
        resolve_dtype(value)


def test_dtype_strings_are_normalized_on_construction():
    config = Config(model="m", dtype="bfloat16")
    assert config.dtype is torch.bfloat16


def test_kv_cache_dtype_defaults_to_dtype():
    assert Config(model="m", dtype="bfloat16").kv_cache_dtype is torch.bfloat16


def test_kv_cache_dtype_can_differ_from_dtype():
    """Phase 6: FP8 cache while the model still runs in FP16."""
    config = Config(model="m", dtype="float16", kv_cache_dtype="float8_e4m3fn")
    assert config.dtype is torch.float16
    assert config.kv_cache_dtype is torch.float8_e4m3fn


# --------------------------------------------------------------------------
# reading model geometry
# --------------------------------------------------------------------------


def test_geometry_derives_absent_head_dim():
    assert kv_geometry(QWEN25_05B) == KVGeometry(num_layers=24, num_kv_heads=2, head_dim=64)


def test_geometry_prefers_explicit_head_dim_over_derivation():
    """The trap: for Qwen3, deriving would give 64 and halve the cache."""
    assert QWEN3_06B.hidden_size // QWEN3_06B.num_attention_heads == 64
    assert kv_geometry(QWEN3_06B).head_dim == 128


def test_geometry_falls_back_to_mha_when_gqa_absent():
    config = hf_stub(num_hidden_layers=4, hidden_size=512, num_attention_heads=8)
    assert kv_geometry(config) == KVGeometry(num_layers=4, num_kv_heads=8, head_dim=64)


def test_geometry_rejects_indivisible_hidden_size():
    config = hf_stub(num_hidden_layers=4, hidden_size=100, num_attention_heads=8)
    with pytest.raises(ValueError, match="cannot derive head_dim"):
        kv_geometry(config)


def test_geometry_rejects_ragged_gqa_grouping():
    config = hf_stub(
        num_hidden_layers=4, hidden_size=512, num_attention_heads=8, num_key_value_heads=3
    )
    with pytest.raises(ValueError, match="not divisible"):
        kv_geometry(config)


def test_geometry_reports_missing_fields():
    with pytest.raises(ValueError, match="num_hidden_layers"):
        kv_geometry(hf_stub(hidden_size=512, num_attention_heads=8))


# --------------------------------------------------------------------------
# the sizing arithmetic (PLAN.md Step 0.2 "Done when")
# --------------------------------------------------------------------------


def test_qwen25_05b_block_size_worked_by_hand():
    """PLAN.md: 8 KiB per layer per block, 192 KiB per block, 12 KiB per token."""
    config = Config(model="Qwen/Qwen2.5-0.5B-Instruct", block_size=16, dtype="float16")

    per_layer_per_block = 2 * 16 * 2 * 64 * 2
    assert per_layer_per_block == 8 * KIB

    assert config.kv_cache_bytes_per_block(QWEN25_05B) == per_layer_per_block * 24
    assert config.kv_cache_bytes_per_block(QWEN25_05B) == 192 * KIB
    assert config.kv_cache_bytes_per_token(QWEN25_05B) == 12 * KIB


@pytest.mark.parametrize(
    ("hf_config", "expected_kib_per_token"),
    [(QWEN25_05B, 12), (QWEN25_15B, 28), (QWEN3_06B, 112)],
)
def test_per_token_cost_across_models(hf_config, expected_kib_per_token):
    """Qwen3 costs ~9x Qwen2.5-0.5B per token: 8 KV heads at 128 vs 2 at 64."""
    config = Config(model="m", block_size=16, dtype="float16")
    assert config.kv_cache_bytes_per_token(hf_config) == expected_kib_per_token * KIB


def test_layer_count_is_included():
    """Guards the classic bug: dropping num_layers from the product."""
    config = Config(model="m", block_size=16, dtype="float16")
    single_layer = hf_stub(
        num_hidden_layers=1, hidden_size=896, num_attention_heads=14, num_key_value_heads=2
    )
    assert config.kv_cache_bytes_per_block(QWEN25_05B) == (
        24 * config.kv_cache_bytes_per_block(single_layer)
    )


def test_fp8_cache_is_exactly_half_of_fp16():
    """DESIGN.md 5.2 promises a 2x reduction; this is that claim as arithmetic."""
    common = dict(model="m", block_size=16, dtype="float16")
    fp16 = Config(**common)
    fp8 = Config(**common, kv_cache_dtype="float8_e4m3fn")
    assert fp8.kv_cache_bytes_per_block(QWEN25_05B) * 2 == fp16.kv_cache_bytes_per_block(
        QWEN25_05B
    )


def test_block_bytes_scale_linearly_with_block_size():
    small = Config(model="m", block_size=16, dtype="float16")
    large = Config(model="m", block_size=32, dtype="float16")
    assert large.kv_cache_bytes_per_block(QWEN25_05B) == (
        2 * small.kv_cache_bytes_per_block(QWEN25_05B)
    )
    # Per-token cost is a property of the model, not of how it is partitioned.
    assert large.kv_cache_bytes_per_token(QWEN25_05B) == small.kv_cache_bytes_per_token(
        QWEN25_05B
    )


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [1, 2, 16, 32, 256])
def test_powers_of_two_accepted(block_size):
    assert Config(model="m", block_size=block_size).block_size == block_size


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"model": ""}, "non-empty"),
        ({"block_size": 24}, "power of two"),
        ({"block_size": 0}, "power of two"),
        ({"block_size": -16}, "power of two"),
        ({"max_num_batched_tokens": 8}, "at least block_size"),
        ({"max_seq_len": 0}, "max_seq_len must be positive"),
        ({"max_num_seqs": 0}, "max_num_seqs must be positive"),
        ({"gpu_memory_utilization": 0.0}, "gpu_memory_utilization"),
        ({"gpu_memory_utilization": 1.5}, "gpu_memory_utilization"),
        ({"dtype": "int32"}, "dtype must be a floating-point type"),
        ({"kv_cache_dtype": "int8"}, "kv_cache_dtype must be a floating-point type"),
    ],
)
def test_invalid_configs_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Config(**{"model": "m", **kwargs})


def test_config_is_frozen():
    config = Config(model="m")
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.block_size = 32  # type: ignore[misc]


def test_defaults_are_self_consistent():
    """The defaults must themselves pass validation."""
    config = Config(model="m")
    assert config.block_size == 16
    assert config.max_num_batched_tokens == 2048  # DESIGN.md 4.3 token budget
    assert config.enable_prefix_caching is True


# --------------------------------------------------------------------------
# against the real published configs
# --------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("Qwen/Qwen2.5-0.5B-Instruct", KVGeometry(24, 2, 64)),
        ("Qwen/Qwen3-0.6B", KVGeometry(28, 8, 128)),
    ],
)
def test_geometry_matches_published_configs(model_id, expected):
    """Guards the stubs above against drift in the real configs."""
    from transformers import AutoConfig

    assert kv_geometry(AutoConfig.from_pretrained(model_id)) == expected
