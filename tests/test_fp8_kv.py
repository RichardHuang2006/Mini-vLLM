"""FP8 KV cache: storage that costs half as much and reads back what it stored.

The cache dominates an inference engine's memory, exceeding the weights for a long
context, so halving it doubles the sequences that fit in flight. FP8 stores each key and
value in one byte instead of two, dividing by a scale on the way in and multiplying back
on the way out.

These tests cover the storage half and run on CPU with no kernel:

* Size: an FP8 pool of a given geometry is exactly half the bytes of the bf16 one.
* Round-trip: a value written and read back survives to e4m3's precision and no better.
  The quantization is lossy by design and the test pins the loss.
* Copy-on-write: a page copied through the byte view carries its quantized contents
  intact, so a forked FP8 sequence attends over real data rather than zeros.

The kernel half — the paged attention kernel dequantizing in registers and landing within
a divergence budget of the bf16 run — is in the CUDA-only tests below.
"""

from __future__ import annotations

import pytest
import torch

from mini_vllm.block.kv_pool import PagedKvPool

FP8_MAX_ERROR = 0.07  # e4m3 has 3 mantissa bits: ~2^-3 relative, comfortably under this


def pool(**kwargs) -> PagedKvPool:
    defaults = dict(
        num_layers=1, num_blocks=8, block_size=4, num_kv_heads=2, head_dim=8, dtype=torch.bfloat16
    )
    defaults.update(kwargs)
    return PagedKvPool(**defaults)


# ------------------------------------------------------------------------ size


def test_fp8_pool_is_half_the_bytes_of_bf16():
    shape = dict(num_layers=28, num_blocks=1024, block_size=16, num_kv_heads=8, head_dim=128)
    bf16 = PagedKvPool.bytes_for(**shape, dtype=torch.bfloat16)
    fp8 = PagedKvPool.bytes_for(**shape, dtype=torch.float8_e4m3fn)
    assert fp8 * 2 == bf16, "one byte per element against two"


def test_a_pool_knows_when_it_is_quantized():
    assert not pool().is_fp8
    quantized = pool(kv_dtype=torch.float8_e4m3fn)
    assert quantized.is_fp8
    assert quantized.dtype is torch.bfloat16, "the activation dtype is unchanged"
    assert quantized.kv_dtype is torch.float8_e4m3fn
    assert quantized.keys.dtype is torch.float8_e4m3fn, "storage is the quantized type"
    assert "fp8" in repr(quantized).lower() or "float8" in repr(quantized).lower()


# ------------------------------------------------------------------ round-trip


def test_write_then_gather_survives_to_fp8_precision():
    quantized = pool(kv_dtype=torch.float8_e4m3fn)
    # One block's worth of tokens with values inside e4m3's comfortable range.
    tokens = quantized.block_size
    key = torch.randn(tokens, quantized.num_kv_heads, quantized.head_dim, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    slots = torch.arange(tokens)  # block 0, offsets 0..P-1

    quantized.write(0, slots, key, value)
    keys, values = quantized.gather(0, [0], tokens)  # 1 x H_k x tokens x D

    got_k = keys.squeeze(0).permute(1, 0, 2)  # tokens x H_k x D
    got_v = values.squeeze(0).permute(1, 0, 2)
    assert torch.allclose(got_k.float(), key.float(), atol=FP8_MAX_ERROR, rtol=0.1)
    assert torch.allclose(got_v.float(), value.float(), atol=FP8_MAX_ERROR, rtol=0.1)


def test_a_scale_widens_the_representable_range():
    """A value past e4m3's ~448 ceiling only survives if a scale brings it back in range."""
    big = pool(kv_dtype=torch.float8_e4m3fn, k_scale=1.0)
    huge_key = torch.full((1, big.num_kv_heads, big.head_dim), 1000.0, dtype=torch.bfloat16)
    big.write(0, torch.tensor([0]), huge_key, huge_key)
    unscaled, _ = big.gather(0, [0], 1)
    lost = unscaled.float()
    assert not torch.isfinite(lost).all() or lost.abs().max() < 900.0, (
        "1000 sits past e4m3's ~448 ceiling; without a scale it overflows or saturates"
    )

    scaled = pool(kv_dtype=torch.float8_e4m3fn, k_scale=8.0, v_scale=8.0)
    scaled.write(0, torch.tensor([0]), huge_key, huge_key)
    recovered, _ = scaled.gather(0, [0], 1)
    assert torch.isfinite(recovered.float()).all(), "the scale kept the stored value in range"
    assert recovered.float().max() > 900.0, "and dequantizing carried it back near 1000"


# -------------------------------------------------------------- copy-on-write


def test_copy_block_carries_the_quantized_bytes():
    quantized = pool(kv_dtype=torch.float8_e4m3fn)
    tokens = quantized.block_size
    key = torch.randn(tokens, quantized.num_kv_heads, quantized.head_dim, dtype=torch.bfloat16)
    quantized.write(0, torch.arange(tokens), key, key)

    quantized.copy_block(source=0, destination=3)

    original, _ = quantized.gather(0, [0], tokens)
    copied, _ = quantized.gather(0, [3], tokens)
    assert torch.equal(original, copied), "the copy did not reproduce the page byte for byte"


# ------------------------------------------------------- fused quantize (CUDA)


@pytest.mark.cuda
def test_fused_quantize_scatter_matches_the_two_pass_oracle():
    """The fused kernel writes the same bytes the PyTorch two-pass path does.

    Both divide by the scale in fp32 and round to nearest even, so the stored FP8 bytes
    must be bit-identical rather than merely close. Comparing the uint8 reinterpretations
    is how "the same quantization" gets checked without a tolerance that would hide a
    disagreement about rounding.
    """
    from mini_vllm.kernels import ops

    torch.manual_seed(0)
    device = "cuda"
    num_slots, num_kv_heads, head_dim, tokens = 32, 2, 64, 20
    k_scale, v_scale = 0.5, 0.25

    key = torch.randn(tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    # Scattered, non-monotonic slots: the paged case, where a token's page is wherever
    # the allocator had room. A kernel that assumed contiguity would pass on arange.
    slots = torch.randperm(num_slots, device=device)[:tokens].to(torch.int64)

    def empty_pool() -> torch.Tensor:
        shape = (num_slots, num_kv_heads, head_dim)
        return torch.zeros(shape, device=device, dtype=torch.float8_e4m3fn)

    fused_k, fused_v = empty_pool(), empty_pool()
    ops.quantize_scatter(key, value, fused_k, fused_v, slots, k_scale, v_scale, use_cuda=True)

    oracle_k, oracle_v = empty_pool(), empty_pool()
    ops.quantize_scatter(key, value, oracle_k, oracle_v, slots, k_scale, v_scale, use_cuda=False)

    assert torch.equal(fused_k.view(torch.uint8), oracle_k.view(torch.uint8))
    assert torch.equal(fused_v.view(torch.uint8), oracle_v.view(torch.uint8))


@pytest.mark.cuda
def test_fused_quantize_leaves_untouched_slots_alone():
    """A scatter writes the tokens' slots and nothing else — no page it was not given."""
    from mini_vllm.kernels import ops

    device = "cuda"
    num_slots, num_kv_heads, head_dim = 16, 2, 32
    key_pool = torch.zeros(num_slots, num_kv_heads, head_dim, device=device, dtype=torch.float8_e4m3fn)
    value_pool = torch.zeros_like(key_pool)

    written = torch.tensor([2, 5], dtype=torch.int64, device=device)
    key = torch.ones(2, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    ops.quantize_scatter(key, key, key_pool, value_pool, written, 1.0, 1.0, use_cuda=True)

    touched = key_pool.float().abs().sum(dim=(1, 2)) > 0
    assert touched.nonzero().flatten().tolist() == [2, 5]


# ------------------------------------------------------- kernel accuracy (CUDA)


@pytest.mark.cuda
def test_fp8_kernel_matches_the_dequant_oracle():
    """The paged kernel on an FP8 pool agrees with the gather oracle on the same pool.

    Both dequantize the same stored bytes by the same scales — the kernel in registers,
    the oracle into a temporary — so the only thing that could separate them is the
    kernel's arithmetic, and it must not. A decode step over a few-hundred-token context
    is the case the kernel is built for, so that is what is checked.
    """
    from mini_vllm.kernels import ops
    from mini_vllm.paged_attention import paged_attention_gathered

    torch.manual_seed(0)
    device = "cuda"
    num_blocks, block_size, num_kv_heads, head_dim, num_query_heads = 64, 16, 2, 64, 4
    context = 200
    k_scale, v_scale = 0.5, 0.25

    # A pool laid out exactly as the engine's: one decode query per sequence.
    key_pool = (torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=device) / 4)
    value_pool = torch.randn_like(key_pool) / 4
    key_fp8 = (key_pool / k_scale).to(torch.float8_e4m3fn)
    value_fp8 = (value_pool / v_scale).to(torch.float8_e4m3fn)

    q = torch.randn(1, num_query_heads, head_dim, device=device, dtype=torch.bfloat16)
    blocks_needed = -(-context // block_size)
    table = torch.arange(blocks_needed, dtype=torch.int32, device=device).unsqueeze(0)
    cu = torch.tensor([0, 1], dtype=torch.int32, device=device)
    context_lens = torch.tensor([context], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([1], dtype=torch.int32, device=device)

    kernel = ops.paged_attention(
        q, key_fp8, value_fp8, table, cu, context_lens, seq_lens,
        max_query_len=1, max_context_len=context, use_cuda=True,
        k_scale=k_scale, v_scale=v_scale,
    )
    oracle = paged_attention_gathered(
        q, key_fp8, value_fp8, table, cu, context_lens, k_scale=k_scale, v_scale=v_scale
    )
    torch.testing.assert_close(kernel.float(), oracle.float(), atol=5e-2, rtol=5e-2)


# ------------------------------------------------------------ end-to-end (CUDA)


def test_fp8_halves_what_a_page_costs():
    """The central arithmetic: a page costs half as much in FP8.

    Not measured through `LLM.blocks_that_fit`, which reads the card's live free memory and
    would therefore depend on whatever else the test session has allocated: the result
    would vary with test order, and on a full card it bottoms out at its one-page floor and
    compares 1 against 1. The claim concerns the model's shape and a dtype's width, both
    known exactly, so the sizing arithmetic is the right instrument.
    """
    from mini_vllm.model.loader import ModelConfig, resolve_model_path

    path = resolve_model_path()
    if not (path / "config.json").is_file():
        pytest.skip("Qwen3-0.6B config is not downloaded")

    config = ModelConfig.from_pretrained(path)
    shape = dict(
        num_layers=config.num_hidden_layers,
        num_blocks=1,
        block_size=16,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )
    bf16_page = PagedKvPool.bytes_for(**shape, dtype=torch.bfloat16)
    fp8_page = PagedKvPool.bytes_for(**shape, dtype=torch.float8_e4m3fn)

    assert 2 * fp8_page == bf16_page, (
        f"a page costs {fp8_page} bytes in fp8 and {bf16_page} in bf16"
    )


@pytest.mark.cuda
def test_the_engine_sizes_more_pages_for_an_fp8_cache():
    """And the live sizing path agrees: cheaper pages mean more of them.

    Only a monotonic check, because the exact counts depend on how much of the card is
    free at this instant — see the test above for the exact claim.
    """
    from conftest import free_cuda_memory

    from mini_vllm import LLM
    from mini_vllm.model.loader import ModelConfig, resolve_model_path

    path = resolve_model_path()
    if not (path / "config.json").is_file():
        pytest.skip("Qwen3-0.6B config is not downloaded")

    free_cuda_memory()
    config = ModelConfig.from_pretrained(path)
    device = torch.device("cuda")
    bf16_blocks = LLM.blocks_that_fit(config, 16, device, 0.5, torch.bfloat16)
    fp8_blocks = LLM.blocks_that_fit(config, 16, device, 0.5, torch.float8_e4m3fn)
    assert fp8_blocks >= bf16_blocks


@pytest.mark.oracle
def test_fp8_greedy_tracks_bf16_before_it_drifts():
    """FP8 is lossy, so greedy eventually diverges — but it tracks bf16 first.

    Greedy divergence *compounds*: the step after the first disagreement runs on a
    different context, so one early flip cascades and the tail agreement is noise. That
    makes a whole-trajectory token count the wrong measure, and the matching prefix the
    right one. Both engines are pinned to the oracle attention path so the one variable
    is quantization rather than kernel-versus-oracle arithmetic, and they are built one
    at a time — two sets of weights on an 8 GB card is how this used to abort for memory
    instead of failing an assertion.
    """
    from conftest import real_engine

    from mini_vllm.sampler import SamplingParams

    prompt = "The capital of France is"
    greedy = SamplingParams(temperature=0.0)
    common = dict(dtype=torch.bfloat16, use_cuda_kernels=False)

    with real_engine(kv_cache_dtype="auto", **common) as llm:
        bf16 = llm.generate(prompt, sampling_params=greedy, max_tokens=16)[0]
    with real_engine(kv_cache_dtype="fp8", **common) as llm:
        fp8 = llm.generate(prompt, sampling_params=greedy, max_tokens=16)[0]

    matching_prefix = 0
    for a, b in zip(bf16.token_ids, fp8.token_ids):
        if a != b:
            break
        matching_prefix += 1
    assert matching_prefix >= 3, (
        f"fp8 diverged after only {matching_prefix} tokens: "
        f"bf16={bf16.token_ids[:6]} fp8={fp8.token_ids[:6]}"
    )
