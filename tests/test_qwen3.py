"""Step 1.8 — the full model against HuggingFace.

Structured so a failure localizes itself. The tiny-model tests walk outwards from
the embedding, through each block, to the logits, so the first failing test names
the layer that broke rather than reporting "the logits are wrong". That is the
forward-hook debugging advice from the plan, written down as tests instead of
something to remember to do under pressure.
"""

from __future__ import annotations

import pytest
import torch
from conftest import (
    BF16_DRIFT_LIMIT,
    assert_allclose,
    assert_relative_error_below,
    assert_tokens_equal,
    config_from_hf,
    qwen3_from_hf,
    relative_error,
    weights_from_hf,
)

from mini_vllm.layer_norm import rms_norm
from mini_vllm.model.qwen3 import Qwen3

PROMPT_SHAPES = [(1, 1), (1, 8), (2, 8), (3, 17)]


@pytest.fixture
def pair(tiny_qwen3):
    """Our model and HF's, sharing one set of random weights."""
    return qwen3_from_hf(tiny_qwen3), tiny_qwen3


# ------------------------------------------------------------------- structure


def test_builds_one_block_per_layer(pair):
    ours, theirs = pair
    assert len(ours.blocks) == theirs.config.num_hidden_layers


def test_layer_weights_are_split_correctly(pair):
    """Block `i` must hold layer `i`'s weights, not layer 0's twice.

    Compared by storage address rather than object identity: `state_dict()` hands
    out a fresh tensor object each call, sharing the same underlying buffer.
    """
    ours, theirs = pair
    weights = weights_from_hf(theirs)

    for index, block in enumerate(ours.blocks):
        assert block.attention.wq.data_ptr() == weights[f"layers.{index}.attn.wq"].data_ptr()
        assert block.mlp.gate.data_ptr() == weights[f"layers.{index}.mlp.gate"].data_ptr()


def test_lm_head_is_tied_to_the_embedding(pair):
    ours, _theirs = pair
    assert ours.embedding.weight is ours.weights["embedding"]


# ------------------------------------------------- component by component


def test_embedding_matches_hf(pair):
    ours, theirs = pair
    ids = torch.randint(0, theirs.config.vocab_size, (2, 6))
    assert_allclose(ours.embedding(ids), theirs.model.embed_tokens(ids))


def test_each_block_matches_hf(pair):
    """Layer-by-layer, so a mismatch names its layer.

    HF's `hidden_states[i]` is the *input* to block `i`, and the final entry is
    the hidden state after the final norm rather than after the last block — so
    the last comparison has to include the norm.
    """
    ours, theirs = pair
    ids = torch.randint(0, theirs.config.vocab_size, (2, 7))
    positions = torch.arange(7)

    with torch.no_grad():
        reference = theirs(ids, output_hidden_states=True).hidden_states

    assert len(reference) == len(ours.blocks) + 1

    h = ours.embedding(ids)
    assert_allclose(h, reference[0], msg="embedding output")

    for index, block in enumerate(ours.blocks):
        h = block(h, positions)
        if index + 1 < len(ours.blocks):
            assert_allclose(h, reference[index + 1], msg=f"after block {index}")

    normed = rms_norm(h, ours.final_norm, ours.config.rms_norm_eps)
    assert_allclose(normed, reference[-1], msg="after the final norm")


def test_final_norm_matches_hf(pair):
    ours, theirs = pair
    h = torch.randn(2, 5, theirs.config.hidden_size)

    ours_out = rms_norm(h, ours.final_norm, ours.config.rms_norm_eps)
    with torch.no_grad():
        theirs_out = theirs.model.norm(h)

    assert_allclose(ours_out, theirs_out)


# ------------------------------------------------------------ full forward


@pytest.mark.parametrize(("batch", "length"), PROMPT_SHAPES)
def test_logits_match_hf(pair, batch, length):
    ours, theirs = pair
    ids = torch.randint(0, theirs.config.vocab_size, (batch, length))

    with torch.no_grad():
        expected = theirs(ids).logits

    assert_allclose(ours(ids), expected, msg=f"batch={batch} length={length}")


def test_greedy_tokens_match_hf_on_the_tiny_model(pair):
    """Argmax agreement, which is stricter than closeness in the way that matters."""
    ours, theirs = pair
    ids = torch.randint(0, theirs.config.vocab_size, (2, 9))

    with torch.no_grad():
        expected = theirs(ids).logits

    assert_tokens_equal(ours(ids).argmax(dim=-1), expected.argmax(dim=-1))


def test_explicit_positions_match_the_default(pair):
    ours, theirs = pair
    ids = torch.randint(0, theirs.config.vocab_size, (1, 6))
    assert_allclose(ours(ids, positions=torch.arange(6)), ours(ids))


def test_shifted_positions_change_the_output(pair):
    """Positions must actually be used, not silently ignored."""
    ours, theirs = pair
    ids = torch.randint(0, theirs.config.vocab_size, (1, 6))

    default = ours(ids)
    shifted = ours(ids, positions=torch.arange(6) + 100)

    assert not torch.allclose(default, shifted)


def test_causal_masking_means_later_tokens_cannot_change_earlier_logits(pair):
    """A one-line check that the mask is doing its job.

    Change the last token and every earlier position's logits must be untouched.
    Without a causal mask, all of them would move.
    """
    ours, theirs = pair
    ids = torch.randint(0, theirs.config.vocab_size, (1, 8))

    original = ours(ids)
    modified_ids = ids.clone()
    modified_ids[0, -1] = (modified_ids[0, -1] + 1) % theirs.config.vocab_size
    modified = ours(modified_ids)

    assert_allclose(modified[:, :-1], original[:, :-1])


def test_batching_does_not_change_per_sequence_results(pair):
    """Row `i` of a batched forward equals that row run alone."""
    ours, theirs = pair
    ids = torch.randint(0, theirs.config.vocab_size, (3, 7))

    batched = ours(ids)
    for row in range(3):
        assert_allclose(batched[row : row + 1], ours(ids[row : row + 1]), msg=f"row {row}")


def test_output_shape_and_dtype(pair):
    ours, theirs = pair
    ids = torch.randint(0, theirs.config.vocab_size, (2, 5))
    logits = ours(ids)

    assert logits.shape == (2, 5, theirs.config.vocab_size)
    assert logits.dtype == torch.float32


@pytest.mark.cuda
def test_runs_on_the_gpu(tiny_qwen3, device):
    theirs = tiny_qwen3.to(device)
    ours = qwen3_from_hf(theirs)
    ids = torch.randint(0, theirs.config.vocab_size, (2, 6), device=device)

    with torch.no_grad():
        expected = theirs(ids).logits

    assert_allclose(ours(ids), expected)


# ------------------------------------------------------- the real weights


@pytest.fixture(scope="module")
def real_models():
    """Loads Qwen3-0.6B at a requested dtype, at most once per dtype.

    Two dtypes because they prove different things, and neither alone is enough:

    * **fp32** isolates *arithmetic*. With rounding out of the way our logits and
      HF's agree to ~1e-6 relative, so a tight assertion there is a real proof
      that the forward pass is the same function.
    * **bf16** is what the model actually runs in, so it is what has to work —
      but 28 layers of it drift a few percent from HF purely by rounding, and no
      elementwise tolerance can tell that apart from a bug.
    """
    from transformers import AutoModelForCausalLM

    from mini_vllm.model.loader import resolve_model_path

    path = resolve_model_path()
    if not (path / "model.safetensors").is_file():
        pytest.skip("Qwen3-0.6B weights are not downloaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache: dict[torch.dtype, tuple] = {}

    def load(dtype: torch.dtype):
        if dtype not in cache:
            theirs = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
            ours = Qwen3(config_from_hf(theirs), weights_from_hf(theirs))
            cache[dtype] = (ours, theirs, device)
        return cache[dtype]

    yield load

    cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


REAL_TEXT = (
    "The city of Rome was founded in 753 BC, according to legend, by the twin brothers "
    "Romulus and Remus. Over the following centuries it grew from a small settlement on "
    "the banks of the Tiber into the capital of an empire that stretched from Britain to Egypt."
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.fixture
def real_prompt_ids(tokenizer):
    return tokenizer("The capital of France is", return_tensors="pt").input_ids


@pytest.fixture
def real_text_ids(tokenizer):
    """57 tokens of ordinary prose — in-distribution input for the bf16 checks."""
    return tokenizer(REAL_TEXT, return_tensors="pt").input_ids


@pytest.mark.oracle
def test_real_model_config_is_the_design_table(real_models):
    ours, _theirs, _device = real_models(torch.bfloat16)

    assert ours.config.num_hidden_layers == 28
    assert ours.config.hidden_size == 1024
    assert ours.config.group_size == 2
    assert ours.config.q_projection_size == 2048
    assert len(ours.blocks) == 28


@pytest.mark.oracle
@pytest.mark.parametrize(("batch", "length"), [(1, 1), (1, 32), (3, 12)])
def test_real_logits_match_hf_in_fp32(real_models, batch, length):
    """The correctness proof for this step.

    In fp32 there is nowhere for a mistake to hide: our forward pass and HF's
    agree to roughly 1e-6 relative, and every greedy token is identical. Any real
    error — a missing QK-norm, a transposed projection, the wrong RoPE base —
    moves this by orders of magnitude.
    """
    ours, theirs, device = real_models(torch.float32)
    ids = torch.randint(0, ours.config.vocab_size, (batch, length), device=device)

    with torch.no_grad():
        expected = theirs(ids).logits
        got = ours(ids)

    assert_relative_error_below(got, expected, limit=1e-4, msg=f"{batch}x{length}")
    assert_tokens_equal(got.argmax(dim=-1), expected.argmax(dim=-1))


@pytest.mark.oracle
def test_real_bf16_logits_stay_within_the_rounding_floor(real_models, real_text_ids):
    """bf16, measured the only way that means anything: aggregate relative error.

    About 1.7% against HF on real prose, which is rounding rather than error — see
    `assert_relative_error_below` for why an elementwise bound cannot express this,
    and the test below for why 5% is not an arbitrary number.
    """
    ours, theirs, device = real_models(torch.bfloat16)
    ids = real_text_ids.to(device)

    with torch.no_grad():
        expected = theirs(ids).logits
        got = ours(ids)

    assert_relative_error_below(got, expected)


@pytest.mark.oracle
def test_the_bf16_drift_limit_still_catches_a_broken_model(real_models, real_text_ids):
    """A negative control, so the limit above cannot silently stop discriminating.

    `rope_theta = 10000` is Qwen2's base rather than Qwen3's, and it is the
    subtlest of the plausible mistakes here: every shape is right, every weight is
    loaded, nothing raises, and the model still produces fluent text. On this input
    it lands at 14% relative error against the 1.7% the correct model achieves, so
    the 5% ceiling sits roughly 3x from each side.
    """
    import dataclasses

    ours, theirs, device = real_models(torch.bfloat16)
    ids = real_text_ids.to(device)

    broken = Qwen3(dataclasses.replace(ours.config, rope_theta=10_000.0), ours.weights)

    with torch.no_grad():
        expected = theirs(ids).logits
        healthy = relative_error(ours(ids), expected)
        drifted = relative_error(broken(ids), expected)

    assert healthy < BF16_DRIFT_LIMIT
    assert drifted > BF16_DRIFT_LIMIT * 2, (
        f"a wrong rope_theta only moved relative error to {drifted:.4f} "
        f"(correct model: {healthy:.4f}), so the {BF16_DRIFT_LIMIT} limit is no "
        "longer discriminating"
    )


@pytest.mark.oracle
def test_real_greedy_argmax_matches_hf_for_32_tokens(real_models, real_prompt_ids):
    """The plan's acceptance criterion for this step, and the sharpest test here.

    Greedy decoding by re-running the whole prefix each step (no cache yet — that
    is Step 2.1). Token-identical output over 32 steps in bf16 is a much stronger
    statement than any logits tolerance: the comparison is exact, and a single
    wrong argmax sends the two sequences apart permanently.
    """
    ours, theirs, device = real_models(torch.bfloat16)
    ids = real_prompt_ids.to(device)

    ours_ids, theirs_ids = ids.clone(), ids.clone()
    for step in range(32):
        with torch.no_grad():
            next_ours = ours(ours_ids)[:, -1, :].argmax(dim=-1, keepdim=True)
            next_theirs = theirs(theirs_ids).logits[:, -1, :].argmax(dim=-1, keepdim=True)

        assert_tokens_equal(next_ours, next_theirs, msg=f"diverged at generation step {step}")
        ours_ids = torch.cat([ours_ids, next_ours], dim=1)
        theirs_ids = torch.cat([theirs_ids, next_theirs], dim=1)

    assert_tokens_equal(ours_ids, theirs_ids)
