"""Step 1.9 — greedy generation against `transformers.generate`.

The acceptance criterion is token identity, not similarity: greedy decoding is
deterministic, so agreeing with HF for 32 steps means every logit that mattered
agreed on its argmax. When it does fail, it fails by diverging at one step and
then never recovering, which `assert_tokens_equal` reports by index.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import assert_tokens_equal, qwen3_from_hf

from mini_vllm.generate import (
    eos_token_ids_for,
    generate,
    generate_ids,
    load,
)
from mini_vllm.model.loader import resolve_model_path

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
]

# Diverges from HF at generation step 14 in bf16, and does so legitimately: both
# models produce the same top two logits (18.0 and 17.875) and disagree only about
# their order. See `test_a_bf16_divergence_is_only_ever_a_near_tie`, which asserts
# that rather than taking it on trust.
NEAR_TIE_PROMPT = "Once upon a time"


def bf16_ulp(value: float) -> float:
    """The gap between adjacent bf16 numbers at ``value``.

    bf16 keeps 8 mantissa bits, so representable values near 18 are 0.125 apart.
    Two logits that close are not *nearly* tied, they are tied: no bf16
    computation can distinguish them, and which one wins an argmax is decided by
    rounding rather than by the model.
    """
    return 2.0 ** (math.floor(math.log2(abs(value))) - 7)


# ------------------------------------------------------- loop, no real weights


def test_greedy_loop_appends_max_tokens(tiny_qwen3):
    ours = qwen3_from_hf(tiny_qwen3)
    ids = torch.randint(0, tiny_qwen3.config.vocab_size, (1, 5))

    tokens = generate_ids(ours, ids, max_tokens=7)

    assert tokens.shape == (1, 12)
    assert_tokens_equal(tokens[:, :5], ids, msg="prompt must be preserved verbatim")


def test_greedy_loop_is_deterministic(tiny_qwen3):
    ours = qwen3_from_hf(tiny_qwen3)
    ids = torch.randint(0, tiny_qwen3.config.vocab_size, (1, 4))

    assert_tokens_equal(generate_ids(ours, ids, 6), generate_ids(ours, ids, 6))


def test_greedy_loop_matches_a_hand_rolled_argmax(tiny_qwen3):
    """Pins the exact contract: argmax of the *last* position, appended."""
    ours = qwen3_from_hf(tiny_qwen3)
    ids = torch.randint(0, tiny_qwen3.config.vocab_size, (1, 4))

    expected = ids
    for _ in range(5):
        with torch.no_grad():
            next_token = ours(expected)[:, -1, :].argmax(dim=-1, keepdim=True)
        expected = torch.cat([expected, next_token], dim=1)

    assert_tokens_equal(generate_ids(ours, ids, 5), expected)


def test_stops_at_eos(tiny_qwen3):
    """Feeding the token it is about to generate as the stop token ends the loop."""
    ours = qwen3_from_hf(tiny_qwen3)
    ids = torch.randint(0, tiny_qwen3.config.vocab_size, (1, 4))

    unstopped = generate_ids(ours, ids, max_tokens=8)
    first_generated = int(unstopped[0, 4])

    stopped = generate_ids(ours, ids, max_tokens=8, eos_token_ids=[first_generated])

    assert stopped.shape[1] == 5, "should have stopped immediately after the stop token"
    assert int(stopped[0, -1]) == first_generated


def test_batched_rows_are_padded_after_they_finish(tiny_qwen3):
    """A finished row must stop influencing the output, and stay rectangular."""
    ours = qwen3_from_hf(tiny_qwen3)
    ids = torch.randint(0, tiny_qwen3.config.vocab_size, (2, 4))

    unstopped = generate_ids(ours, ids, max_tokens=6)
    # Stop only whatever row 0 produces first; row 1 should keep going.
    stop = int(unstopped[0, 4])
    tokens = generate_ids(ours, ids, max_tokens=6, eos_token_ids=[stop], pad_token_id=0)

    row0 = tokens[0].tolist()
    assert row0[4] == stop
    assert set(row0[5:]) <= {0}, "a finished row must be padded, not kept generating"


def test_batched_generation_matches_row_by_row(tiny_qwen3):
    ours = qwen3_from_hf(tiny_qwen3)
    ids = torch.randint(0, tiny_qwen3.config.vocab_size, (3, 5))

    batched = generate_ids(ours, ids, max_tokens=4)
    for row in range(3):
        single = generate_ids(ours, ids[row : row + 1], max_tokens=4)
        assert_tokens_equal(batched[row : row + 1], single, msg=f"row {row}")


def test_rejects_unbatched_input(tiny_qwen3):
    ours = qwen3_from_hf(tiny_qwen3)
    with pytest.raises(ValueError, match="expected B x L"):
        generate_ids(ours, torch.tensor([1, 2, 3]), max_tokens=2)


# -------------------------------------------------------------- stop tokens


@pytest.mark.oracle
def test_reads_both_of_qwens_stop_tokens():
    """Qwen3 has two, and missing one generates past the end of a turn."""
    ids = eos_token_ids_for(resolve_model_path())
    assert ids == (151645, 151643)


# --------------------------------------------------- against transformers


@pytest.fixture(scope="module")
def real():
    """Ours and HF's, sharing bf16 weights, plus the tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = resolve_model_path()
    if not (path / "model.safetensors").is_file():
        pytest.skip("Qwen3-0.6B weights are not downloaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    theirs = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
    ours = qwen3_from_hf(theirs)
    tokenizer = AutoTokenizer.from_pretrained(path)

    yield ours, theirs, tokenizer, device

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def hf_greedy(theirs, ids, max_tokens=32):
    return theirs.generate(
        ids,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=theirs.generation_config.pad_token_id,
    )


def our_greedy(ours, ids, max_tokens=32, pad_token_id=151643):
    return generate_ids(
        ours,
        ids,
        max_tokens=max_tokens,
        eos_token_ids=eos_token_ids_for(resolve_model_path()),
        pad_token_id=pad_token_id,
    )


@pytest.mark.oracle
@pytest.mark.parametrize("prompt", PROMPTS)
def test_matches_transformers_generate(real, prompt):
    """The milestone check: token-identical to `generate(do_sample=False)`, 32 steps."""
    ours, theirs, tokenizer, device = real
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    assert_tokens_equal(our_greedy(ours, ids), hf_greedy(theirs, ids), msg=f"prompt: {prompt!r}")


@pytest.mark.oracle
def test_a_bf16_divergence_is_only_ever_a_near_tie(real):
    """When bf16 greedy decoding *does* disagree with HF, prove it is representational.

    Token identity is the right criterion for greedy decoding, but it has one
    honest limit in bf16: when the top two logits land on adjacent representable
    values, their order is decided by rounding, and both answers are equally
    correct. Rather than quietly dropping such a prompt, this pins down what a
    legitimate disagreement has to look like — the same two candidates, no more
    than a couple of ULP apart. An actual bug satisfies neither condition: it
    picks a token the other model does not rank highly at all.
    """
    ours, theirs, tokenizer, device = real
    ids = tokenizer(NEAR_TIE_PROMPT, return_tensors="pt").input_ids.to(device)

    got, expected = our_greedy(ours, ids), hf_greedy(theirs, ids)
    if got.tolist() == expected.tolist():
        pytest.skip("no divergence to characterize on this build")

    index = next(i for i in range(min(got.shape[1], expected.shape[1])) if got[0, i] != expected[0, i])

    # Both models agreed on everything up to `index`, so feed that shared prefix
    # and compare the two distributions that actually disagreed.
    prefix = expected[:, :index]
    with torch.no_grad():
        our_logits = ours(prefix)[0, -1].float()
        their_logits = theirs(prefix).logits[0, -1].float()

    our_top = our_logits.topk(2)
    their_top = their_logits.topk(2)

    assert set(our_top.indices.tolist()) == set(their_top.indices.tolist()), (
        "a divergence with different candidates is a bug, not a tie: "
        f"ours {our_top.indices.tolist()} vs theirs {their_top.indices.tolist()}"
    )

    for name, top in (("ours", our_top), ("theirs", their_top)):
        gap = (top.values[0] - top.values[1]).item()
        limit = 2 * bf16_ulp(top.values[0].item())
        assert gap <= limit, (
            f"{name} preferred its token by {gap:.4f}, more than {limit:.4f} "
            "(two bf16 ULP), so this divergence is not explained by rounding"
        )


@pytest.mark.oracle
def test_generated_text_is_real_language(real):
    """A sanity check a human can read, and the milestone in one line."""
    ours, _theirs, tokenizer, _device = real
    from mini_vllm.generate import Loaded

    loaded = Loaded(ours, tokenizer, eos_token_ids_for(resolve_model_path()), 151643)
    text = generate(loaded, "The capital of France is", max_tokens=8)

    assert isinstance(text, str) and text.strip()
    assert "Paris" in text, f"expected the model to know this one, got {text!r}"


@pytest.mark.oracle
def test_load_returns_a_usable_bundle():
    """`load()` is what the CLI uses, so it is worth one end-to-end check."""
    loaded = load(device="cuda" if torch.cuda.is_available() else "cpu")

    assert loaded.eos_token_ids == (151645, 151643)
    assert loaded.pad_token_id == 151643
    assert loaded.model.config.num_hidden_layers == 28

    text = generate(loaded, "The capital of France is", max_tokens=6)
    assert "Paris" in text
