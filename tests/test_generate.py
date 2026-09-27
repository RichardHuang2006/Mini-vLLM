"""generate.py: the two loops agree, and greedy decoding matches mlx_lm on the real model."""

import mlx.core as mx
import pytest
from mlx_lm.generate import generate_step

from mini_vllm.generate import generate_with_kv_cache, simple_generate
from mini_vllm.kv_cache import KvFullCache
from mini_vllm.models import from_mlx

# Logits for Qwen3's top tokens sit in [16, 32), where one bf16 step is 0.125: a greedy choice
# decided by less than two steps is a tie that rounding alone can flip.
TIE_MARGIN = 0.25

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Paged attention stores the KV cache in",
    "Once upon a time, in a small village,",
    "The three laws of thermodynamics are",
    "SELECT name, COUNT(*) FROM users",
    "To make a cup of tea, first",
    "import numpy as np\nx = np.",
    "The mitochondria is",
    "Translate to French: I would like a coffee.",
    "A haiku about autumn:",
    "The derivative of x^2 is",
    "Continuous batching improves throughput because",
    "Q: What is 17 * 23?\nA:",
    "The quick brown fox",
    "In 1969, Apollo 11",
]


@pytest.mark.parametrize("prefill_chunk", [None, 1, 3])
def test_the_generation_loops_agree(tiny_qwen3, prefill_chunk):
    """The quadratic loop and the cached loop, token-identical."""
    model = from_mlx(tiny_qwen3)
    prompt = mx.random.randint(0, 512, (7,)).tolist()

    assert generate_with_kv_cache(model, prompt, 12, prefill_chunk=prefill_chunk) == simple_generate(
        model, prompt, 12
    )


def test_generation_stops_at_an_eos_token(tiny_qwen3):
    model = from_mlx(tiny_qwen3)
    prompt = mx.random.randint(0, 512, (7,)).tolist()
    free = generate_with_kv_cache(model, prompt, 12)

    stop = free[3]
    expected = free[: free.index(stop) + 1]
    assert generate_with_kv_cache(model, prompt, 12, eos_token_ids={stop}) == expected
    assert simple_generate(model, prompt, 12, eos_token_ids={stop}) == expected


def test_greedy_choices_match_mlx_lm_away_from_bf16_ties(real_qwen3):
    """Teacher-forced along mlx_lm's greedy output, so both models see the same context at every
    step: wherever mlx_lm's top token wins by more than a bf16 tie, ours must pick it too."""
    reference, tokenizer = real_qwen3
    model = from_mlx(reference)
    decided = 0

    for text in PROMPTS:
        prompt = tokenizer.encode(text)
        continuation = [token for token, _ in generate_step(mx.array(prompt), reference, max_tokens=32)]

        # Our cached loop, fed mlx_lm's tokens instead of its own.
        caches = [KvFullCache() for _ in model.layers]
        logits = model(mx.array([prompt]), mx.arange(len(prompt)), caches)
        ours = [mx.argmax(logits[0, -1]).item()]
        for token in continuation[:-1]:
            logits = model(mx.array([[token]]), mx.array([caches[0].offset]), caches)
            ours.append(mx.argmax(logits[0, -1]).item())

        # mlx_lm's own choice and margin at each of those steps, from one full forward pass.
        theirs = reference(mx.array([prompt + continuation]))[0, len(prompt) - 1 : -1].astype(mx.float32)
        top_two = mx.sort(theirs, axis=-1)[:, -2:]
        margins = (top_two[:, 1] - top_two[:, 0]).tolist()
        choices = mx.argmax(theirs, axis=-1).tolist()

        for step, margin in enumerate(margins):
            if margin > TIE_MARGIN:
                decided += 1
                assert ours[step] == choices[step], f"{text!r} step {step}: margin {margin:.3f}"

    # The tie exclusion must leave most steps decided, or this test would check nothing.
    assert decided > 0.8 * len(PROMPTS) * 32, decided
