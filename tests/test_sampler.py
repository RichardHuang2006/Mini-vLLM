"""sampler.py: greedy, temperature, top-k and top-p truncation, per-row params, the draw."""

import math

import mlx.core as mx
import pytest
from utils import assert_allclose

from mini_vllm.sampler import SamplingParams, sample, sampling_probabilities

# Probabilities 0.5, 0.3, 0.15, 0.05, stored out of rank order to exercise the sort.
PROBABILITIES = [0.15, 0.5, 0.05, 0.3]
LOGITS = mx.array([[math.log(p) for p in PROBABILITIES]])


def test_greedy_rows_are_one_hot_and_ignore_the_key():
    logits = mx.random.normal((4, 100))
    greedy = SamplingParams(temperature=0.0)

    probabilities = sampling_probabilities(logits, greedy)
    assert mx.array_equal(mx.argmax(probabilities, axis=-1), mx.argmax(logits, axis=-1)).item()
    assert mx.array_equal(mx.sum(probabilities, axis=-1), mx.ones((4,))).item()

    for seed in range(3):
        tokens = sample(logits, greedy, key=mx.random.key(seed))
        assert tokens.dtype == mx.int32
        assert mx.array_equal(tokens, mx.argmax(logits, axis=-1)).item()


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_untruncated_rows_are_softmax_over_temperature(temperature):
    logits = 3 * mx.random.normal((3, 50))
    probabilities = sampling_probabilities(logits, SamplingParams(temperature=temperature))
    assert_allclose(probabilities, mx.softmax(logits / temperature, axis=-1, precise=True))


@pytest.mark.parametrize("top_k", [1, 5, 17])
def test_top_k_keeps_exactly_k_tokens(top_k):
    logits = mx.random.normal((3, 50))
    probabilities = sampling_probabilities(logits, SamplingParams(top_k=top_k))

    kept = probabilities > 0
    assert mx.sum(kept, axis=-1).tolist() == [top_k] * 3
    # The kept tokens are the k largest logits.
    threshold = mx.sort(logits, axis=-1)[:, -top_k][:, None]
    assert mx.array_equal(kept, logits >= threshold).item()


@pytest.mark.parametrize(
    ("top_p", "expected"),
    [
        (0.75, [0.0, 0.5 / 0.8, 0.0, 0.3 / 0.8]),            # 0.5 < p, 0.5 + 0.3 reaches it
        (0.85, [0.15 / 0.95, 0.5 / 0.95, 0.0, 0.3 / 0.95]),  # the third token crosses p
        (0.01, [0.0, 1.0, 0.0, 0.0]),                        # the top token is always kept
    ],
)
def test_top_p_keeps_the_smallest_prefix_reaching_p(top_p, expected):
    probabilities = sampling_probabilities(LOGITS, SamplingParams(top_p=top_p))
    assert_allclose(probabilities, mx.array([expected]))


def test_per_row_parameters_apply_per_row():
    logits = mx.random.normal((3, 40))
    params = [SamplingParams(temperature=0.0), SamplingParams(top_k=3), SamplingParams(top_p=0.5)]

    together = sampling_probabilities(logits, params)
    for row, row_params in enumerate(params):
        assert_allclose(together[row : row + 1], sampling_probabilities(logits[row : row + 1], row_params))


def test_a_seeded_key_reproduces_the_draw():
    logits = mx.random.normal((64, 100))
    key = mx.random.key(7)
    assert mx.array_equal(sample(logits, key=key), sample(logits, key=key)).item()


def test_draws_follow_the_truncated_distribution():
    draws = 20_000
    params = SamplingParams(top_k=3)
    tokens = sample(mx.broadcast_to(LOGITS, (draws, 4)), params, key=mx.random.key(0))

    frequencies = [mx.sum(tokens == token).item() / draws for token in range(4)]
    expected = sampling_probabilities(LOGITS, params)[0].tolist()
    # Three standard errors of a 20k-draw frequency is under 0.011 for any p.
    assert all(abs(f - p) < 0.011 for f, p in zip(frequencies, expected, strict=True)), frequencies
