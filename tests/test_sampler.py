"""Step 1.10 — sampling: greedy, temperature, top-k, top-p.

Sampling is awkward to test because the output is random, so the tests here lean
on three things that are not: the distribution `sampling_probabilities` returns,
exact reproducibility under a seeded generator, and — for the one genuinely
statistical claim — enough draws that Monte-Carlo error is smaller than the effect
being measured.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import assert_allclose

from mini_vllm.sampler import SamplingParams, sample, sampling_probabilities

VOCAB = 64


@pytest.fixture
def logits():
    return torch.randn(4, VOCAB)


def generator(seed: int = 0) -> torch.Generator:
    engine = torch.Generator()
    engine.manual_seed(seed)
    return engine


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"temperature": -0.1}, "temperature must be >= 0"),
        ({"top_k": -1}, "top_k must be >= 0"),
        ({"top_p": 0.0}, r"top_p must be in \(0, 1\]"),
        ({"top_p": 1.5}, r"top_p must be in \(0, 1\]"),
    ],
)
def test_rejects_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SamplingParams(**kwargs)


def test_defaults_are_plain_temperature_one_sampling():
    params = SamplingParams()
    assert (params.temperature, params.top_k, params.top_p) == (1.0, 0, 1.0)
    assert not params.is_greedy


def test_rejects_unbatched_logits():
    with pytest.raises(ValueError, match="expected B x V"):
        sampling_probabilities(torch.randn(VOCAB), SamplingParams())


def test_rejects_a_param_list_of_the_wrong_length(logits):
    with pytest.raises(ValueError, match="for a batch of 4"):
        sampling_probabilities(logits, [SamplingParams()] * 3)


# ---------------------------------------------------------------------- greedy


def test_greedy_is_the_argmax(logits):
    tokens = sample(logits, SamplingParams(temperature=0.0))
    assert torch.equal(tokens, logits.argmax(dim=-1))


def test_greedy_is_deterministic(logits):
    greedy = SamplingParams(temperature=0.0)
    first = sample(logits, greedy, generator=generator(1))
    second = sample(logits, greedy, generator=generator(99))
    assert torch.equal(first, second)


def test_greedy_distribution_is_one_hot(logits):
    probabilities = sampling_probabilities(logits, SamplingParams(temperature=0.0))

    assert_allclose(probabilities.sum(dim=-1), torch.ones(4))
    assert torch.equal(probabilities.argmax(dim=-1), logits.argmax(dim=-1))
    assert (probabilities.max(dim=-1).values == 1.0).all()


def test_greedy_survives_extreme_logits():
    """Temperature 0 must not divide by zero, however large the logits."""
    logits = torch.tensor([[1000.0, 999.0, -1000.0]])
    probabilities = sampling_probabilities(logits, SamplingParams(temperature=0.0))

    assert torch.isfinite(probabilities).all()
    assert int(sample(logits, SamplingParams(temperature=0.0))) == 0


# ----------------------------------------------------------------- temperature


def test_temperature_one_is_plain_softmax(logits):
    assert_allclose(
        sampling_probabilities(logits, SamplingParams()),
        logits.softmax(dim=-1),
    )


def test_low_temperature_sharpens_and_high_flattens(logits):
    """Entropy is the honest summary: cold is more certain, hot is less."""

    def entropy(temperature: float) -> torch.Tensor:
        probabilities = sampling_probabilities(logits, SamplingParams(temperature=temperature))
        return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)

    assert (entropy(0.5) < entropy(1.0)).all()
    assert (entropy(2.0) > entropy(1.0)).all()


def test_temperature_scales_logits_before_softmax(logits):
    assert_allclose(
        sampling_probabilities(logits, SamplingParams(temperature=0.25)),
        (logits / 0.25).softmax(dim=-1),
    )


def test_very_low_temperature_approaches_greedy(logits):
    probabilities = sampling_probabilities(logits, SamplingParams(temperature=1e-3))
    assert torch.equal(probabilities.argmax(dim=-1), logits.argmax(dim=-1))
    assert (probabilities.max(dim=-1).values > 0.99).all()


def test_a_fixed_seed_reproduces_exactly(logits):
    first = sample(logits, SamplingParams(temperature=1.0), generator=generator(7))
    second = sample(logits, SamplingParams(temperature=1.0), generator=generator(7))
    assert torch.equal(first, second)


def test_different_seeds_eventually_differ():
    logits = torch.randn(32, VOCAB)
    first = sample(logits, SamplingParams(temperature=1.0), generator=generator(1))
    second = sample(logits, SamplingParams(temperature=1.0), generator=generator(2))
    assert not torch.equal(first, second)


# ----------------------------------------------------------------------- top-k


@pytest.mark.parametrize("k", [1, 2, 5, 40])
def test_top_k_keeps_exactly_k_tokens(logits, k):
    probabilities = sampling_probabilities(logits, SamplingParams(top_k=k))
    assert ((probabilities > 0).sum(dim=-1) == k).all()


def test_top_k_keeps_the_k_largest_logits(logits):
    k = 5
    probabilities = sampling_probabilities(logits, SamplingParams(top_k=k))

    kept = (probabilities > 0)
    expected = torch.zeros_like(kept).scatter_(1, logits.topk(k, dim=-1).indices, True)
    assert torch.equal(kept, expected)


def test_top_k_one_is_greedy_in_distribution(logits):
    """Not the same code path as `temperature=0`, but it must agree with it."""
    assert_allclose(
        sampling_probabilities(logits, SamplingParams(top_k=1)),
        sampling_probabilities(logits, SamplingParams(temperature=0.0)),
    )


def test_top_k_renormalizes(logits):
    probabilities = sampling_probabilities(logits, SamplingParams(top_k=3))
    assert_allclose(probabilities.sum(dim=-1), torch.ones(4))


def test_top_k_preserves_relative_odds_within_the_kept_set(logits):
    """Truncation must only remove mass, never redistribute it unevenly."""
    full = sampling_probabilities(logits, SamplingParams())
    truncated = sampling_probabilities(logits, SamplingParams(top_k=4))

    kept = truncated > 0
    ratio = (truncated[kept] / full[kept]).reshape(4, 4)
    # Every kept token in a row is scaled by the same constant.
    assert_allclose(ratio, ratio[:, :1].expand_as(ratio))


def test_top_k_larger_than_the_vocabulary_is_a_no_op(logits):
    assert_allclose(
        sampling_probabilities(logits, SamplingParams(top_k=VOCAB * 10)),
        sampling_probabilities(logits, SamplingParams()),
    )


def test_top_k_never_samples_outside_the_kept_set(logits):
    k = 3
    allowed = logits.topk(k, dim=-1).indices

    for seed in range(50):
        tokens = sample(logits, SamplingParams(top_k=k), generator=generator(seed))
        assert (tokens.unsqueeze(1) == allowed).any(dim=1).all()


# ----------------------------------------------------------------------- top-p


def test_top_p_one_is_a_no_op(logits):
    assert_allclose(
        sampling_probabilities(logits, SamplingParams(top_p=1.0)),
        logits.softmax(dim=-1),
    )


def test_top_p_keeps_the_smallest_prefix_reaching_p(logits):
    """The nucleus is the shortest run of top tokens whose mass reaches p.

    Checked both ways: the kept set must reach p, and dropping its least likely
    member must fall short of it. Together those pin the set exactly.
    """
    p = 0.9
    full = logits.softmax(dim=-1)
    kept = sampling_probabilities(logits, SamplingParams(top_p=p)) > 0

    for row in range(logits.shape[0]):
        masses = full[row][kept[row]].sort(descending=True).values
        assert masses.sum() >= p, "the nucleus does not reach p"
        assert masses[:-1].sum() < p, "the nucleus is one token larger than it needs to be"


def test_top_p_never_samples_outside_the_nucleus(logits):
    """The plan's criterion, checked over many draws.

    `sampling_probabilities` gives zero weight outside the nucleus, so this holds
    by construction — the point is to confirm the draw honours the distribution
    rather than falling back on the untruncated logits.
    """
    p = 0.9
    nucleus = sampling_probabilities(logits, SamplingParams(top_p=p)) > 0

    for seed in range(100):
        tokens = sample(logits, SamplingParams(top_p=p), generator=generator(seed))
        assert nucleus[torch.arange(logits.shape[0]), tokens].all(), (
            f"seed {seed} sampled outside the {p} nucleus"
        )


def test_top_p_always_keeps_at_least_one_token():
    """A tiny p must not produce an empty, un-normalizable row."""
    logits = torch.randn(3, VOCAB)
    probabilities = sampling_probabilities(logits, SamplingParams(top_p=1e-6))

    assert ((probabilities > 0).sum(dim=-1) == 1).all()
    assert_allclose(probabilities.sum(dim=-1), torch.ones(3))
    assert torch.equal(probabilities.argmax(dim=-1), logits.argmax(dim=-1))


def test_top_p_includes_the_token_that_crosses_p():
    """A uniform distribution is the case that catches an off-by-one nucleus.

    With four equal quarters and `p = 0.5`, the mass before the second token is
    0.25 (< 0.5) and before the third is 0.5 (not < 0.5), so exactly two tokens
    are kept. A rule written as `cumsum <= p` would keep two here as well but drop
    to one for `p = 0.5 - eps`; this one keeps the token that crosses the
    threshold, which is what makes the nucleus reach p rather than stop below it.
    """
    logits = torch.zeros(1, 4)
    kept = sampling_probabilities(logits, SamplingParams(top_p=0.5)) > 0
    assert int(kept.sum()) == 2

    kept = sampling_probabilities(logits, SamplingParams(top_p=0.6)) > 0
    assert int(kept.sum()) == 3


def test_top_p_renormalizes(logits):
    probabilities = sampling_probabilities(logits, SamplingParams(top_p=0.5))
    assert_allclose(probabilities.sum(dim=-1), torch.ones(4))


# ------------------------------------------------------------------- combined


def test_top_k_and_top_p_together_take_the_smaller_set(logits):
    """Both filters apply, so the result is the intersection."""
    by_k = sampling_probabilities(logits, SamplingParams(top_k=3)) > 0
    by_p = sampling_probabilities(logits, SamplingParams(top_p=0.9)) > 0
    together = sampling_probabilities(logits, SamplingParams(top_k=3, top_p=0.9)) > 0

    assert torch.equal(together, by_k & by_p)


def test_temperature_is_applied_before_truncation():
    """Order matters: a hot temperature widens the nucleus.

    If top-p were computed on the unscaled logits, temperature could not change
    the size of the kept set at all — so this is what pins the ordering down.
    """
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])

    cold = (sampling_probabilities(logits, SamplingParams(temperature=0.5, top_p=0.9)) > 0).sum()
    hot = (sampling_probabilities(logits, SamplingParams(temperature=2.0, top_p=0.9)) > 0).sum()

    assert cold < hot


# --------------------------------------------------------- per-row parameters


def test_each_row_honours_its_own_parameters(logits):
    """The requirement that shapes this module: one batch, four different configs."""
    per_row = [
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.0),
        SamplingParams(top_k=2),
        SamplingParams(top_p=0.5),
    ]

    probabilities = sampling_probabilities(logits, per_row)

    for row, params in enumerate(per_row):
        alone = sampling_probabilities(logits[row : row + 1], params)
        assert_allclose(probabilities[row : row + 1], alone, msg=f"row {row} ({params})")


def test_a_greedy_row_beside_a_sampled_row_stays_greedy(logits):
    per_row = [SamplingParams(temperature=0.0), SamplingParams(temperature=2.0)] * 2

    for seed in range(20):
        tokens = sample(logits, per_row, generator=generator(seed))
        assert int(tokens[0]) == int(logits[0].argmax())
        assert int(tokens[2]) == int(logits[2].argmax())


def test_batch_of_one_and_list_of_one_agree(logits):
    single = logits[:1]
    assert_allclose(
        sampling_probabilities(single, SamplingParams(top_k=3)),
        sampling_probabilities(single, [SamplingParams(top_k=3)]),
    )


# ------------------------------------------------------------------- shapes


def test_sample_returns_one_token_per_row(logits):
    tokens = sample(logits, SamplingParams())
    assert tokens.shape == (4,)
    assert tokens.dtype == torch.int64
    assert (tokens >= 0).all() and (tokens < VOCAB).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_accepts_any_logit_dtype(dtype):
    logits = torch.randn(2, VOCAB, dtype=dtype)
    probabilities = sampling_probabilities(logits, SamplingParams(top_k=4))

    assert probabilities.dtype == torch.float32
    assert_allclose(probabilities.sum(dim=-1), torch.ones(2))


@pytest.mark.cuda
def test_matches_cpu_on_gpu(logits, device):
    params = SamplingParams(temperature=0.8, top_k=8, top_p=0.9)
    assert_allclose(
        sampling_probabilities(logits.to(device), params).cpu(),
        sampling_probabilities(logits, params),
    )


# ------------------------------------------------------------- distribution


@pytest.mark.slow
def test_empirical_frequencies_match_the_target_distribution():
    """100k draws against the analytic distribution, within Monte-Carlo error.

    The tolerance is not chosen, it is derived. Each token's count is binomial, so
    an observed frequency has standard error `sqrt(p(1-p)/n)`; at 4 sigma a correct
    sampler exceeds it about once in 16k tokens, which over 8 tokens is a
    negligible flake rate while still catching any real bias.
    """
    draws = 100_000
    logits = torch.tensor([[2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0]])
    expected = logits.softmax(dim=-1)[0]

    tokens = sample(
        logits.expand(draws, -1).contiguous(),
        SamplingParams(temperature=1.0),
        generator=generator(0),
    )
    observed = torch.bincount(tokens, minlength=logits.shape[1]).float() / draws

    for token in range(logits.shape[1]):
        p = expected[token].item()
        standard_error = math.sqrt(p * (1.0 - p) / draws)
        difference = abs(observed[token].item() - p)
        assert difference < 4 * standard_error, (
            f"token {token}: observed {observed[token]:.5f} vs expected {p:.5f}, "
            f"off by {difference / standard_error:.1f} sigma"
        )


@pytest.mark.slow
def test_temperature_shifts_the_empirical_distribution():
    """A hot sampler must visibly favour the tail more than a cold one."""
    draws = 50_000
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])

    def tail_fraction(temperature: float) -> float:
        tokens = sample(
            logits.expand(draws, -1).contiguous(),
            SamplingParams(temperature=temperature),
            generator=generator(0),
        )
        return (tokens >= 2).float().mean().item()

    assert tail_fraction(0.5) < tail_fraction(1.0) < tail_fraction(3.0)


@pytest.mark.slow
def test_truncated_distribution_is_sampled_faithfully():
    """Under top-k, the kept tokens must keep their *relative* frequencies."""
    draws = 100_000
    k = 3
    logits = torch.tensor([[2.0, 1.5, 1.0, 0.5, 0.0, -1.0]])
    expected = sampling_probabilities(logits, SamplingParams(top_k=k))[0]

    tokens = sample(
        logits.expand(draws, -1).contiguous(),
        SamplingParams(top_k=k),
        generator=generator(0),
    )
    observed = torch.bincount(tokens, minlength=logits.shape[1]).float() / draws

    assert int((observed > 0).sum()) == k
    for token in range(k):
        p = expected[token].item()
        standard_error = math.sqrt(p * (1.0 - p) / draws)
        assert abs(observed[token].item() - p) < 4 * standard_error
