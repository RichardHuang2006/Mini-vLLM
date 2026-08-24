"""Rejection sampling: the guarantee that speculation is exact rather than approximate.

The rest of speculative decoding is bookkeeping — propose, score, roll back. This is the
part supporting the distributional claim: the tokens a speculative run emits are
distributed exactly as the target model's own tokens would be, so the speedup costs
nothing in quality.

An example cannot demonstrate that, so the central test is statistical: draw a hundred
thousand tokens through the accept/reject procedure with a draft that disagrees with the
target, histogram them, and check against the target's distribution with a chi-square
test. A missing or wrong residual correction would lean the histogram toward the draft.

The rest pins the algorithm's properties: acceptance is certain when the models agree, a
rejection truncates rather than skipping, a step always yields at least one token, and the
greedy case is the prefix where the two argmaxes agree.
"""

from __future__ import annotations

import pytest
import torch

from mini_vllm.spec.rejection import (
    ResidualError,
    accept_proposals,
    residual_distribution,
)


def distribution(*weights: float) -> torch.Tensor:
    """A normalized 1-D distribution from unnormalized weights."""
    tensor = torch.tensor(weights, dtype=torch.float64)
    return tensor / tensor.sum()


def rows(*distributions: torch.Tensor) -> torch.Tensor:
    return torch.stack(list(distributions))


# ----------------------------------------------------------------- the residual


def test_the_residual_is_the_mass_acceptance_leaves_behind():
    """``relu(p - q)`` normalized, and its raw mass is exactly ``1 - sum(min(p, q))``.

    That identity is the whole reason the correction is this expression and not another:
    the acceptance step emits token t with probability min(p(t), q(t)), so the shortfall
    it leaves is precisely the mass of relu(p - q).
    """
    target = distribution(0.6, 0.3, 0.1)
    draft = distribution(0.1, 0.7, 0.2)

    residual = residual_distribution(target, draft)

    raw = (target - draft).clamp_min(0.0)
    shortfall = 1.0 - torch.minimum(target, draft).sum()
    assert torch.isclose(raw.sum(), shortfall)
    assert torch.isclose(residual.sum(), torch.tensor(1.0, dtype=torch.float64))
    # Only where the target wants more than the draft gave it.
    assert residual[0] > 0 and residual[1] == 0


def test_an_empty_residual_is_an_error_not_a_silent_zero():
    """p == q cannot be rejected, so being asked for its residual is a caller bug."""
    same = distribution(0.5, 0.5)
    with pytest.raises(ResidualError, match="no mass"):
        residual_distribution(same, same)


# ------------------------------------------------------- distribution preserved


@pytest.mark.parametrize("num_proposals", [1, 3])
def test_the_emitted_token_matches_the_target_distribution(num_proposals: int):
    """The headline claim, as a chi-square test over 120k sampled first tokens.

    The draft is deliberately wrong — it puts its mass where the target does not — so a
    procedure that simply trusted the draft, or that corrected rejections with the wrong
    distribution, would produce a visibly different histogram. Only the accept/reject rule
    with the `relu(p - q)` residual reproduces the target.

    The *first* emitted token is the one measured. It is the position every trial has in
    common, and it is where the correction acts; later positions are conditioned on
    different prefixes in a real model, so lumping them together would be measuring a
    mixture rather than a distribution.
    """
    vocabulary = 5
    target = distribution(0.40, 0.25, 0.20, 0.10, 0.05)
    draft = distribution(0.05, 0.10, 0.20, 0.25, 0.40)  # the target, reversed

    generator = torch.Generator().manual_seed(1234)
    trials = 120_000
    counts = torch.zeros(vocabulary, dtype=torch.float64)

    target_rows = rows(*[target] * (num_proposals + 1))
    draft_rows = rows(*[draft] * num_proposals)

    for _ in range(trials):
        # The draft proposes from its own distribution, as it does in the engine; sampling
        # the proposals otherwise would test a procedure the algorithm never runs.
        proposals = torch.multinomial(draft, num_proposals, replacement=True, generator=generator)
        tokens, _accepted = accept_proposals(
            target_rows, draft_rows, proposals, generator=generator
        )
        counts[tokens[0]] += 1

    expected = target * trials
    chi_square = float(((counts - expected) ** 2 / expected).sum())
    # 4 degrees of freedom: the 99.9th percentile of chi-square(4) is 18.47. A correct
    # implementation sits near 4; a missing residual correction lands in the hundreds.
    assert chi_square < 18.47, (
        f"chi-square {chi_square:.2f} rejects the target distribution; "
        f"got {(counts / trials).tolist()}, wanted {target.tolist()}"
    )


def test_a_matching_draft_is_always_accepted_and_never_perturbs_the_result():
    """When q == p the ratio is 1, so every proposal is kept and the bonus is drawn.

    This is the self-draft case, and it is the reason acceptance rate is a meaningful
    diagnostic: a draft that agrees with the target perfectly should waste nothing.
    """
    target = distribution(0.5, 0.3, 0.2)
    generator = torch.Generator().manual_seed(7)

    for _ in range(200):
        proposals = torch.multinomial(target, 3, replacement=True, generator=generator)
        tokens, accepted = accept_proposals(
            rows(target, target, target, target), rows(target, target, target), proposals,
            generator=generator,
        )
        assert accepted == 3, "an identical draft was rejected"
        assert tokens[:3] == [int(t) for t in proposals]
        assert len(tokens) == 4, "three accepted proposals plus the bonus"


# ------------------------------------------------------------------- mechanics


def test_a_step_always_emits_at_least_one_token():
    """Even total rejection yields the residual draw, so speculation cannot stall."""
    target = distribution(1.0, 0.0, 0.0)  # certain about token 0
    draft = distribution(0.0, 1.0, 0.0)  # certain about token 1, which target forbids

    tokens, accepted = accept_proposals(rows(target, target), rows(draft), [1])

    assert accepted == 0, "the target gives token 1 zero probability"
    assert tokens == [0], "and the residual can only be token 0"


def test_rejection_truncates_rather_than_skipping():
    """Proposals after a rejection are discarded, not reconsidered.

    They were generated conditioned on a token that is no longer in the sequence, so they
    say nothing about the sequence that now exists. Keeping them would be the subtlest
    possible way to corrupt the output.
    """
    certain_target = distribution(1.0, 0.0)
    # Position 0 is rejected (the draft proposes token 1, which the target forbids), so
    # the proposal at position 1 must never appear even though it would be acceptable.
    target_rows = rows(certain_target, certain_target, certain_target)
    draft_rows = rows(distribution(0.0, 1.0), certain_target)

    tokens, accepted = accept_proposals(target_rows, draft_rows, [1, 0])

    assert accepted == 0
    assert len(tokens) == 1, "a rejected position ends the step"


def test_greedy_keeps_exactly_the_prefix_where_the_models_agree():
    """Temperature zero: accept while the argmaxes match, then take the target's token."""
    # The target's argmax is 0 at every position; the draft gets the first two right.
    target_rows = rows(*[distribution(0.9, 0.1)] * 4)
    draft_rows = rows(*[distribution(0.9, 0.1)] * 3)

    tokens, accepted = accept_proposals(target_rows, draft_rows, [0, 0, 1], greedy=True)

    assert accepted == 2, "the first two proposals matched the target's argmax"
    assert tokens == [0, 0, 0], "the third is replaced by the target's own choice"


def test_greedy_full_acceptance_takes_the_bonus_argmax():
    target_rows = rows(distribution(0.1, 0.9), distribution(0.8, 0.2))
    draft_rows = rows(distribution(0.1, 0.9))

    tokens, accepted = accept_proposals(target_rows, draft_rows, [1], greedy=True)

    assert accepted == 1
    assert tokens == [1, 0], "the bonus is the target's argmax at the next position"


# -------------------------------------------------------------- shape contracts


def test_target_must_carry_one_more_row_than_there_are_proposals():
    """The extra row is the bonus position; without it a full acceptance has no token."""
    probability = distribution(0.5, 0.5)
    with pytest.raises(ValueError, match="plus the bonus"):
        accept_proposals(rows(probability), rows(probability), [0])


def test_the_two_models_must_share_a_vocabulary():
    with pytest.raises(ValueError, match="vocabulary"):
        accept_proposals(
            rows(distribution(0.5, 0.5), distribution(0.5, 0.5)),
            rows(distribution(0.3, 0.3, 0.4)),
            [0],
        )
