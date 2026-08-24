"""Rejection sampling for speculative decoding: free tokens that cost no accuracy.

A draft model proposes tokens `x_1..x_k` from its own distributions `q_1..q_k`; the target
scores the same positions and produces `p_1..p_{k+1}`. This module decides which proposals
to keep, under the requirement that the emitted tokens be distributed exactly as if
sampled from the target one at a time. Anything weaker turns speculation into a quality
setting rather than a latency optimization.

The rule (Leviathan et al. 2023; Chen et al. 2023):

* Accept proposal `x_i` with probability `min(1, p_i(x_i) / q_i(x_i))`. A token the target
  likes at least as much as the draft did is always kept; one it likes half as much is kept
  half the time.
* On the first rejection, stop and draw one token from the residual
  `norm(relu(p_i - q_i))`: the target's belief minus the part the draft accounted for. The
  acceptance step under-samples exactly where `q` exceeded `p`, and the residual restores
  precisely that deficit.
* If every proposal is accepted, take a bonus token from `p_{k+1}`, which the target scored
  in the same forward pass.

A step therefore returns between 1 and `k + 1` tokens and never zero: a total rejection
still yields the residual draw, so speculation cannot stall.

The correction is exact because the probability this procedure emits token `t` at position
`i` is `q(t)·min(1, p(t)/q(t)) = min(q(t), p(t))`, whose sum over `t` falls short of one by
`1 - Σ min(p, q)` — exactly the mass of `relu(p - q)`. Drawing the rejection case from the
normalized `relu(p - q)` restores `p`. `tests/test_spec_decode.py` verifies this
empirically with a chi-square test.

Greedy decoding is the degenerate case on the same code path: at temperature zero `p` and
`q` are point masses, so `p(x)/q(x)` is 1 when the argmaxes agree and 0 otherwise, and
acceptance reduces to keeping the prefix where the two models agree with no randomness.
Greedy speculative output is therefore bit-identical to greedy non-speculative output.
"""

from __future__ import annotations

import torch

__all__ = ["ResidualError", "accept_proposals", "residual_distribution"]


class ResidualError(ValueError):
    """Raised when a residual distribution has no mass to sample from."""


def residual_distribution(target: torch.Tensor, draft: torch.Tensor) -> torch.Tensor:
    """``norm(relu(p - q))``: the target's belief minus what the draft already covered.

    ::

        target, draft: V     probabilities over the vocabulary
        returns:       V     probabilities, summing to one

    The clamp makes this a distribution rather than a signed difference. The mass before
    renormalizing is `1 - Σ min(p, q)`, the exact shortfall the acceptance test leaves
    behind, so this is the missing term rather than a heuristic repair.
    """
    residual = (target - draft).clamp_min(0.0)
    total = residual.sum()
    if total <= 0:
        # Only reachable when p is numerically dominated by q everywhere, which for
        # genuine distributions means p == q and the proposal should have been accepted.
        # The caller's contract is that a rejected position has residual mass; raising
        # surfaces a violation rather than masking it.
        raise ResidualError(
            "the residual relu(p - q) has no mass: p is dominated by q everywhere, so "
            "this position should not have been rejected"
        )
    return residual / total


def _draw(probabilities: torch.Tensor, generator: torch.Generator | None) -> int:
    """One sample from a 1-D probability vector."""
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def accept_proposals(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    proposals: list[int] | torch.Tensor,
    generator: torch.Generator | None = None,
    greedy: bool = False,
) -> tuple[list[int], int]:
    """Verify one sequence's proposals, returning the tokens to commit.

    ::

        target_probs: (k + 1) x V   the target's distribution at each proposed position,
                                    plus one more for the bonus
        draft_probs:  k x V         the draft's distribution at each proposed position
        proposals:    k             the draft's tokens
        returns:      (tokens, num_accepted)

    ``tokens`` is what the sequence should append: the accepted prefix followed by either
    the bonus token (nothing rejected) or the residual token (something rejected). Its
    length is ``num_accepted + 1``, so a step always makes progress.

    ``num_accepted`` is reported separately because it is the speedup: `k` accepted
    proposals turn `k + 1` target passes into one, while zero accepted means the draft
    passes were spent for nothing. Its average over a run determines whether the draft
    model pays for itself.
    """
    if isinstance(proposals, torch.Tensor):
        proposals = [int(token) for token in proposals]
    num_proposals = len(proposals)

    if target_probs.dim() != 2 or draft_probs.dim() != 2:
        raise ValueError(
            f"expected 2-D probabilities, got target {tuple(target_probs.shape)} "
            f"and draft {tuple(draft_probs.shape)}"
        )
    if target_probs.shape[0] != num_proposals + 1:
        raise ValueError(
            f"{num_proposals} proposals need {num_proposals + 1} rows of target "
            f"probabilities (one per position, plus the bonus), got {target_probs.shape[0]}"
        )
    if draft_probs.shape[0] != num_proposals:
        raise ValueError(
            f"{num_proposals} proposals need {num_proposals} rows of draft "
            f"probabilities, got {draft_probs.shape[0]}"
        )
    if target_probs.shape[1] != draft_probs.shape[1]:
        raise ValueError(
            f"target and draft disagree on the vocabulary: {target_probs.shape[1]} "
            f"vs {draft_probs.shape[1]}"
        )

    accepted: list[int] = []
    for index, token in enumerate(proposals):
        target_row = target_probs[index]
        draft_row = draft_probs[index]

        if greedy:
            # Point masses: keep the proposal while the two models' argmaxes agree. No
            # random draw, so a greedy speculative run is reproducible and identical to
            # the non-speculative one.
            if int(target_row.argmax()) != token:
                return accepted + [int(target_row.argmax())], len(accepted)
            accepted.append(token)
            continue

        target_p = target_row[token]
        draft_q = draft_row[token]
        # A token the draft could not have produced would make the ratio infinite, so
        # any target mass at all is enough to keep it.
        if draft_q <= 0:
            ratio = torch.ones((), dtype=target_probs.dtype, device=target_probs.device)
        else:
            ratio = (target_p / draft_q).clamp(max=1.0)

        uniform = torch.rand((), generator=generator, device=target_probs.device)
        if uniform < ratio:
            accepted.append(token)
            continue

        # Rejected: this position's token comes from the residual, and every later
        # proposal is discarded — each was conditioned on a token that is no longer
        # there.
        residual = residual_distribution(target_row, draft_row)
        return accepted + [_draw(residual, generator)], len(accepted)

    # Every proposal survived, so the target's distribution for the following position
    # is already computed and the bonus token is free.
    bonus_row = target_probs[num_proposals]
    bonus = int(bonus_row.argmax()) if greedy else _draw(bonus_row, generator)
    return accepted + [bonus], len(accepted)
