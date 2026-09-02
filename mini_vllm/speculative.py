"""Speculative decoding: free tokens that cost no accuracy.

What this file teaches
    The complete speculative-decoding loop, bottom-up:

    1. `residual_distribution` — the corrective distribution `norm(relu(p - q))`.
    2. `accept_proposals` — the rejection-sampling rule that preserves the
       target's output distribution exactly.
    3. `Proposal` / `DraftProposer` — the draft model, its own KV pool, its
       shadow sequences, and the cache resynchronization after every verdict.
    4. `SpeculativeDecoder` — one iteration: propose, verify in a single
       target pass, commit the accepted prefix, roll back the rejected tail's
       pages, and count what happened (`SpecStats`).

Inputs and outputs
    The proposer takes running `Sequence`s and returns `k` proposed tokens
    plus the draft distributions `q` they were drawn from. The verifier takes
    the target's `T x V` logits for the whole ragged iteration and returns,
    per sequence, the run of tokens actually committed (1 to k+1 of them).

Read next
    `benchmark.py` — where acceptance rate and wall clock are measured.

One invariant
    The emitted tokens are distributed exactly as if the target model had
    sampled them one at a time. A proposal is accepted with probability
    `min(1, p/q)`; the first rejection draws once from `norm(relu(p - q))`
    and discards everything after; a fully accepted run takes a bonus token
    from `p_{k+1}`, which the same target pass already computed. The draft's
    quality therefore affects only latency, never output — and at temperature
    zero the same code path degenerates to "keep the prefix where the two
    argmaxes agree", with no randomness, so greedy speculative output is
    token-identical to greedy non-speculative output.

Why this wins at all: a decode step reads every weight in the model to
produce one token, so it is bound by memory bandwidth with the arithmetic
units mostly idle. Reading the same weights to score `k + 1` tokens costs
nearly the same, because scoring existing tokens is what attention
parallelizes. If a cheap draft guesses `k` tokens and the target agrees with
most of them, the engine gets several tokens per expensive pass instead of
one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mini_vllm.cache import BlockManager
from mini_vllm.ops import sampling_probabilities
from mini_vllm.scheduler import ForwardBatch, Sequence, SequenceStatus

__all__ = [
    "ResidualError",
    "residual_distribution",
    "accept_proposals",
    "Proposal",
    "DraftProposer",
    "SpeculativeDecoder",
    "SpecStats",
]


# ---------------------------------------------------- 1. residual distribution


class ResidualError(ValueError):
    """Raised when a residual distribution has no mass to sample from."""


def residual_distribution(target: torch.Tensor, draft: torch.Tensor) -> torch.Tensor:
    """``norm(relu(p - q))``: the target's belief minus what the draft already covered.

    ::

        target, draft: V     probabilities over the vocabulary
        returns:       V     probabilities, summing to one

    The clamp makes this a distribution rather than a signed difference. The mass before
    renormalizing is `1 - Σ min(p, q)`, the exact shortfall the acceptance test leaves
    behind, so this is the missing term rather than a heuristic repair: the probability
    the acceptance step emits token `t` is `q(t)·min(1, p(t)/q(t)) = min(q(t), p(t))`,
    and drawing the rejection case from the normalized `relu(p - q)` restores `p`
    exactly. The test suite verifies this empirically with a chi-square test.
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


# --------------------------------------------------- 2. rejection-sampling rule


def accept_proposals(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    proposals: list[int] | torch.Tensor,
    generator: torch.Generator | None = None,
    greedy: bool = False,
) -> tuple[list[int], int]:
    """Verify one sequence's proposals, returning the tokens to commit.

    The rule (Leviathan et al. 2023; Chen et al. 2023):

    * Accept proposal `x_i` with probability `min(1, p_i(x_i) / q_i(x_i))`. A token the
      target likes at least as much as the draft did is always kept; one it likes half
      as much is kept half the time.
    * On the first rejection, stop and draw one token from the residual
      `norm(relu(p_i - q_i))`. Every later proposal is discarded — each was conditioned
      on a token that is no longer there.
    * If every proposal is accepted, take a bonus token from `p_{k+1}`, which the
      target scored in the same forward pass.

    ::

        target_probs: (k + 1) x V   the target's distribution at each proposed position,
                                    plus one more for the bonus
        draft_probs:  k x V         the draft's distribution at each proposed position
        proposals:    k             the draft's tokens
        returns:      (tokens, num_accepted)

    ``tokens`` is what the sequence should append: the accepted prefix followed by either
    the bonus token (nothing rejected) or the residual token (something rejected). Its
    length is ``num_accepted + 1``, so a step always makes progress and never stalls.

    ``num_accepted`` is reported separately because it is the speedup: `k` accepted
    proposals turn `k + 1` target passes into one, while zero accepted means the draft
    passes were spent for nothing. Its average over a run determines whether the draft
    model pays for itself.

    ``greedy=True`` is the explicit temperature-zero path: `p` and `q` are point
    masses, so `p(x)/q(x)` is 1 when the argmaxes agree and 0 otherwise, and acceptance
    reduces to keeping the prefix where the two models agree with no randomness.
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
        # proposal is discarded.
        residual = residual_distribution(target_row, draft_row)
        return accepted + [_draw(residual, generator)], len(accepted)

    # Every proposal survived, so the target's distribution for the following position
    # is already computed and the bonus token is free.
    bonus_row = target_probs[num_proposals]
    bonus = int(bonus_row.argmax()) if greedy else _draw(bonus_row, generator)
    return accepted + [bonus], len(accepted)


# ------------------------------------------------- 3. proposal representation


@dataclass
class Proposal:
    """What a draft produced for one sequence, and the distributions it drew from.

    ``probabilities`` is the `q` of the acceptance rule, and must be the distribution the
    token was actually sampled from: the draft's logits after the request's temperature
    and top-p, not the raw softmax. The rejection test compares `p/q` and is exact only
    if `q` is the true proposal distribution.
    """

    token_ids: list[int]
    probabilities: torch.Tensor  # k x V, fp32

    def __len__(self) -> int:
        return len(self.token_ids)


# ------------------------------------------- 4. the draft model and its state


class DraftProposer:
    """A draft model, its own KV pool, and one shadow sequence per request in flight.

    The proposer runs a smaller model forward `k` times to guess what the target will
    say. Two constraints shape the implementation.

    The draft keeps its own KV cache. It is a different model over the same tokens, so
    its keys and values are its own, in its own pool with its own block tables. The
    shadow sequences hold those tables: one per target sequence, carrying the same
    tokens and a separate count of how many the draft has computed.

    The draft must be resynchronized every step, and is not always behind by one. The
    target accepts some prefix of the proposals and replaces the rest, so after
    verification the draft's cache holds tokens the sequence no longer contains.
    :meth:`_synchronize` finds the first divergence, returns the pages past it, and
    lets the next forward pass recompute from there. Skipping this fails *silently*:
    the draft would propose from a context including rejected tokens, acceptance would
    collapse, and rejection sampling would still produce correct output, only slower.

    The `k` proposals cost `k` sequential forward passes. They are cheap passes over a
    small model and are batched across every sequence in flight, so the cost is `k`
    launches rather than `k` times the batch size.
    """

    def __init__(
        self,
        model,
        manager: BlockManager,
        num_speculative_tokens: int = 4,
    ) -> None:
        if num_speculative_tokens < 1:
            raise ValueError(
                f"speculating needs at least one proposal, got {num_speculative_tokens}"
            )
        self.model = model
        self.manager = manager
        self.num_speculative_tokens = num_speculative_tokens
        self.device = manager.kv.device
        self._shadows: dict[int, Sequence] = {}

    # ---------------------------------------------------------------- lifecycle

    def release(self, sequence: Sequence) -> None:
        """Drop a finished sequence's draft cache. Called when the target frees it."""
        shadow = self._shadows.pop(sequence.seq_id, None)
        if shadow is not None:
            self.manager.free(shadow)

    def release_all(self) -> None:
        for shadow in list(self._shadows.values()):
            self.manager.free(shadow)
        self._shadows.clear()

    @property
    def num_tracked(self) -> int:
        return len(self._shadows)

    def _shadow_for(self, sequence: Sequence) -> Sequence:
        """The draft-side twin of a target sequence, created on first sight."""
        shadow = self._shadows.get(sequence.seq_id)
        if shadow is None:
            shadow = Sequence(
                prompt_token_ids=list(sequence.prompt_token_ids),
                sampling_params=sequence.sampling_params,
                # The target decides when a request is done; a shadow that hit its own
                # limit mid-run would refuse another token, so give it headroom it will
                # never use.
                max_tokens=sequence.max_tokens + self.num_speculative_tokens + 1,
            )
            shadow.set_status(SequenceStatus.RUNNING)
            self._shadows[sequence.seq_id] = shadow
        return shadow

    # ----------------------------------------- 5. draft-cache synchronization

    def _synchronize(self, sequence: Sequence, shadow: Sequence) -> None:
        """Bring the shadow's tokens and cache in line with what the target committed.

        The draft's cache is valid exactly as far as the two agree. Past the first
        divergence it describes a sequence that no longer exists, so those pages are
        returned and the next forward pass recomputes from there.
        """
        committed = sequence.prompt_token_ids + sequence.output_token_ids
        existing = shadow.token_ids

        shared = 0
        for mine, theirs in zip(existing, committed):
            if mine != theirs:
                break
            shared += 1

        if shadow.num_computed_tokens > shared:
            self.manager.trim(shadow, shadow.num_computed_tokens - shared)
            shadow.num_computed_tokens = shared

        shadow.output_token_ids = list(sequence.output_token_ids)

    # --------------------------------------------------- 6. proposal generation

    @torch.no_grad()
    def propose(self, sequences: list[Sequence]) -> dict[int, Proposal]:
        """Run `k` batched draft decodes, returning one proposal per sequence.

        The first pass does double duty. A shadow that is behind — freshly created with
        its whole prompt uncomputed, or resynchronized after a rejection — folds its
        catch-up into the same forward that produces the first proposal, since the logits
        at the end of a catch-up chunk are the next token's distribution. `k` proposals
        therefore cost `k` passes, never `k + 1`.
        """
        if not sequences:
            return {}

        k = self.num_speculative_tokens
        selected: list[Sequence] = []
        shadows: list[Sequence] = []
        reserved = 0
        for sequence in sequences:
            shadow = self._shadow_for(sequence)
            self._synchronize(sequence, shadow)

            # The draft pool has no scheduler in front of it: the target's admission
            # control sizes the batch against the target's pages and knows nothing about
            # these, so this is the only place that can decline. A sequence whose draft
            # cache will not fit is not speculated this iteration and decodes normally,
            # so speculation can never be why the engine fails to make progress.
            needed = self.manager.blocks_needed(shadow, shadow.num_uncomputed_tokens + k)
            if needed > self.manager.num_free_blocks - reserved:
                continue
            reserved += needed
            selected.append(sequence)
            shadows.append(shadow)

        if not selected:
            return {}
        sequences = selected
        params = [sequence.sampling_params for sequence in sequences]
        tokens: list[list[int]] = [[] for _ in sequences]
        step_probabilities: list[torch.Tensor] = []

        for _ in range(k):
            logits = self._forward(shadows)
            probabilities = sampling_probabilities(logits, params)  # N x V, fp32
            drawn = self._draw(probabilities, params)

            step_probabilities.append(probabilities)
            for index, shadow in enumerate(shadows):
                token = int(drawn[index])
                tokens[index].append(token)
                shadow.append_token(token)

        # k x N x V -> one k x V block per sequence.
        stacked = torch.stack(step_probabilities)
        return {
            sequence.seq_id: Proposal(token_ids=tokens[index], probabilities=stacked[:, index, :])
            for index, sequence in enumerate(sequences)
        }

    def _draw(self, probabilities: torch.Tensor, params) -> list[int]:
        """One token per row, from the distributions the proposals must record.

        Sampled from `probabilities` rather than by calling the sampler again on the
        logits, so the token and the `q` handed to the verifier cannot disagree; that
        disagreement would break the acceptance test silently.
        """
        if all(parameter.is_greedy for parameter in params):
            return probabilities.argmax(dim=-1).tolist()
        return torch.multinomial(probabilities, 1).squeeze(-1).tolist()

    def _forward(self, shadows: list[Sequence]) -> torch.Tensor:
        """One ragged draft pass over whatever each shadow has left to compute.

        ::

            returns: N x V   the last row of each shadow, its next-token distribution
        """
        scheduled = [(shadow, shadow.num_uncomputed_tokens) for shadow in shadows]
        for shadow, count in scheduled:
            self.manager.allocate(shadow, count)

        batch = ForwardBatch.from_scheduled(scheduled, self.device, manager=self.manager)
        logits = self.model(batch)

        for shadow, count in scheduled:
            shadow.advance(count)
        return logits


# ---------------------------------------- 7-11. verification and rollback


class SpeculativeDecoder:
    """Wraps a target model and a draft proposer into one verified decode step.

    The ordering of a speculative iteration (driven by `engine.LLM._speculative_step`)
    is forced:

    1. Propose, *before* the scheduler runs, because proposals lengthen a sequence and
       the scheduler must reserve slots for them.
    2. Verify with `all_rows=True`, because the target's distribution is needed at
       every proposed position rather than only the last.
    3. Accept, per sequence, by the rejection rule.
    4. Roll back in the same iteration, returning the pages the rejected tail no longer
       needs, so no page is ever held by a token that does not exist.
    """

    def __init__(self, proposer: DraftProposer, model, manager, runner=None) -> None:
        self.proposer = proposer
        self.model = model
        self.manager = manager
        self.runner = runner
        self.stats = SpecStats()

    @property
    def num_speculative_tokens(self) -> int:
        return self.proposer.num_speculative_tokens

    def candidates(self, sequences: list[Sequence]) -> list[Sequence]:
        """Which sequences can be speculated for this iteration.

        Only sequences past their prefill. There is nothing to speculate from until a
        prompt is computed, and a chunked prefill's later chunks already keep the GPU
        busy, so speculating on top of them would add draft passes to an iteration that
        was not bandwidth-starved.
        """
        return [
            sequence
            for sequence in sequences
            if not sequence.is_prefill() and sequence.output_token_ids
        ]

    def propose(self, sequences: list[Sequence]) -> dict[int, Proposal]:
        """Attach draft proposals to each sequence, so the scheduler reserves for them."""
        proposals = self.proposer.propose(sequences)
        for sequence in sequences:
            proposal = proposals.get(sequence.seq_id)
            if proposal is not None:
                sequence.propose(proposal.token_ids)
        return proposals

    @torch.no_grad()
    def verify(
        self,
        scheduled: list[tuple[Sequence, int]],
        proposals: dict[int, Proposal],
        logits: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> list[tuple[Sequence, list[int]]]:
        """Judge every proposal from one `T x V` target pass, committing what survives.

        ``logits`` covers all `T` tokens of the iteration, so a sequence's `k + 1`
        distributions are the `k + 1` consecutive rows it contributed. Row `i` is the
        target's distribution for the token following position `i`, so the rows aligned
        with a sequence's pending token and its `k` proposals are exactly the `k + 1`
        distributions the acceptance rule needs.

        Commits through `Sequence.accept` (which truncates at a stop token) and rolls
        back the rejected tail's cache slots through `manager.trim` in the same
        iteration.
        """
        emitted: list[tuple[Sequence, list[int]]] = []
        row = 0
        for sequence, count in scheduled:
            span = slice(row, row + count)
            row += count

            proposal = proposals.get(sequence.seq_id)
            if proposal is None or not sequence.proposed_token_ids:
                continue

            params = sequence.sampling_params
            target_probs = sampling_probabilities(logits[span], [params] * count)
            tokens, num_accepted = accept_proposals(
                target_probs,
                proposal.probabilities,
                sequence.proposed_token_ids,
                generator=generator,
                greedy=params.is_greedy,
            )

            before = len(sequence.output_token_ids)
            to_trim = sequence.accept(tokens, num_accepted)
            self.stats.rolled_back_blocks += self.manager.trim(sequence, to_trim)

            # What was committed, which is not always what the sampler returned: a stop
            # token inside the accepted run ends the sequence and later tokens are
            # dropped. A streaming caller must see the committed run.
            committed = sequence.output_token_ids[before:]

            self.stats.steps += 1
            self.stats.proposed += len(proposal)
            self.stats.accepted += num_accepted
            self.stats.emitted += len(committed)
            self.stats._per_step.append(len(committed))
            emitted.append((sequence, committed))
        return emitted

    def release(self, sequence: Sequence) -> None:
        """Let the draft drop a finished sequence's cache alongside the target's."""
        self.proposer.release(sequence)


# ------------------------------------------------- 12. acceptance statistics


@dataclass
class SpecStats:
    """How well the draft is doing, which is the only tunable thing about speculation."""

    steps: int = 0
    proposed: int = 0
    accepted: int = 0
    emitted: int = 0
    rolled_back_blocks: int = 0
    _per_step: list[int] = field(default_factory=list, repr=False)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposals the target kept. The number that decides `k`.

        Below roughly `1/(k+1)` the draft is not paying for its own forward passes and
        speculation is a slowdown; near 1 a larger `k` would do better. It is a property
        of the draft-target pair and the text, not of this code.
        """
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_step(self) -> float:
        """Tokens emitted per expensive target pass: the speedup before overheads."""
        return self.emitted / self.steps if self.steps else 0.0
