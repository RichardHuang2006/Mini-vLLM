"""Speculative decoding: propose k tokens, verify them in one target pass, roll back."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mini_vllm.cache import BlockManager
from mini_vllm.ops import sampling_probabilities
from mini_vllm.scheduler import ForwardBatch, Sequence, SequenceStatus

__all__ = [
    "DraftProposer",
    "Proposal",
    "ResidualError",
    "SpecStats",
    "SpeculativeDecoder",
    "accept_proposals",
    "residual_distribution",
]


class ResidualError(ValueError):
    """Raised when a residual distribution has no mass to sample from."""


def residual_distribution(target: torch.Tensor, draft: torch.Tensor) -> torch.Tensor:
    """norm(relu(p - q)): the target's belief minus what the draft already covered.

    Its mass before renormalizing is 1 - sum(min(p, q)), the exact shortfall acceptance
    leaves behind, so drawing a rejection from it restores p rather than repairing it.
    """
    residual = (target - draft).clamp_min(0.0)
    total = residual.sum()
    if total <= 0:
        # Only reachable when p == q, in which case the proposal should have been kept.
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
    """Verify one sequence's k proposals, returning (tokens, num_accepted).

    target_probs is (k + 1) x V, draft_probs k x V. The rule (Leviathan et al. 2023;
    Chen et al. 2023): accept x_i with probability min(1, p_i/q_i); on the first
    rejection draw once from norm(relu(p_i - q_i)) and discard the rest; on full
    acceptance take a bonus from p_{k+1}. greedy=True is the temperature-zero path.
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
            # Point masses: keep the proposal while the argmaxes agree, with no draw.
            if int(target_row.argmax()) != token:
                return [*accepted, int(target_row.argmax())], len(accepted)
            accepted.append(token)
            continue

        target_p = target_row[token]
        draft_q = draft_row[token]
        # A token the draft could not produce makes the ratio infinite, so keep it.
        if draft_q <= 0:
            ratio = torch.ones((), dtype=target_probs.dtype, device=target_probs.device)
        else:
            ratio = (target_p / draft_q).clamp(max=1.0)

        uniform = torch.rand((), generator=generator, device=target_probs.device)
        if uniform < ratio:
            accepted.append(token)
            continue

        # Rejected: draw from the residual, and discard proposals conditioned on it.
        residual = residual_distribution(target_row, draft_row)
        return [*accepted, _draw(residual, generator)], len(accepted)

    # Every proposal survived, so the target's next distribution is already computed.
    bonus_row = target_probs[num_proposals]
    bonus = int(bonus_row.argmax()) if greedy else _draw(bonus_row, generator)
    return [*accepted, bonus], len(accepted)


@dataclass
class Proposal:
    """What a draft produced for one sequence, and the q the tokens were drawn from.

    `probabilities` must be the draft's logits after the request's temperature and
    top-p, since the p/q test is exact only then.
    """

    token_ids: list[int]
    probabilities: torch.Tensor  # k x V, fp32

    def __len__(self) -> int:
        return len(self.token_ids)


class DraftProposer:
    """A draft model, its own KV pool, and one shadow sequence per request in flight."""

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
                # The target decides when a request is done; this headroom is never used.
                max_tokens=sequence.max_tokens + self.num_speculative_tokens + 1,
            )
            shadow.set_status(SequenceStatus.RUNNING)
            self._shadows[sequence.seq_id] = shadow
        return shadow

    def _synchronize(self, sequence: Sequence, shadow: Sequence) -> None:
        """Bring the shadow's tokens and cache in line with what the target committed.

        The draft's cache is valid exactly as far as the two agree; past the first
        divergence those pages are returned and the next pass recomputes from there.
        """
        committed = sequence.prompt_token_ids + sequence.output_token_ids
        existing = shadow.token_ids

        shared = 0
        for mine, theirs in zip(existing, committed, strict=False):
            if mine != theirs:
                break
            shared += 1

        if shadow.num_computed_tokens > shared:
            self.manager.trim(shadow, shadow.num_computed_tokens - shared)
            shadow.num_computed_tokens = shared

        shadow.output_token_ids = list(sequence.output_token_ids)

    @torch.no_grad()
    def propose(self, sequences: list[Sequence]) -> dict[int, Proposal]:
        """Run k batched draft decodes, returning one proposal per sequence.

        The first pass does double duty: a shadow that is behind folds its catch-up into
        the forward that produces the first proposal, so k proposals cost k passes.
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

            # The draft pool has no scheduler, so this is the only place that can decline.
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
        """One token per row, drawn from the same probabilities handed to the verifier."""
        if all(parameter.is_greedy for parameter in params):
            return probabilities.argmax(dim=-1).tolist()
        return torch.multinomial(probabilities, 1).squeeze(-1).tolist()

    def _forward(self, shadows: list[Sequence]) -> torch.Tensor:
        """One ragged draft pass, returning N x V: each shadow's next-token distribution."""
        scheduled = [(shadow, shadow.num_uncomputed_tokens) for shadow in shadows]
        for shadow, count in scheduled:
            self.manager.allocate(shadow, count)

        batch = ForwardBatch.from_scheduled(scheduled, self.device, manager=self.manager)
        logits = self.model(batch)

        for shadow, count in scheduled:
            shadow.advance(count)
        return logits


class SpeculativeDecoder:
    """Wraps a target model and a draft proposer into one verified decode step.

    The ordering is forced: propose before the scheduler runs, because proposals
    lengthen a sequence and it must reserve slots for them; verify with all_rows=True;
    accept per sequence; and roll back in the same iteration.
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
        """Which sequences can be speculated for this iteration: only those past prefill."""
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
        """Judge every proposal from one T x V target pass, committing what survives.

        Row i is the target's distribution for the token following position i, so a
        sequence's pending token and its k proposals span exactly the k + 1 rows the
        rule needs. The rejected tail's slots go back through `BlockManager.trim`.
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

            # What was committed: a stop token inside the run drops the tokens after it.
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
        """Fraction of proposals the target kept: the number that decides k.

        Below roughly 1/(k+1) the draft is not paying for its own forward passes; near 1
        a larger k would do better.
        """
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_step(self) -> float:
        """Tokens emitted per expensive target pass: the speedup before overheads."""
        return self.emitted / self.steps if self.steps else 0.0
