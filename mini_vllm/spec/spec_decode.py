"""One speculative iteration: propose, verify in a single target pass, accept, roll back.

A decode step reads every weight in the model to produce one token, so it is bound by
memory bandwidth with the arithmetic units mostly idle. Reading the same weights to score
`k + 1` tokens costs nearly the same, because scoring existing tokens is what attention
parallelizes. If a cheap draft guesses `k` tokens and the target agrees with most of them,
the engine gets several tokens per expensive pass instead of one.

The cost is `k` cheap forward passes plus the ones a rejection discards. Accuracy is not
part of the cost: :mod:`mini_vllm.spec.rejection` preserves the target's output
distribution exactly.

The ordering in :meth:`SpeculativeDecoder.step` is forced:

1. Propose, before the scheduler runs, because proposals lengthen a sequence and the
   scheduler must reserve slots for them.
2. Verify with `all_rows=True`, because the target's distribution is needed at every
   proposed position rather than only the last.
3. Accept, per sequence, by the rejection rule.
4. Roll back in the same iteration, returning the pages the rejected tail no longer needs,
   so no page is ever held by a token that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mini_vllm.sampler import sampling_probabilities
from mini_vllm.serve.sequence import Sequence
from mini_vllm.spec.proposer import DraftProposer
from mini_vllm.spec.rejection import accept_proposals

__all__ = ["SpecStats", "SpeculativeDecoder"]


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


class SpeculativeDecoder:
    """Wraps a target model and a draft proposer into one verified decode step."""

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

    def propose(self, sequences: list[Sequence]) -> dict[int, object]:
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
        proposals: dict[int, object],
        logits: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> list[tuple[Sequence, list[int]]]:
        """Judge every proposal from one `T x V` target pass, committing what survives.

        ``logits`` covers all `T` tokens of the iteration, so a sequence's `k + 1`
        distributions are the `k + 1` consecutive rows it contributed. Row `i` is the
        target's distribution for the token following position `i`, so the rows aligned
        with a sequence's pending token and its `k` proposals are exactly the `k + 1`
        distributions the acceptance rule needs.
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
