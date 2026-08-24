"""The draft half of speculative decoding: `k` cheap tokens and their distributions.

The proposer runs a smaller model forward `k` times to guess what the target will say.
Two constraints shape the implementation.

The draft keeps its own KV cache. It is a different model over the same tokens, so its
keys and values are its own, in its own pool with its own block tables. The shadow
sequences hold those tables: one per target sequence, carrying the same tokens and a
separate count of how many the draft has computed.

The draft must be resynchronized every step, and is not always behind by one. The target
accepts some prefix of the proposals and replaces the rest, so after verification the
draft's cache holds tokens the sequence no longer contains. :meth:`_synchronize` finds the
first divergence, returns the pages past it, and lets the next forward pass recompute from
there. Skipping this fails silently: the draft would propose from a context including
rejected tokens, acceptance would collapse, and rejection sampling would still produce
correct output, only slower.

The `k` proposals cost `k` sequential forward passes. They are cheap passes over a small
model and are batched across every sequence in flight, so the cost is `k` launches rather
than `k` times the batch size.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_vllm.block.block_manager import BlockManager
from mini_vllm.sampler import sampling_probabilities
from mini_vllm.serve.batch import ForwardBatch
from mini_vllm.serve.sequence import Sequence, SequenceStatus

__all__ = ["DraftProposer", "Proposal"]


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

    def _synchronize(self, sequence: Sequence, shadow: Sequence) -> None:
        """Bring the shadow's tokens and cache in line with what the target committed.

        The draft's cache is valid exactly as far as the two agree. Past the first
        divergence it describes a sequence that no longer exists, so those pages are
        returned and the next forward pass recomputes over the committed tokens.
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

    # ----------------------------------------------------------------- proposing

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
