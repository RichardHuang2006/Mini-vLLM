"""Speculative decoding: a draft proposes k tokens, the target verifies them in one pass,
rejection sampling keeps its distribution exact, and the rejected tail is rolled back."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx

from mini_vllm.batch import ForwardBatch
from mini_vllm.paged_kv_cache import BlockManager, PagedKvPool
from mini_vllm.qwen3 import Qwen3Model
from mini_vllm.sampler import sampling_probabilities
from mini_vllm.scheduler import Request

__all__ = [
    "DraftProposer",
    "SpecStats",
    "SpeculativeDecoder",
    "accept_proposals",
    "residual_distribution",
    "self_draft",
]


def residual_distribution(target: mx.array, draft: mx.array) -> mx.array:
    """norm(relu(p - q)): the target's belief minus what the draft already covered.

    Its mass before normalizing is 1 - sum(min(p, q)), exactly the shortfall acceptance
    leaves behind, so a rejection drawn from it restores p rather than repairing it.
    """
    residual = mx.maximum(target - draft, 0.0)
    return residual / mx.sum(residual)


def _draw(probabilities: mx.array) -> int:
    """One sample from a 1-D probability vector."""
    return mx.random.categorical(mx.log(probabilities)).item()


def accept_proposals(
    target_probs: mx.array,
    draft_probs: mx.array,
    proposals: list[int],
    greedy: bool = False,
) -> tuple[list[int], int]:
    """Verify one request's k proposals, returning (tokens, num_accepted).

    target_probs is (k + 1) x V, draft_probs k x V. The rule (Leviathan et al. 2023;
    Chen et al. 2023): accept x_i with probability min(1, p_i / q_i); on the first
    rejection draw once from norm(relu(p_i - q_i)) and discard the rest; on full
    acceptance take a bonus token from p_{k+1}. Greedy keeps the prefix whose argmaxes agree.
    """
    accepted = []
    for index, token in enumerate(proposals):
        target_row, draft_row = target_probs[index], draft_probs[index]

        if greedy:
            choice = mx.argmax(target_row).item()
            if choice != token:
                return [*accepted, choice], len(accepted)
            accepted.append(token)
            continue

        # q > 0 for any token the draft drew, so the ratio is finite.
        ratio = mx.minimum(target_row[token] / draft_row[token], 1.0)
        if mx.random.uniform().item() < ratio.item():
            accepted.append(token)
            continue

        # Rejected: draw from the residual, and discard the proposals conditioned on this one.
        return [*accepted, _draw(residual_distribution(target_row, draft_row))], len(accepted)

    # Every proposal survived, so the target's next distribution is already computed.
    bonus_row = target_probs[len(proposals)]
    bonus = mx.argmax(bonus_row).item() if greedy else _draw(bonus_row)
    return [*accepted, bonus], len(accepted)


def self_draft(model: Qwen3Model, num_layers: int) -> Qwen3Model:
    """The target's first num_layers layers and its head, sharing every weight: a draft
    that needs no second checkpoint."""
    return Qwen3Model(
        replace(model.config, num_hidden_layers=num_layers),
        model.embedding,
        model.layers[:num_layers],
        model.norm,
        model.lm_head,
    )


class DraftProposer:
    """A draft model, its own KV pool, and one shadow request per target request.

    The draft pool has no scheduler: a request whose next proposals would not fit is
    simply not speculated this step.
    """

    def __init__(self, model: Qwen3Model, kv: PagedKvPool, num_speculative_tokens: int) -> None:
        self.model = model
        self.kv = kv
        self.manager = BlockManager(kv)
        self.num_speculative_tokens = num_speculative_tokens
        self.shadows: dict[int, Request] = {}

    def release(self, request: Request) -> None:
        """Drop a request's draft cache: it finished, or was preempted and will recompute."""
        shadow = self.shadows.pop(request.request_id, None)
        if shadow is not None:
            self.manager.free(shadow)

    def propose(self, requests: list[Request]) -> dict[int, tuple[list[int], mx.array]]:
        """k batched draft passes, returning each request's k tokens and the k x V
        distribution q they were drawn from, after its own temperature, top-k and top-p:
        the acceptance test is exact only against the q that was actually sampled.

        The first pass does double duty: a shadow that is behind folds its catch-up into
        the forward that yields the first proposal, so k proposals cost k passes.
        """
        k = self.num_speculative_tokens
        chosen, reserved = [], 0
        for request in requests:
            shadow = self._synchronize(request)
            needed = self.manager.blocks_needed(shadow, shadow.num_uncomputed_tokens + k)
            if needed <= self.manager.pool.num_free - reserved:
                reserved += needed
                chosen.append((request, shadow))
        if not chosen:
            return {}

        params = [request.sampling_params for request, _ in chosen]
        steps = []
        for _ in range(k):
            scheduled = [(shadow, shadow.num_uncomputed_tokens) for _, shadow in chosen]
            for shadow, count in scheduled:
                self.manager.allocate(shadow, count)
            batch = ForwardBatch.from_scheduled(scheduled, self.manager)
            caches = self.kv.caches(batch)
            logits = self.model(batch.input_ids[None], batch.positions, caches, rows=batch.last_rows)
            q = sampling_probabilities(logits[0], params)
            # log(0) = -inf, so a greedy row's one-hot q draws its argmax exactly.
            drawn = mx.random.categorical(mx.log(q), axis=-1)
            mx.eval(drawn, q, self.kv.keys, self.kv.values)
            steps.append(q)
            for (shadow, count), token in zip(scheduled, drawn.tolist(), strict=True):
                shadow.num_computed_tokens += count
                shadow.output_token_ids.append(token)

        q_all = mx.stack(steps, axis=1)  # N x k x V
        return {
            request.request_id: (shadow.output_token_ids[-k:], q_all[index])
            for index, (request, shadow) in enumerate(chosen)
        }

    def _synchronize(self, request: Request) -> Request:
        """Bring the request's shadow in line with what the target committed. The draft's
        cache is valid exactly as far as the two agree; past the first divergence its
        slots go back, and the next pass recomputes from there."""
        shadow = self.shadows.get(request.request_id)
        if shadow is None:
            shadow = self.shadows[request.request_id] = Request(prompt_token_ids=request.prompt_token_ids)

        committed = request.prompt_token_ids + request.output_token_ids
        shared = 0
        for mine, theirs in zip(shadow.token_ids, committed, strict=False):
            if mine != theirs:
                break
            shared += 1
        if shadow.num_computed_tokens > shared:
            self.manager.trim(shadow, shadow.num_computed_tokens - shared)
            shadow.num_computed_tokens = shared

        shadow.output_token_ids = list(request.output_token_ids)
        return shadow


@dataclass
class SpecStats:
    """How well the draft is doing, which is the only tunable thing about speculation."""

    steps: int = 0
    proposed: int = 0
    accepted: int = 0
    emitted: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposals kept: below about 1 / (k + 1) the draft is not paying for
        its own passes; near 1 a larger k would do better."""
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_step(self) -> float:
        """Tokens emitted per target pass: the speedup before overheads."""
        return self.emitted / self.steps if self.steps else 0.0


class SpeculativeDecoder:
    """Proposes before the scheduler runs, since proposals lengthen a request and it must
    reserve slots for them; verifies after the target pass; rolls back in the same step."""

    def __init__(self, proposer: DraftProposer, manager: BlockManager) -> None:
        self.proposer = proposer
        self.manager = manager
        self.stats = SpecStats()
        self.draft_probs: dict[int, mx.array] = {}

    def propose(self, requests: list[Request]) -> None:
        """Attach proposals to every decoding request the draft pool can take."""
        proposals = self.proposer.propose(requests)
        for request in requests:
            if request.request_id in proposals:
                tokens, q = proposals[request.request_id]
                request.proposed_token_ids = list(tokens)
                self.draft_probs[request.request_id] = q

    def verify(self, request: Request, target_logits: mx.array) -> list[int]:
        """Judge a request's proposals from its k + 1 target rows, commit what survives,
        give back the rejected tail's slots, and return the tokens it emitted.

        Row i is the target's distribution for the token after position i, so the pending
        token and the k proposals span exactly the k + 1 rows the rule needs.
        """
        params = request.sampling_params
        proposals = request.proposed_token_ids
        target_probs = sampling_probabilities(target_logits, [params] * (len(proposals) + 1))
        q = self.draft_probs.pop(request.request_id)
        tokens, num_accepted = accept_proposals(target_probs, q, proposals, params.is_greedy)

        before = len(request.output_token_ids)
        self.stats.steps += 1
        self.stats.proposed += len(proposals)
        self.stats.accepted += num_accepted
        self.manager.trim(request, request.accept(tokens, num_accepted))
        emitted = request.output_token_ids[before:]
        self.stats.emitted += len(emitted)
        return emitted

    def discard(self, request: Request) -> None:
        """Drop proposals that were not scheduled: q was drawn for this context only."""
        request.proposed_token_ids = []
        self.draft_probs.pop(request.request_id, None)

    def release(self, request: Request) -> None:
        self.discard(request)
        self.proposer.release(request)
