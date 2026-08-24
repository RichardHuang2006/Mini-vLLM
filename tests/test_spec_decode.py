"""Speculative decoding end to end: faster, and identical.

Speculation's guarantee is identity rather than closeness, so the central test is an
equality: the same prompt, greedily decoded, with and without a draft model, token for
token. Wrong rejection sampling, a wrong rollback, or misaligned rows in verification all
break that equality and none of them announces itself otherwise, since a wrong
implementation still produces fluent text.

The second concern is leaks. Speculation writes KV for tokens it then discards, in two
pools (the target's and the draft's), and a page dropped without a decref is an
out-of-memory a thousand requests later with nothing pointing at the cause. Every test
here ends by checking both pools are as full as they started.

Engines are built one at a time and torn down: the card has 8 GB, and a real Qwen3-0.6B
plus two KV pools is most of what fits. See `conftest.real_engine`.
"""

from __future__ import annotations

import pytest
import torch
from conftest import real_engine

from mini_vllm.sampler import SamplingParams

GREEDY = SamplingParams(temperature=0.0)


def assert_no_leaks(llm) -> None:
    """Both pools back to full: the target's and the draft's."""
    assert llm.manager.num_free_blocks == llm.manager.num_blocks, "the target pool leaked"
    draft = llm.spec.proposer.manager
    assert draft.num_free_blocks == draft.num_blocks, "the draft pool leaked"
    assert llm.spec.proposer.num_tracked == 0, "a shadow sequence outlived its request"


# ------------------------------------------------------- the equality that matters


@pytest.mark.oracle
def test_greedy_speculative_output_is_identical_to_non_speculative():
    """The whole claim, as an exact token comparison.

    fp32 rather than bf16, for the reason `test_engine.py` documents: bf16 greedy steps have
    genuine ties, and a speculative run reduces in a different order from a plain one, so
    the two could differ by a rounding without either being wrong. fp32 removes that
    ambiguity and leaves the comparison meaning what it says.

    One engine at a time — an fp32 Qwen3-0.6B is 2.4 GB, and the speculative one carries a
    second KV pool besides.
    """
    prompt = "The capital of France is"

    with real_engine(dtype=torch.float32, num_speculative_tokens=4) as llm:
        speculative = llm.generate(prompt, sampling_params=GREEDY, max_tokens=24)[0]
        stats = llm.spec.stats
        assert_no_leaks(llm)

    with real_engine(dtype=torch.float32) as llm:
        plain = llm.generate(prompt, sampling_params=GREEDY, max_tokens=24)[0]

    assert speculative.token_ids == plain.token_ids, (
        "speculation changed the output, which it is not allowed to do"
    )
    # And it did so in fewer target passes than tokens, or it was not speculating at all.
    assert stats.steps < len(plain.token_ids)


@pytest.mark.oracle
def test_a_self_draft_is_accepted_every_time():
    """A draft that *is* the target must agree with it on every proposal.

    Acceptance below 1.0 here would mean the two disagree about their own arithmetic — a
    misaligned verification row, or a `q` that is not the distribution the proposal was
    drawn from. It is the sharpest available check on the plumbing, separate from whether
    the output is right.
    """
    with real_engine(dtype=torch.float32, num_speculative_tokens=4) as llm:
        llm.generate("The capital of France is", sampling_params=GREEDY, max_tokens=24)
        stats = llm.spec.stats

        assert stats.acceptance_rate == 1.0, (
            f"a self-draft was rejected: {stats.accepted}/{stats.proposed} accepted"
        )
        # Four proposals plus the bonus, every step.
        assert stats.tokens_per_step == pytest.approx(5.0, abs=0.5)
        assert stats.rolled_back_blocks == 0, "nothing was rejected, so nothing rolled back"
        assert_no_leaks(llm)


@pytest.mark.oracle
def test_speculation_emits_the_whole_accepted_run_to_a_streaming_caller():
    """A step yields several tokens, and the stream must show all of them in order.

    The concatenated deltas have to reconstruct the completion exactly: a speculative step
    that emitted only its last token would still finish with the right output but would
    stream a truncated version of it.
    """
    with real_engine(dtype=torch.float32, num_speculative_tokens=4) as llm:
        updates = list(
            llm.generate_stream("The capital of France is", sampling_params=GREEDY, max_tokens=20)
        )
        streamed = [update.token_id for update in updates]
        completion = llm.generate("The capital of France is", sampling_params=GREEDY, max_tokens=20)[0]

        assert streamed == list(completion.token_ids), "the stream and the completion disagree"
        assert sum(update.finished for update in updates) == 1, "finished was reported twice"
        assert_no_leaks(llm)


# ---------------------------------------------------------------- rejection paths


@pytest.mark.oracle
def test_a_weak_draft_is_rejected_and_leaves_nothing_behind():
    """The case rollback exists for, driven hard.

    A four-layer prefix of Qwen3-0.6B is a genuinely bad draft — its proposals are rejected
    almost always — which makes it the right instrument for testing the rejection path.
    Every rejected proposal had KV written for it in both pools and has to give it back,
    and the output still has to be correct, because rejection sampling does not care how
    bad the draft is.
    """
    prompt = "The capital of France is"

    with real_engine(dtype=torch.float32, num_speculative_tokens=4, num_draft_layers=4) as llm:
        weak = llm.generate(prompt, sampling_params=GREEDY, max_tokens=24)[0]
        stats = llm.spec.stats
        assert stats.acceptance_rate < 0.5, "a 4-layer draft should mostly be rejected"
        assert_no_leaks(llm)

    with real_engine(dtype=torch.float32) as llm:
        plain = llm.generate(prompt, sampling_params=GREEDY, max_tokens=24)[0]

    assert weak.token_ids == plain.token_ids, (
        "a bad draft changed the output; rejection sampling must make the draft's quality "
        "irrelevant to correctness"
    )


@pytest.mark.oracle
def test_sampling_with_speculation_stays_in_the_vocabulary_and_leaks_nothing():
    """Non-greedy speculation: the residual path, exercised end to end.

    Temperature sampling cannot be checked against a fixed answer, so what is checked is
    that it produces real tokens, the requested number of them, and no leaked pages. The
    *distribution* is proven separately and much more sharply, by chi-square, in
    `test_rejection.py`.
    """
    warm = SamplingParams(temperature=0.9, top_p=0.95)
    with real_engine(num_speculative_tokens=3) as llm:
        completions = llm.generate(
            ["The capital of France is", "Once upon a time"],
            sampling_params=warm,
            max_tokens=20,
        )
        for completion in completions:
            assert len(completion.token_ids) <= 20
            assert all(0 <= token < llm.config.vocab_size for token in completion.token_ids)
        assert llm.spec.stats.proposed > 0, "nothing was speculated"
        assert_no_leaks(llm)


# -------------------------------------------------------------------- interaction


@pytest.mark.oracle
def test_speculation_survives_chunked_prefill_beside_it():
    """A long prompt chunking through the batch while other sequences speculate.

    The two features touch the same scheduler budget from opposite ends: a chunk wants as
    many tokens as it can get, and a speculative group is atomic and must have all `k + 1`
    or none. This is the test that they can share an iteration without either being
    corrupted — the prompts still produce their tokens, and the pools still balance.
    """
    long_prompt = "In a distant kingdom by the sea, the old chronicles record that " * 12
    with real_engine(num_speculative_tokens=4, chunk_size=64, max_batched_tokens=128) as llm:
        completions = llm.generate(
            [long_prompt, "The capital of France is", "Once upon a time"],
            sampling_params=GREEDY,
            max_tokens=16,
        )

        assert len(completions) == 3
        for completion in completions:
            assert completion.token_ids, "a sequence produced nothing"
        assert_no_leaks(llm)


@pytest.mark.oracle
def test_speculation_survives_preemption():
    """A pool too small for everything in flight, so the scheduler has to preempt.

    Preemption discards a sequence's unverified proposals and its whole cache, then
    recomputes it from the prompt — including, importantly, only the tokens it really
    committed. If a preempted sequence kept its proposals, its recomputed prefill would
    contain tokens the caller never received.
    """
    # The preemptions must come from the decodes rather than admission: admission declines
    # under memory pressure instead of resolving it, so a pool that merely cannot hold all
    # six would admit fewer. Forcing a preemption needs a pool that admission fills almost
    # exactly, leaving running sequences to cross page boundaries with nothing spare —
    # roughly 70 tokens of prompt each against 20 pages of 16 slots.
    prompts = [
        f"Tell me about the number {index} and why it matters. " * 6 for index in range(6)
    ]
    with real_engine(num_blocks=20, num_speculative_tokens=3, max_sequences=6) as llm:
        completions = llm.generate(prompts, sampling_params=GREEDY, max_tokens=24)

        assert len(completions) == len(prompts)
        for completion in completions:
            assert completion.token_ids
        assert llm.stats.preemptions > 0, "the pool was not tight enough to preempt"
        assert_no_leaks(llm)


@pytest.mark.oracle
def test_the_draft_pool_is_shallower_than_the_target_and_costs_less():
    """A self-draft shares weights, so the only memory it adds is its own KV.

    Worth pinning because it is the reason this arrangement fits at all: the draft's pool
    has the draft's layer count, not the target's, and a shallower pool is a smaller pool
    for the same number of pages.
    """
    from mini_vllm.block.kv_pool import PagedKvPool

    with real_engine(num_speculative_tokens=2, num_draft_layers=4) as llm:
        target_pool, draft_pool = llm.manager.kv, llm.spec.proposer.manager.kv

        assert draft_pool.num_layers == 4
        assert target_pool.num_layers == llm.config.num_hidden_layers

        def pool_bytes(pool, num_blocks: int | None = None) -> int:
            return PagedKvPool.bytes_for(
                num_layers=pool.num_layers,
                num_blocks=pool.num_blocks if num_blocks is None else num_blocks,
                block_size=pool.block_size,
                num_kv_heads=pool.num_kv_heads,
                head_dim=pool.head_dim,
                dtype=pool.kv_dtype,
            )

        def per_page(pool) -> int:
            return pool_bytes(pool, num_blocks=1)

        # A seventh of the cost per page, 4 layers against 28.
        assert per_page(draft_pool) * 7 == per_page(target_pool)
        # The draft holds more pages than the target, caching the same tokens plus the
        # proposals in flight, while remaining far smaller overall, which is what makes
        # this arrangement fit on an 8 GB card.
        assert draft_pool.num_blocks > target_pool.num_blocks
        assert pool_bytes(draft_pool) * 4 < pool_bytes(target_pool)
        # And the weights really are shared, not a second copy.
        assert llm.spec.proposer.model.blocks[0] is llm.model.blocks[0]
