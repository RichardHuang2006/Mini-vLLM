"""speculative.py: the acceptance rule keeps the target's distribution, the bookkeeping
commits and rolls back exactly, and greedy speculation changes no token."""

import mlx.core as mx
import pytest
from test_engine import PROMPTS, assert_nothing_held, engine
from utils import assert_allclose

from mini_vllm.generate import generate_with_kv_cache
from mini_vllm.paged_kv_cache import BlockManager, PagedKvPool
from mini_vllm.sampler import SamplingParams
from mini_vllm.scheduler import Request, Scheduler, SchedulerConfig
from mini_vllm.speculative import accept_proposals, residual_distribution

# --- the rule -------------------------------------------------------------------------------

P = mx.array([0.30, 0.25, 0.20, 0.10, 0.10, 0.05])  # the target
Q = mx.array([0.05, 0.10, 0.10, 0.20, 0.25, 0.30])  # a draft that disagrees with it


def test_the_residual_is_the_mass_acceptance_leaves_behind():
    """min(p, q) is kept by acceptance; the residual, scaled by what is left, restores p."""
    kept = mx.minimum(P, Q)
    assert_allclose(kept + (1 - mx.sum(kept)) * residual_distribution(P, Q), P)


@pytest.mark.parametrize("num_proposals", [1, 3])
def test_the_emitted_token_follows_the_target_not_the_draft(num_proposals):
    """Proposals drawn from q, verified against p: the first emitted token is distributed
    exactly as p, which is the whole claim of speculative decoding."""
    draws = 3000
    target = mx.broadcast_to(P, (num_proposals + 1, 6))
    draft = mx.broadcast_to(Q, (num_proposals, 6))
    proposals = mx.random.categorical(mx.log(draft), num_samples=draws).T.tolist()  # draws x k

    counts = [0] * 6
    for proposed in proposals:
        tokens, _ = accept_proposals(target, draft, proposed)
        counts[tokens[0]] += 1

    expected = [p * draws for p in P.tolist()]
    chi_square = sum((c - e) ** 2 / e for c, e in zip(counts, expected, strict=True))
    # 5 degrees of freedom: 20.5 is the 0.1% critical value. Following q would score ~1200.
    assert chi_square < 20.5, (chi_square, counts)


def test_a_matching_draft_is_always_accepted():
    target = mx.broadcast_to(P, (4, 6))
    for _ in range(20):
        tokens, accepted = accept_proposals(target, target[:3], [0, 1, 2])
        assert accepted == 3 and tokens[:3] == [0, 1, 2] and len(tokens) == 4, "plus the bonus"


def test_a_rejection_truncates_rather_than_skipping():
    """A proposal the target gives no mass is always rejected, and nothing after it is kept."""
    target = mx.broadcast_to(mx.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0]), (3, 6))
    tokens, accepted = accept_proposals(target, mx.broadcast_to(Q, (2, 6)), [0, 1])
    assert accepted == 0 and len(tokens) == 1 and tokens[0] in (1, 2)


def test_greedy_keeps_exactly_the_prefix_where_the_models_agree():
    one_hot = mx.eye(6)
    target = one_hot[mx.array([2, 4, 1, 5])]  # the target's argmaxes: 2, 4, 1, then bonus 5

    assert accept_proposals(target, target[:3], [2, 4, 0], greedy=True) == ([2, 4, 1], 2)
    assert accept_proposals(target, target[:3], [2, 4, 1], greedy=True) == ([2, 4, 1, 5], 3)
    assert accept_proposals(target, target[:3], [3, 4, 1], greedy=True) == ([2], 0)


# --- the bookkeeping ------------------------------------------------------------------------


def speculating(output=(9,), proposals=(1, 2, 3), max_tokens=16, stop=()) -> Request:
    """A request mid-decode: 4 prompt tokens computed, one pending output, k proposals."""
    request = Request(prompt_token_ids=[5, 6, 7, 8], max_tokens=max_tokens, stop_token_ids=frozenset(stop))
    request.output_token_ids = list(output)
    request.num_computed_tokens = 4
    request.proposed_token_ids = list(proposals)
    return request


def test_proposals_lengthen_the_request_without_joining_its_output():
    request = speculating()
    assert len(request) == 8 and request.num_uncomputed_tokens == 4
    assert request.token_ids[-3:] == [1, 2, 3] and request.output_token_ids == [9]
    assert not request.is_done()


def test_accepting_everything_commits_the_run_and_the_bonus():
    request = speculating()
    assert request.accept([1, 2, 3, 4], 3) == 0, "nothing to give back"
    assert request.output_token_ids == [9, 1, 2, 3, 4] and not request.proposed_token_ids
    assert request.num_computed_tokens == 8 and request.num_uncomputed_tokens == 1


def test_rejecting_the_tail_gives_back_exactly_the_rejected_slots():
    request = speculating()
    assert request.accept([1, 7], 1) == 2, "proposals 2 and 3 were computed and are now wrong"
    assert request.output_token_ids == [9, 1, 7] and request.num_computed_tokens == 6


def test_a_stop_token_mid_run_truncates_the_rest():
    request = speculating(stop={2})
    assert request.accept([1, 2, 3, 4], 3) == 1
    assert request.output_token_ids == [9, 1, 2] and request.is_done()


def test_trim_returns_the_pages_a_rejected_tail_spilled_into():
    manager = BlockManager(PagedKvPool(1, 8, 4, 1, 1))
    request = Request(prompt_token_ids=[1, 2, 3, 4])
    manager.allocate(request, 4)  # exactly one full page
    free_after_prompt = manager.pool.num_free
    manager.allocate(request, 5)  # a pending token plus four proposals: two more pages

    assert manager.trim(request, 5) == 2
    assert manager.pool.num_free == free_after_prompt and request.block_table.num_tokens == 4


def test_a_speculative_group_is_scheduled_whole_or_not_at_all():
    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=3), BlockManager(PagedKvPool(1, 64, 4, 1, 1)))
    request = Request(prompt_token_ids=[1, 2, 3], max_tokens=8)
    scheduler.add(request)
    scheduler.commit(scheduler.schedule(), [4])

    request.proposed_token_ids = [5, 6, 7]  # 4 rows to verify, 3 tokens of budget
    assert scheduler.schedule().tokens_for(request) == 0, "a partial group cannot be verified"


def test_preemption_drops_unverified_proposals():
    scheduler = Scheduler(SchedulerConfig(), BlockManager(PagedKvPool(1, 64, 4, 1, 1)))
    request = Request(prompt_token_ids=[1, 2, 3], max_tokens=8)
    scheduler.add(request)
    scheduler.commit(scheduler.schedule(), [4])
    request.proposed_token_ids = [5, 6]

    scheduler.preempt(request)

    assert request.proposed_token_ids == [] and request.token_ids == [1, 2, 3, 4]


# --- through the engine ---------------------------------------------------------------------


@pytest.mark.parametrize(("k", "layers"), [(1, 1), (3, 1), (5, 2), (4, None)])
def test_greedy_speculation_changes_no_token(tiny_qwen3, k, layers):
    llm = engine(tiny_qwen3, num_speculative_tokens=k, num_draft_layers=layers)
    completions = llm.generate(PROMPTS, max_tokens=10)

    for prompt, completion in zip(PROMPTS, completions, strict=True):
        assert completion.token_ids == generate_with_kv_cache(llm.model, prompt, 10)
    assert llm.spec.stats.steps > 0, "nothing was speculated"
    assert_nothing_held(llm)
    assert llm.spec.proposer.manager.pool.num_free == llm.spec.proposer.kv.num_blocks, "draft blocks leaked"


def test_a_full_depth_self_draft_is_always_accepted(tiny_qwen3):
    """The draft is the target, so every greedy proposal is what the target would pick."""
    llm = engine(tiny_qwen3, num_speculative_tokens=4)
    llm.generate([list(range(10, 20))], max_tokens=21, ignore_eos=True)
    stats = llm.spec.stats
    assert stats.acceptance_rate == 1.0
    assert stats.tokens_per_step == 5.0, "k accepted plus the bonus, every step"


def test_a_shallow_draft_is_mostly_rejected_and_still_exact(tiny_qwen3):
    llm = engine(tiny_qwen3, num_speculative_tokens=4, num_draft_layers=1)
    completion = llm.generate([list(range(10, 20))], max_tokens=16)[0]
    assert completion.token_ids == generate_with_kv_cache(llm.model, list(range(10, 20)), 16)
    assert llm.spec.stats.acceptance_rate < 1.0, "a one-layer draft of a random model agreed every time"
    assert llm.spec.proposer.kv.keys.__len__() == 1, "the draft pool is as shallow as the draft"


def test_speculation_streams_each_run_in_order(tiny_qwen3):
    """A full-depth draft emits 1 + 5 + 5 tokens, then a run cut to 3 by max_tokens: every
    token of that run arrives in its own update, and only the very last one finishes."""
    llm = engine(tiny_qwen3, num_speculative_tokens=4)
    updates = list(llm.generate_stream([[3, 1, 4, 1, 5]], max_tokens=14))

    assert [u.token_id for u in updates] == generate_with_kv_cache(llm.model, [3, 1, 4, 1, 5], 14)
    assert [u.finish_reason for u in updates] == [None] * 13 + ["length"], "only the last update finishes"
    assert "".join(u.text for u in updates) == llm.tokenizer.decode([u.token_id for u in updates])


@pytest.mark.parametrize(
    "config",
    [
        {"max_batched_tokens": 12, "max_sequences": 4, "chunk_size": 8},  # chunked prefill beside it
        {"num_blocks": 14, "max_batched_tokens": 16},                      # steady preemption
    ],
)
def test_speculation_survives_chunking_and_preemption(tiny_qwen3, config):
    llm = engine(tiny_qwen3, num_speculative_tokens=3, num_draft_layers=1, **config)
    completions = llm.generate(PROMPTS, max_tokens=8)

    for prompt, completion in zip(PROMPTS, completions, strict=True):
        assert completion.token_ids == generate_with_kv_cache(llm.model, prompt, 8)
    assert_nothing_held(llm)


def test_sampled_speculation_replays_under_a_seed_and_leaks_nothing(tiny_qwen3):
    params = SamplingParams(temperature=1.0)
    runs = []
    for _ in range(2):
        llm = engine(tiny_qwen3, num_speculative_tokens=3, num_draft_layers=1, seed=11)
        runs.append([c.token_ids for c in llm.generate(PROMPTS, params, max_tokens=8)])
        assert_nothing_held(llm)
    assert runs[0] == runs[1]
    assert all(len(tokens) == 8 and max(tokens) < 512 for tokens in runs[0])
