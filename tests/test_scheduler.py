"""scheduler.py: the policy (admission, budget, chunking, piggybacking, preemption), then
token identity: batched, chunked, preempted and prefix-cached runs of the tiny model through
the paged pool, each against the request run alone through the dense cache."""

import random

import mlx.core as mx
import pytest

from mini_vllm.batch import ForwardBatch
from mini_vllm.generate import GREEDY, generate_with_kv_cache
from mini_vllm.models import from_mlx
from mini_vllm.paged_kv_cache import BlockManager, PagedKvPool
from mini_vllm.sampler import sample
from mini_vllm.scheduler import Request, RequestStatus, Scheduler, SchedulerConfig

WHOLE = {"enable_chunked_prefill": False}


def make(prompt_len: int, max_tokens: int = 4) -> Request:
    prompt = list(range(1, prompt_len + 1))
    return Request(prompt_token_ids=prompt, sampling_params=GREEDY, max_tokens=max_tokens)


def make_scheduler(num_blocks: int = 1024, block_size: int = 16, **config) -> Scheduler:
    """Policy tests only count blocks, so the pages are one number wide."""
    manager = BlockManager(PagedKvPool(1, num_blocks, block_size, 1, 1))
    return Scheduler(SchedulerConfig(**config), manager)


def add_all(scheduler: Scheduler, requests: list[Request]) -> None:
    for request in requests:
        scheduler.add(request)


# --- policy ---------------------------------------------------------------------------------


def test_admission_is_fcfs_and_bounded():
    scheduler = make_scheduler(max_sequences=2)
    first, second, third = make(4), make(4), make(4)
    add_all(scheduler, [first, second, third])

    assert scheduler.schedule().requests == [first, second]
    assert third.status is RequestStatus.WAITING


def test_admission_stops_at_the_token_budget():
    scheduler = make_scheduler(max_batched_tokens=512, **WHOLE)
    add_all(scheduler, [make(300), make(300)])
    assert scheduler.schedule().total_tokens == 300
    assert len(scheduler.waiting) == 1


def test_a_prompt_larger_than_the_budget_runs_alone_and_overruns_it():
    """Refusing it would deadlock the queue: the head-of-line stall chunking removes."""
    scheduler = make_scheduler(max_batched_tokens=512, **WHOLE)
    huge = make(2000)
    scheduler.add(huge)
    assert scheduler.schedule().scheduled == [(huge, 2000)]


def test_a_finished_request_is_replaced_in_the_same_iteration():
    """The point of continuous batching: no drain between one request and the next."""
    scheduler = make_scheduler(max_sequences=1)
    a, b = make(3, max_tokens=1), make(3, max_tokens=1)
    add_all(scheduler, [a, b])

    first = scheduler.schedule()
    assert first.requests == [a], "b must wait: only one slot"
    assert scheduler.commit(first, [99]) == [a]
    assert a.block_table is None, "a finished request gives its blocks back"

    assert scheduler.schedule().requests == [b], "b runs immediately, no idle iteration"


def test_a_long_prompt_is_split_into_chunks():
    """2000 tokens, 512 to an iteration: 512, 512, 512, 464."""
    scheduler = make_scheduler(max_batched_tokens=2048, chunk_size=512)
    request = make(2000, max_tokens=1)
    scheduler.add(request)

    counts = []
    while request.is_prefill():
        output = scheduler.schedule()
        counts.append(output.tokens_for(request))
        scheduler.commit(output)

    assert counts == [512, 512, 512, 464]


def test_a_chunk_is_bounded_by_budget_and_chunk_size():
    scheduler = make_scheduler(max_batched_tokens=100, chunk_size=512)
    scheduler.add(make(2000))
    assert scheduler.schedule().total_tokens == 100

    scheduler = make_scheduler(max_batched_tokens=2048, chunk_size=512)
    scheduler.add(make(2000))
    assert scheduler.schedule().total_tokens == 512, "the head-of-line stall is bounded"


def test_a_mid_prefill_chunk_takes_no_token():
    """Sampling a position in the middle of a prompt would invent a token."""
    scheduler = make_scheduler(chunk_size=4)
    request = make(10)
    scheduler.add(request)

    scheduler.commit(scheduler.schedule(), [777])

    assert request.output_token_ids == []
    assert request.num_computed_tokens == 4 and request.is_prefill()


def decoding_scheduler(count: int, **config) -> tuple[Scheduler, list[Request]]:
    """A scheduler with count requests past their prompts and into decode."""
    scheduler = make_scheduler(**config)
    requests = [make(2, max_tokens=8) for _ in range(count)]
    add_all(scheduler, requests)
    scheduler.commit(scheduler.schedule(), [5] * count)
    return scheduler, requests


def test_decodes_ride_along_with_a_prefill_chunk():
    """One pass carrying a 300-token chunk and three decodes: piggybacked decoding."""
    scheduler, decoders = decoding_scheduler(3, max_batched_tokens=1024, chunk_size=300)
    prompt = make(2000)
    scheduler.add(prompt)

    output = scheduler.schedule()

    assert [output.tokens_for(r) for r in decoders] == [1, 1, 1]
    assert output.tokens_for(prompt) == 300 and output.total_tokens == 303


def test_a_decode_is_never_stalled_by_a_long_prefill():
    """The budget is exactly one chunk, so the running prompt could take all of it: the
    decode goes first every iteration, and the chunk gets what is left."""
    scheduler, (decoder,) = decoding_scheduler(1, max_batched_tokens=512, chunk_size=512)
    prompt = make(2000, max_tokens=1)
    scheduler.add(prompt)

    for _ in range(3):
        output = scheduler.schedule()
        assert output.tokens_for(decoder) == 1, "the decode was left out of an iteration"
        assert output.tokens_for(prompt) == 511
        scheduler.commit(output, [5] * len(output.scheduled))
    assert len(decoder.output_token_ids) == 4


def test_the_chunk_takes_what_the_decodes_leave():
    """Decodes are scheduled first, so the chunk shrinks rather than the budget growing."""
    scheduler, _ = decoding_scheduler(4, max_batched_tokens=10, chunk_size=512)
    prompt = make(2000)
    scheduler.add(prompt)

    output = scheduler.schedule()

    assert output.total_tokens == 10 and output.tokens_for(prompt) == 6


def test_prefill_priority_runs_the_chunk_first():
    scheduler, decoders = decoding_scheduler(4, max_batched_tokens=10, chunk_size=512, prefill_priority=True)
    prompt = make(2000)
    scheduler.add(prompt)

    output = scheduler.schedule()

    assert output.tokens_for(prompt) == 10, "the prompt took the whole budget"
    assert all(output.tokens_for(r) == 0 for r in decoders)


def test_positions_resume_where_the_previous_chunk_stopped():
    """The chunk-boundary bug, caught in the metadata rather than in the logits."""
    scheduler = make_scheduler(max_batched_tokens=64, chunk_size=8)
    request = make(24)
    scheduler.add(request)

    seen = []
    while request.is_prefill():
        output = scheduler.schedule()
        seen.append(ForwardBatch.from_scheduled(output.scheduled, scheduler.manager).positions.tolist())
        scheduler.commit(output)

    assert seen == [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))]


def test_preemption_frees_now_and_requeues_at_the_front():
    scheduler = make_scheduler()
    request = make(4)
    scheduler.add(request)
    scheduler.commit(scheduler.schedule(), [5])

    scheduler.preempt(request)

    assert request.block_table is None and scheduler.manager.pool.num_free == 1024
    assert scheduler.waiting[0] is request and request.status is RequestStatus.WAITING
    assert scheduler.schedule().scheduled == [(request, 5)], "4 prompt + 1 emitted, recomputed"


def test_a_chunked_recompute_takes_no_token_until_it_catches_up():
    """A preempted request re-prefills its prompt and its output. A chunk that ends past the
    prompt but short of the output is still mid-recompute: its row predicts a token the
    request already has, so nothing may be appended until every token is computed again."""
    scheduler = make_scheduler(chunk_size=4)
    request = make(4, max_tokens=8)
    scheduler.add(request)
    for token in (11, 12, 13):
        scheduler.commit(scheduler.schedule(), [token])
    scheduler.preempt(request)

    first = scheduler.schedule()
    scheduler.commit(first, [777])
    assert first.tokens_for(request) == 4 and request.output_token_ids == [11, 12, 13]

    second = scheduler.schedule()
    scheduler.commit(second, [14])
    assert second.tokens_for(request) == 3 and request.output_token_ids == [11, 12, 13, 14]


def test_admission_waits_rather_than_preempting_for_a_new_request():
    """Memory pressure preempts for a request already in flight, and merely postpones one
    that has not started: a queued request holds nothing."""
    scheduler = make_scheduler(num_blocks=4, block_size=8)
    running, queued = make(24), make(16)
    add_all(scheduler, [running, queued])

    output = scheduler.schedule()

    assert output.requests == [running] and not output.preempted
    assert queued.status is RequestStatus.WAITING


def test_a_running_request_preempts_the_newest_to_grow():
    """Two decodes both about to cross a page boundary, one page free: the newer one goes."""
    scheduler = make_scheduler(num_blocks=3, block_size=4, max_sequences=2)
    older, newer = make(4, max_tokens=8), make(4, max_tokens=8)
    add_all(scheduler, [older, newer])
    scheduler.commit(scheduler.schedule(), [5, 5])  # 2 of 3 pages, both full

    output = scheduler.schedule()

    assert output.requests == [older] and output.preempted == [newer]
    assert scheduler.waiting[0] is newer


# --- token identity through the paged pool --------------------------------------------------


def run(scheduler: Scheduler, model, kv: PagedKvPool, max_iterations: int = 2000) -> int:
    """The engine loop: schedule, forward the ragged batch, sample each last row, commit.
    Returns how many requests were preempted along the way."""
    preempted = 0
    for _ in range(max_iterations):
        if not scheduler.has_work:
            return preempted
        output = scheduler.schedule()
        preempted += len(output.preempted)
        batch = ForwardBatch.from_scheduled(output.scheduled, scheduler.manager)
        logits = model(batch.input_ids[None], batch.positions, kv.caches(batch))[0, batch.last_rows]
        tokens = sample(logits, [r.sampling_params for r in output.requests]).tolist()
        # MLX is lazy: materialize the pages so the next step's graph does not reach back.
        mx.eval(kv.keys, kv.values)
        scheduler.commit(output, tokens)
    raise AssertionError("the scheduler stopped making progress")


def paged(tiny_qwen3, num_blocks=64, block_size=8, prefix_caching=False, **config):
    model = from_mlx(tiny_qwen3)
    c = model.config
    kv = PagedKvPool(c.num_hidden_layers, num_blocks, block_size, c.num_key_value_heads, c.head_dim,
                     dtype=mx.float32)
    scheduler = Scheduler(SchedulerConfig(**config), BlockManager(kv, prefix_caching))
    return model, kv, scheduler


def check_identity(tiny_qwen3, prompts, max_tokens, **setup) -> int:
    model, kv, scheduler = paged(tiny_qwen3, **setup)
    requests = [Request(prompt_token_ids=p, sampling_params=GREEDY, max_tokens=max_tokens) for p in prompts]
    add_all(scheduler, requests)

    preempted = run(scheduler, model, kv)

    for index, request in enumerate(requests):
        alone = generate_with_kv_cache(model, request.prompt_token_ids, max_tokens)
        assert request.output_token_ids == alone, f"prompt {index} changed"
    assert scheduler.manager.pool.num_free == kv.num_blocks, "blocks leaked"
    return preempted


PROMPTS = [[1, 2, 3], [5], [7] * 9, [2, 4], [9, 8, 7, 6, 5], [1]]


def test_a_batched_run_is_token_identical_to_each_request_alone(tiny_qwen3):
    check_identity(tiny_qwen3, PROMPTS, 6, max_batched_tokens=16, max_sequences=3)


def test_a_chunked_and_piggybacked_run_is_token_identical(tiny_qwen3):
    prompts = [mx.random.randint(0, 512, (n,)).tolist() for n in (33, 5, 17, 1, 40, 2)]
    check_identity(tiny_qwen3, prompts, 5, max_batched_tokens=12, max_sequences=4, chunk_size=8)


def test_a_small_pool_forces_preemption_and_changes_nothing(tiny_qwen3):
    """Recomputing a preempted request over its prompt and its own output must land on
    exactly the tokens it would have produced uninterrupted."""
    preempted = check_identity(tiny_qwen3, PROMPTS, 12, num_blocks=5, max_batched_tokens=16, max_sequences=6)
    assert preempted > 0, "this pool is too big to be testing preemption"


def test_prefix_caching_changes_nothing(tiny_qwen3):
    shared = mx.random.randint(0, 512, (20,)).tolist()
    prompts = [[*shared, tail] for tail in (1, 2, 3, 4)] + [shared]
    # One at a time, so each request finds the previous one's pages cached.
    check_identity(tiny_qwen3, prompts, 5, prefix_caching=True, max_sequences=1)


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_a_stress_run_completes_identically_and_leaks_nothing(tiny_qwen3, prefix_caching):
    """60 requests of mixed sizes, with repeats, through a pool that forces steady preemption."""
    rng = random.Random(0)
    pool_of_prompts = [[rng.randrange(512) for _ in range(rng.randrange(1, 30))] for _ in range(20)]
    prompts = [rng.choice(pool_of_prompts) for _ in range(60)]
    check_identity(tiny_qwen3, prompts, 4, num_blocks=12, prefix_caching=prefix_caching,
                   max_batched_tokens=32, max_sequences=6)
