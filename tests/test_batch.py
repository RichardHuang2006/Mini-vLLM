"""batch.py: the ragged layout, built from scheduled requests and their block tables."""

from mini_vllm.batch import PADDING_BLOCK, ForwardBatch
from mini_vllm.paged_kv_cache import BlockManager, PagedKvPool
from mini_vllm.scheduler import Request


def make_manager(num_blocks=16, block_size=4) -> BlockManager:
    return BlockManager(PagedKvPool(1, num_blocks, block_size, 1, 1))


def build(manager, scheduled) -> ForwardBatch:
    for request, count in scheduled:
        manager.allocate(request, count)
    return ForwardBatch.from_scheduled(scheduled, manager)


def test_a_mixed_batch_flattens_onto_one_token_axis():
    manager = make_manager()
    decoding = Request(prompt_token_ids=[5, 6, 7], output_token_ids=[8])
    manager.allocate(decoding, 3)
    decoding.num_computed_tokens = 3
    fresh = Request(prompt_token_ids=[1, 2, 3, 4, 5])

    batch = build(manager, [(decoding, 1), (fresh, 5)])

    assert batch.input_ids.tolist() == [8, 1, 2, 3, 4, 5]
    assert batch.positions.tolist() == [3, 0, 1, 2, 3, 4], "each sequence has its own positions"
    assert batch.cu_seqlens_q.tolist() == [0, 1, 6]
    assert batch.context_lens.tolist() == [4, 5]
    assert batch.last_rows.tolist() == [0, 5]


def test_a_chunk_resumes_where_the_last_one_stopped():
    manager = make_manager()
    request = Request(prompt_token_ids=list(range(100, 110)))
    first = build(manager, [(request, 4)])
    request.num_computed_tokens = 4

    second = build(manager, [(request, 6)])

    assert first.slot_mapping.tolist() == [0, 1, 2, 3]
    assert second.slot_mapping.tolist() == [4, 5, 6, 7, 8, 9], "the second chunk rewrote the first"
    assert second.positions.tolist() == list(range(4, 10))
    assert second.input_ids.tolist() == list(range(104, 110))
    assert second.context_lens.tolist() == [10], "it attends over both chunks"


def test_tables_are_padded_to_the_widest():
    manager = make_manager()
    short, long = Request(prompt_token_ids=[1, 2]), Request(prompt_token_ids=list(range(9)))

    batch = build(manager, [(short, 2), (long, 9)])

    assert batch.block_tables.shape == (2, 3)
    assert batch.block_tables[0, 1:].tolist() == [PADDING_BLOCK, PADDING_BLOCK]
    assert batch.block_tables[1].tolist() == long.block_table.block_ids
