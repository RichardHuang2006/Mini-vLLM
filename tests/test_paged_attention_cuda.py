"""Step 4.8 — paged attention kernels, against Step 4.7's dense-gather oracle.

The oracle copies each sequence's cache out of the pool and calls Phase 1's attention;
the kernel walks the block table inside its inner loop. Everything below is that
comparison, at the shapes where the two could differ:

* a **shuffled** block table, so physical order and logical order disagree. This is
  the test the step exists for — with a freshly allocated pool the two coincide, and
  an indexing bug looks correct.
* a **mixed** batch of prefill chunks and decode steps in one launch, where each row
  must equal what it produces alone.
* the **split** path, where a long context is divided across blocks and merged, plus
  the ragged case where one sequence's splits are all empty but the first.

The kernel is called directly rather than through the model, so nothing here depends
on how `ops.paged_attention` routes.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import OP_TOLERANCES, assert_allclose, assert_relative_error_below

from mini_vllm.kernels import ops
from mini_vllm.kernels.extension import load_extension
from mini_vllm.paged_attention import paged_attention_gathered

# Qwen3-0.6B's attention shape, scaled down where the test does not need the width.
HEADS, KV_HEADS, HEAD_DIM = 8, 4, 64
BLOCK_SIZE = 16

pytestmark = pytest.mark.cuda


@pytest.fixture
def extension(device):
    return load_extension()


class PagedCase:
    """A pool, a shuffled set of block tables, and the metadata for one iteration.

    Building this by hand rather than through `BlockManager` is deliberate: the kernel
    takes tensors, and a test that went through the manager would fail for two
    different reasons without saying which.
    """

    def __init__(
        self,
        contexts: list[int],
        query_lens: list[int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        shuffle: bool = True,
        block_size: int = BLOCK_SIZE,
        heads: int = HEADS,
        kv_heads: int = KV_HEADS,
        head_dim: int = HEAD_DIM,
        spare_blocks: int = 3,
    ) -> None:
        blocks_each = [-(-context // block_size) for context in contexts]
        total = sum(blocks_each) + spare_blocks

        generator = torch.Generator(device="cpu").manual_seed(sum(contexts) + len(contexts))
        self.keys = torch.randn(
            total, block_size, kv_heads, head_dim, generator=generator, dtype=torch.float32
        ).to(device=device, dtype=dtype)
        self.values = torch.randn(
            total, block_size, kv_heads, head_dim, generator=generator, dtype=torch.float32
        ).to(device=device, dtype=dtype)

        # A separate generator for the layout, so that shuffling changes *only* the
        # layout: drawing the permutation from `generator` would advance it and give
        # the shuffled and unshuffled cases different keys to compare.
        layout = torch.Generator(device="cpu").manual_seed(7)
        order = (
            torch.randperm(total, generator=layout) if shuffle else torch.arange(total)
        ).tolist()

        widest = max(blocks_each)
        rows, cursor = [], 0
        for count in blocks_each:
            row = order[cursor : cursor + count]
            cursor += count
            rows.append(row + [-1] * (widest - count))

        self.block_tables = torch.tensor(rows, dtype=torch.int32, device=device)
        self.context_lens = torch.tensor(contexts, dtype=torch.int32, device=device)
        self.seq_lens = torch.tensor(query_lens, dtype=torch.int32, device=device)
        offsets = [0]
        for length in query_lens:
            offsets.append(offsets[-1] + length)
        self.cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device=device)

        self.q = torch.randn(
            offsets[-1], heads, head_dim, generator=generator, dtype=torch.float32
        ).to(device=device, dtype=dtype)
        self.scale = 1.0 / math.sqrt(head_dim)
        self.max_query_len = max(query_lens)
        self.max_context_len = max(contexts)

    def kernel(self, extension) -> torch.Tensor:
        return extension.paged_attention(
            self.q,
            self.keys,
            self.values,
            self.block_tables,
            self.cu_seqlens_q,
            self.context_lens,
            self.seq_lens,
            self.max_query_len,
            self.max_context_len,
            self.scale,
        )

    def reference(self) -> torch.Tensor:
        return paged_attention_gathered(
            self.q,
            self.keys,
            self.values,
            self.block_tables,
            self.cu_seqlens_q,
            self.context_lens,
            self.scale,
        )


# --------------------------------------------------------------------- decode


@pytest.mark.parametrize("context", [1, 15, 16, 17, 63, 64, 65, 200, 1000])
def test_decode_matches_the_oracle_at_every_boundary(extension, device, context):
    """Block size, tile size, and their multiples: where an off-by-one would hide."""
    case = PagedCase([context], [1], device)
    assert_allclose(case.kernel(extension), case.reference(), msg=f"context {context}")


def test_decode_over_a_long_context_uses_the_split_path(extension, device):
    """8192 keys for one sequence is 16 blocks of work for 36 SMs without splitting.

    The result must not change, which is the whole claim of the split: the merge
    reapplies the same rescaling one level up.
    """
    case = PagedCase([8192], [1], device)
    assert_allclose(case.kernel(extension), case.reference())


def test_a_ragged_decode_batch_with_empty_splits(extension, device):
    """One long sequence beside short ones, so most of the short ones' splits get no
    keys at all. A split with nothing in it reports (-inf, 0) and must contribute
    exactly nothing rather than a NaN."""
    case = PagedCase([8192, 33, 4096, 17], [1, 1, 1, 1], device)
    got = case.kernel(extension)

    assert torch.isfinite(got).all(), "an empty split poisoned the merge"
    assert_allclose(got, case.reference())


@pytest.mark.parametrize("kv_heads", [1, 2, 8])
def test_every_group_size(extension, device, kv_heads):
    case = PagedCase([100], [1], device, heads=8, kv_heads=kv_heads)
    assert_allclose(case.kernel(extension), case.reference())


@pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
def test_every_head_dimension_the_kernel_claims(extension, device, head_dim):
    case = PagedCase([100], [1], device, head_dim=head_dim)
    assert_allclose(case.kernel(extension), case.reference())


@pytest.mark.parametrize("block_size", [1, 2, 16, 64])
def test_every_block_size(extension, device, block_size):
    """`P` is a power of two so the mapping is a shift and a mask; check the extremes."""
    case = PagedCase([100], [1], device, block_size=block_size)
    assert_allclose(case.kernel(extension), case.reference())


# -------------------------------------------------------------------- prefill


@pytest.mark.parametrize("length", [2, 8, 9, 16, 17, 64, 300])
def test_prefill_matches_the_oracle(extension, device, length):
    case = PagedCase([length], [length], device)
    assert_allclose(case.kernel(extension), case.reference(), msg=f"length {length}")


def test_a_chunk_sees_its_prefix_and_not_its_future(extension, device):
    """The chunked-prefill shape: 8 queries at the end of a 40-token context.

    The causal diagonal has to be shifted right by `S - L`. Both ways of getting this
    wrong are silent — too little context makes a chunk forget its prompt, too much
    lets it read ahead — so it is checked against the oracle's own offset mask.
    """
    case = PagedCase([40], [8], device)
    assert_allclose(case.kernel(extension), case.reference())


def test_the_first_query_of_a_prompt_returns_its_own_value(extension, device):
    """With one visible key, softmax is 1 and the output is that key's value exactly.

    An end-to-end check of the gather that does not depend on the oracle at all: if
    the block table is walked wrongly, this returns some other page's value.
    """
    case = PagedCase([12], [12], device, shuffle=True)
    got = case.kernel(extension)

    first_block = int(case.block_tables[0, 0])
    expected = case.values[first_block, 0]  # H_k x D, logical position 0
    for head in range(HEADS):
        assert_allclose(got[0, head], expected[head // (HEADS // KV_HEADS)])


def test_output_is_a_convex_combination_of_visible_values(extension, device):
    """Attention is an average, so no output can exceed the range of what it averaged.

    Catches a whole family of bugs the oracle comparison would also catch, but names
    them: a missing normalization, a negative weight, a mask applied after softmax.
    """
    case = PagedCase([64], [64], device)
    got = case.kernel(extension)

    # The pool is indexed by *block*, so a block id selects a whole page of P tokens.
    visible = torch.stack([case.values[int(b)] for b in case.block_tables[0]])
    lower, upper = visible.min(), visible.max()

    assert got.min() >= lower - 1e-4
    assert got.max() <= upper + 1e-4


# ---------------------------------------------------------------- mixed batch


def test_a_mixed_batch_gives_each_sequence_what_it_gets_alone(extension, device):
    """`[prefill(300), decode, decode, prefill(50)]` — the batch the plan asks for.

    One launch of each kernel covers it: a block whose sequence is in the other phase
    returns immediately. So the interesting failure is interference — a prefill block
    writing a decode row, or the merge overwriting a prefill answer.
    """
    contexts = [300, 700, 4096, 50]
    query_lens = [300, 1, 1, 50]
    case = PagedCase(contexts, query_lens, device)

    together = case.kernel(extension)

    for index in range(len(contexts)):
        start, end = int(case.cu_seqlens_q[index]), int(case.cu_seqlens_q[index + 1])
        alone = extension.paged_attention(
            case.q[start:end].contiguous(),
            case.keys,
            case.values,
            case.block_tables[index : index + 1].contiguous(),
            torch.tensor([0, end - start], dtype=torch.int32, device=device),
            case.context_lens[index : index + 1],
            case.seq_lens[index : index + 1],
            end - start,
            contexts[index],
            case.scale,
        )
        assert_allclose(together[start:end], alone, msg=f"sequence {index} changed in the batch")


def test_a_mixed_batch_matches_the_oracle(extension, device):
    case = PagedCase([300, 700, 4096, 50], [300, 1, 1, 50], device)
    assert_allclose(case.kernel(extension), case.reference())


def test_the_shuffle_is_what_is_being_tested(extension, device):
    """Same tokens, same logical order, two different physical layouts.

    If this passes only in the unshuffled case, the gather arithmetic is wrong and
    ordinary generation would still look fine — which is why the fixture shuffles by
    default.
    """
    ordered = PagedCase([200], [1], device, shuffle=False, spare_blocks=0)
    shuffled = PagedCase([200], [1], device, shuffle=True, spare_blocks=0)

    # The pools and queries are identical (same seed); only the tables differ.
    assert not torch.equal(ordered.block_tables, shuffled.block_tables)
    assert torch.equal(ordered.q, shuffled.q)

    # Each is correct for its own layout, which is the claim. They are *not* equal to
    # each other: a different table means each logical position holds a different key.
    assert_allclose(ordered.kernel(extension), ordered.reference())
    assert_allclose(shuffled.kernel(extension), shuffled.reference())


# ------------------------------------------------------------------- dtypes


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_stays_within_a_rounding(extension, device, dtype):
    """The kernel keeps fp32 accumulators to the store; the oracle rounds between ops.

    So they disagree by about one ULP of the storage type, with the kernel the more
    accurate of the two — the same relationship Step 3.4 measured. Compared by
    aggregate relative norm, because an elementwise bound at this magnitude mostly
    reports magnitude.
    """
    case = PagedCase([512], [1], device, dtype=dtype)
    got, expected = case.kernel(extension), case.reference()

    assert got.dtype is dtype
    assert_relative_error_below(got, expected, limit=OP_TOLERANCES[dtype])


# ------------------------------------------------------------------ refusals


def test_a_strided_pool_is_refused(extension, device):
    """The kernel walks the pool by slot arithmetic, so a narrowed view would have it
    reading the wrong pages rather than reading slowly."""
    case = PagedCase([64], [1], device)
    narrowed = case.keys.narrow(0, 0, case.keys.shape[0] - 1)[::2]

    with pytest.raises(RuntimeError, match="contiguous"):
        extension.paged_attention(
            case.q,
            narrowed,
            narrowed,
            case.block_tables,
            case.cu_seqlens_q,
            case.context_lens,
            case.seq_lens,
            1,
            64,
            case.scale,
        )


def test_a_non_power_of_two_block_size_is_refused(extension, device):
    case = PagedCase([64], [1], device)
    keys = case.keys.repeat(1, 3, 1, 1)[:, :3].contiguous()  # P = 3

    with pytest.raises(RuntimeError, match="power of two"):
        extension.paged_attention(
            case.q,
            keys,
            keys,
            case.block_tables,
            case.cu_seqlens_q,
            case.context_lens,
            case.seq_lens,
            1,
            64,
            case.scale,
        )


def test_int64_metadata_is_refused(extension, device):
    case = PagedCase([64], [1], device)

    with pytest.raises(RuntimeError, match="int32"):
        extension.paged_attention(
            case.q,
            case.keys,
            case.values,
            case.block_tables.long(),
            case.cu_seqlens_q,
            case.context_lens,
            case.seq_lens,
            1,
            64,
            case.scale,
        )


def test_a_padded_rectangle_is_refused(extension, device):
    """Query rows are `T x H_q x D`. Demanding `B x H x L x D` here would reintroduce
    the padding paging exists to avoid."""
    case = PagedCase([64], [1], device)

    with pytest.raises(RuntimeError, match="T x H_q x D"):
        extension.paged_attention(
            case.q.unsqueeze(0),
            case.keys,
            case.values,
            case.block_tables,
            case.cu_seqlens_q,
            case.context_lens,
            case.seq_lens,
            1,
            64,
            case.scale,
        )


def test_float64_is_refused(extension, device):
    case = PagedCase([64], [1], device, dtype=torch.float64)

    with pytest.raises(RuntimeError, match="float64"):
        case.kernel(extension)


def test_a_context_shorter_than_the_query_is_refused(extension, device):
    case = PagedCase([64], [1], device)

    with pytest.raises(RuntimeError, match="at least max_query_len"):
        extension.paged_attention(
            case.q,
            case.keys,
            case.values,
            case.block_tables,
            case.cu_seqlens_q,
            case.context_lens,
            case.seq_lens,
            8,  # claims 8 query rows...
            4,  # ...over a 4-token context
            case.scale,
        )


# ------------------------------------------------------------------ dispatch


def test_the_op_routes_to_the_kernel(extension, device, monkeypatch):
    case = PagedCase([256], [1], device)
    calls = []

    original = extension.paged_attention

    def spy(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(extension, "paged_attention", spy)

    got = ops.paged_attention(
        case.q,
        case.keys,
        case.values,
        case.block_tables,
        case.cu_seqlens_q,
        case.context_lens,
        case.seq_lens,
        1,
        256,
        case.scale,
        use_cuda=True,
    )

    assert len(calls) == 1, "use_cuda=True did not reach the kernel"
    assert_allclose(got, case.reference())


def test_the_op_falls_back_to_the_oracle_without_cuda_requested(extension, device):
    """`use_cuda=False` must still be correct, since it is the oracle the tests use."""
    case = PagedCase([256], [1], device)

    got = ops.paged_attention(
        case.q,
        case.keys,
        case.values,
        case.block_tables,
        case.cu_seqlens_q,
        case.context_lens,
        case.seq_lens,
        1,
        256,
        case.scale,
        use_cuda=False,
    )

    assert_allclose(got, case.reference())
