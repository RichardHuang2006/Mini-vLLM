"""Step 4.2 — forward batch metadata.

Every field here is an index into another, and the failure mode of a wrong index
is not an exception but a sequence attending over another sequence's tokens. So
the tests are mostly invariants, and they are checked on construction rather than
at use: this object is the entire contract between the scheduler and the GPU, and
a batch that violates it should not be constructible.

The example from the plan is spelled out in `test_the_worked_example`, which is the
one to read first.
"""

from __future__ import annotations

import pytest
import torch

from mini_vllm.sampler import SamplingParams
from mini_vllm.serve.batch import ForwardBatch
from mini_vllm.serve.sequence import Sequence


def make(prompt_len: int, computed: int = 0, output: int = 0, **kwargs) -> Sequence:
    """A sequence with `computed` tokens already through the model."""
    sequence = Sequence(prompt_token_ids=list(range(100, 100 + prompt_len)), **kwargs)
    for token in range(output):
        sequence.output_token_ids.append(900 + token)
    sequence.num_computed_tokens = computed
    return sequence


def batch_of(*pairs) -> ForwardBatch:
    return ForwardBatch.from_scheduled(pairs)


# ------------------------------------------------------------------ the example


def test_the_worked_example():
    """`[prefill(A, 300), decode(B, 1), decode(C, 1)]` from the plan.

    B has 511 tokens computed and one just sampled, so it attends over 512; C has
    46 computed and attends over 47. Nothing about the tensor shapes says any of
    that — it is entirely carried by `context_lens`, which is why the paged kernels
    take it as an argument rather than deriving it.
    """
    a = make(300)
    b = make(prompt_len=500, computed=511, output=12)
    c = make(prompt_len=40, computed=46, output=7)

    batch = ForwardBatch.from_scheduled([(a, 300), (b, 1), (c, 1)])

    assert batch.cu_seqlens_q.tolist() == [0, 300, 301, 302]
    assert batch.seq_lens.tolist() == [300, 1, 1]
    assert batch.context_lens.tolist() == [300, 512, 47]
    assert batch.total_tokens == 302
    assert batch.num_sequences == 3
    assert not batch.is_pure_decode
    assert batch.num_prefill_tokens == 300


# ------------------------------------------------------------------- the tokens


def test_tokens_are_the_uncomputed_ones_in_order():
    sequence = make(prompt_len=5, computed=2)

    batch = batch_of((sequence, 3))

    assert batch.input_ids.tolist() == [102, 103, 104]


def test_a_decode_step_feeds_the_token_just_sampled():
    """The one uncomputed token is the sampled one, not the last prompt token."""
    sequence = make(prompt_len=3, computed=3, output=1)

    batch = batch_of((sequence, 1))

    assert batch.input_ids.tolist() == [900]
    assert batch.context_lens.tolist() == [4]


def test_positions_resume_where_the_sequence_left_off():
    """Positions are absolute, so a chunk starting at 512 says 512.

    This is the payoff of Step 1.3's explicit positions. A kernel that inferred
    positions from `arange(L)` would rotate a resumed chunk as if it were the start
    of the prompt, and the result is a continuation that is fluent and wrong — the
    failure mode that motivated the whole chunk-boundary test in Step 4.4.
    """
    sequence = make(prompt_len=1000, computed=512)

    batch = batch_of((sequence, 8))

    assert batch.positions.tolist() == list(range(512, 520))


def test_positions_are_per_sequence_not_per_batch():
    """Each sequence restarts its own count, and the ragged axis does not."""
    batch = batch_of((make(3), 3), (make(prompt_len=2, computed=5, output=4), 1))

    assert batch.positions.tolist() == [0, 1, 2, 5]


def test_dtypes_are_what_the_kernels_expect():
    batch = batch_of((make(4), 4))

    assert batch.input_ids.dtype is torch.int64
    assert batch.positions.dtype is torch.int64
    assert batch.cu_seqlens_q.dtype is torch.int32
    assert batch.seq_lens.dtype is torch.int32
    assert batch.context_lens.dtype is torch.int32


# --------------------------------------------------------------- the invariants


def test_offsets_end_at_the_token_count():
    batch = batch_of((make(7), 7), (make(3), 3))

    assert int(batch.cu_seqlens_q[-1]) == batch.total_tokens == 10


def test_slice_of_recovers_each_sequence():
    a, b, c = make(4), make(1), make(6)

    batch = batch_of((a, 4), (b, 1), (c, 6))

    assert batch.input_ids[batch.slice_of(0)].tolist() == a.token_ids
    assert batch.input_ids[batch.slice_of(1)].tolist() == b.token_ids
    assert batch.input_ids[batch.slice_of(2)].tolist() == c.token_ids


def test_context_length_is_never_below_query_length():
    """`S >= L` is causality, not convention, so it is refused rather than fixed."""
    with pytest.raises(ValueError, match="must be >= "):
        ForwardBatch(
            input_ids=torch.zeros(4, dtype=torch.int64),
            positions=torch.arange(4),
            cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            context_lens=torch.tensor([3], dtype=torch.int32),
            seq_ids=(1,),
            sampling_params=(SamplingParams(),),
        )


def test_offsets_must_agree_with_seq_lens():
    """The invariant that catches an admitted sequence with no tokens appended."""
    with pytest.raises(ValueError, match="disagree with"):
        ForwardBatch(
            input_ids=torch.zeros(5, dtype=torch.int64),
            positions=torch.arange(5),
            cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
            seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            context_lens=torch.tensor([2, 2], dtype=torch.int32),
            seq_ids=(1, 2),
            sampling_params=(SamplingParams(), SamplingParams()),
        )


def test_offsets_must_cover_the_tokens():
    with pytest.raises(ValueError, match="but there are"):
        ForwardBatch(
            input_ids=torch.zeros(9, dtype=torch.int64),
            positions=torch.arange(9),
            cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            context_lens=torch.tensor([4], dtype=torch.int32),
            seq_ids=(1,),
            sampling_params=(SamplingParams(),),
        )


def test_positions_must_index_the_same_tokens():
    with pytest.raises(ValueError, match="index the same tokens"):
        ForwardBatch(
            input_ids=torch.zeros(4, dtype=torch.int64),
            positions=torch.arange(3),
            cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            context_lens=torch.tensor([4], dtype=torch.int32),
            seq_ids=(1,),
            sampling_params=(SamplingParams(),),
        )


def test_an_empty_batch_is_refused():
    """There is nothing to launch, and a zero-row forward pass is a scheduler bug."""
    with pytest.raises(ValueError, match="at least one sequence"):
        ForwardBatch.from_sequences([])


def test_sampling_params_must_cover_every_sequence():
    with pytest.raises(ValueError, match="sampling params for"):
        ForwardBatch(
            input_ids=torch.zeros(4, dtype=torch.int64),
            positions=torch.arange(4),
            cu_seqlens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
            seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            context_lens=torch.tensor([2, 2], dtype=torch.int32),
            seq_ids=(1, 2),
            sampling_params=(SamplingParams(),),
        )


# ------------------------------------------------------------------ the builders


def test_scheduling_beyond_a_sequence_is_refused():
    with pytest.raises(ValueError, match="cannot schedule"):
        batch_of((make(prompt_len=4, computed=2), 3))


def test_scheduling_no_tokens_is_refused():
    """An admitted sequence that contributes nothing is a budget bug, not a no-op."""
    with pytest.raises(ValueError, match="was scheduled 0 tokens"):
        batch_of((make(4), 0))


def test_from_sequences_takes_everything_uncomputed():
    batch = ForwardBatch.from_sequences([make(prompt_len=5, computed=2), make(3)])

    assert batch.seq_lens.tolist() == [3, 3]
    assert batch.context_lens.tolist() == [5, 3]


def test_pure_decode_is_recognized():
    batch = ForwardBatch.from_scheduled(
        [(make(prompt_len=3, computed=3, output=1), 1), (make(prompt_len=9, computed=9, output=1), 1)]
    )

    assert batch.is_pure_decode
    assert batch.num_prefill_tokens == 0


def test_sampling_params_come_from_the_sequences():
    """Per-row, because a batch is whatever arrived together and will not agree.

    The sampler in Step 1.10 is vectorized over exactly this: row 0 greedy while
    row 1 is at temperature 1.5, in one call.
    """
    greedy = make(2, sampling_params=SamplingParams(temperature=0.0))
    hot = make(2, sampling_params=SamplingParams(temperature=1.5))

    batch = batch_of((greedy, 2), (hot, 2))

    assert batch.sampling_params == (greedy.sampling_params, hot.sampling_params)
    assert batch.sampling_params[0].is_greedy and not batch.sampling_params[1].is_greedy


def test_describe_names_the_phases():
    batch = ForwardBatch.from_scheduled(
        [(make(300), 300), (make(prompt_len=9, computed=9, output=1), 1)]
    )

    described = batch.describe()

    assert "prefill 300" in described and "decode" in described
    assert "301 tokens" in described


def test_builds_on_the_device_asked_for():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    batch = ForwardBatch.from_sequences([make(4)], device="cuda")

    assert batch.input_ids.is_cuda and batch.cu_seqlens_q.is_cuda
