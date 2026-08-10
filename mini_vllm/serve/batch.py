"""Step 4.2 — the contract between the scheduler and the GPU.

One forward pass, one `ForwardBatch`. It describes a **ragged** batch: sequences of
different lengths, some prefilling and some decoding, flattened into a single token
axis with offsets rather than padded into a rectangle.

::

    for a mixed batch [prefill(A, 300), decode(B, 1), decode(C, 1)]:
      input_ids     300 + 1 + 1 = 302 tokens, concatenated
      cu_seqlens_q  [0, 300, 301, 302]   query-token offsets
      seq_lens      [300, 1, 1]          new tokens per sequence   (L)
      context_lens  [300, 512, 47]       total attended tokens     (S)

Why flattened and not padded: padding a 300-token prefill next to two decode steps
into a `3 x 300` rectangle is 598 wasted token-slots of compute, and the waste
grows with the length spread — the same argument as [§6](./DESIGN.md#6-paged-kv-cache)
makes about memory. The paged kernels in [Step 4.8] read `cu_seqlens_q` and
`context_lens` and serve the whole ragged batch in one launch with no host-side
per-sequence branching, so this object is shaped for them now rather than
reshaped later.

`seq_lens` and `context_lens` are separate for the same reason [Step 4.1] separates
`num_computed_tokens` from `len(sequence)`: `L` is how many tokens this pass
computes and `S` is how many it attends over. They differ whenever a prefix is
already cached, which is every decode step and every chunk after the first.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence as SequenceABC
from dataclasses import dataclass

import torch

from mini_vllm.sampler import SamplingParams
from mini_vllm.serve.sequence import Sequence

__all__ = ["ForwardBatch"]

# What an unused block-table entry holds. See the field's comment: a kernel bounded by
# `context_lens` never reads it, so a read is a bug worth making impossible to miss.
PADDING_BLOCK = -1


def _pad_tables(tables: list[tuple[int, ...]], device: torch.device | str) -> torch.Tensor:
    """Stack per-sequence block tables into one rectangle, right-padded.

    Rectangular because the kernel indexes it as `block_tables[seq, logical_block]`,
    and a ragged tensor of pointers would put a second indirection inside the inner
    loop. The width is the widest table in *this* batch, not the longest a sequence
    could ever be, so a batch of short sequences carries a small tensor.
    """
    widest = max((len(table) for table in tables), default=0)
    padded = [list(table) + [PADDING_BLOCK] * (widest - len(table)) for table in tables]
    return torch.tensor(padded, dtype=torch.int32, device=device).reshape(len(tables), widest)


@dataclass(frozen=True)
class ForwardBatch:
    """Everything the model and the kernels need for one ragged forward pass.

    A validated dataclass rather than a bag of tensors, because every field is an
    index into another one and the failure mode of getting that wrong is silently
    wrong text rather than an exception. The invariants are checked once, here, on
    construction — which is cheap next to a forward pass and is what makes the rest
    of Phase 4 debuggable.
    """

    input_ids: torch.Tensor  # int64 [total_tokens], flattened across sequences
    positions: torch.Tensor  # int64 [total_tokens], each sequence's own RoPE positions
    cu_seqlens_q: torch.Tensor  # int32 [num_sequences + 1], exclusive prefix sums
    seq_lens: torch.Tensor  # int32 [num_sequences], L per sequence
    context_lens: torch.Tensor  # int32 [num_sequences], S per sequence
    seq_ids: tuple[int, ...]
    sampling_params: tuple[SamplingParams, ...]

    # Paging ([Step 4.7]). Optional because the dense runner has no use for them and
    # the pure-scheduling tests should not have to build a block manager to construct
    # a batch.
    #
    #   slot_mapping  int32 [total_tokens]        where each new K/V is written
    #   block_tables  int32 [num_sequences, max]  right-padded with -1
    #
    # The padding value is -1 rather than 0 on purpose: a kernel bounds its walk by
    # `context_lens` and never reads the padding, so a read that *does* happen is a
    # bug, and -1 makes it a visibly impossible block id instead of a quiet read of
    # whatever sequence owns block 0.
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    def __post_init__(self) -> None:
        count = len(self.seq_ids)
        if not count:
            raise ValueError("a forward batch needs at least one sequence")

        for name, expected in (
            ("seq_lens", count),
            ("context_lens", count),
            ("cu_seqlens_q", count + 1),
        ):
            actual = getattr(self, name).shape[0]
            if actual != expected:
                raise ValueError(f"{name} has {actual} entries, expected {expected}")

        if len(self.sampling_params) != count:
            raise ValueError(
                f"got {len(self.sampling_params)} sampling params for {count} sequences"
            )
        if self.input_ids.shape != self.positions.shape:
            raise ValueError(
                f"input_ids is {tuple(self.input_ids.shape)} but positions is "
                f"{tuple(self.positions.shape)}; they index the same tokens"
            )

        # The offsets must actually describe the token axis they index into. This
        # is the invariant that catches a scheduler that admitted a sequence but
        # forgot to extend the flattened ids, which would otherwise read whatever
        # tokens happen to sit at the end of the batch.
        total = int(self.cu_seqlens_q[-1].item())
        if total != self.input_ids.shape[0]:
            raise ValueError(
                f"cu_seqlens_q ends at {total} but there are {self.input_ids.shape[0]} tokens"
            )
        if int(self.cu_seqlens_q[0].item()) != 0:
            raise ValueError("cu_seqlens_q must start at 0")

        lengths = self.cu_seqlens_q[1:] - self.cu_seqlens_q[:-1]
        if not torch.equal(lengths.to(self.seq_lens.dtype), self.seq_lens):
            raise ValueError(
                f"cu_seqlens_q differences {lengths.tolist()} disagree with "
                f"seq_lens {self.seq_lens.tolist()}"
            )

        # S >= L is not a convention, it is causality: a pass cannot compute more
        # tokens than it is allowed to attend over, and a kernel handed S < L would
        # mask every query in the overhang down to nothing.
        if bool((self.context_lens < self.seq_lens).any()):
            raise ValueError(
                f"context_lens {self.context_lens.tolist()} must be >= "
                f"seq_lens {self.seq_lens.tolist()} elementwise"
            )
        if bool((self.seq_lens < 1).any()):
            raise ValueError(f"every sequence must contribute a token, got {self.seq_lens.tolist()}")

        if self.slot_mapping is not None and self.slot_mapping.shape[0] != total:
            raise ValueError(
                f"slot_mapping has {self.slot_mapping.shape[0]} entries for {total} tokens; "
                "every token written this pass needs a slot"
            )
        if self.block_tables is not None:
            if self.block_tables.shape[0] != count:
                raise ValueError(
                    f"block_tables has {self.block_tables.shape[0]} rows for {count} sequences"
                )
            # Whether each row holds *enough* blocks for its context is not checkable
            # here — that needs the block size, which belongs to the manager — and the
            # attention path checks it against the pool it is about to read. What is
            # checkable is that a sequence with a context has any pages at all.
            empty = [i for i in range(count) if not bool((self.block_tables[i] >= 0).any())]
            if empty:
                raise ValueError(f"sequences {empty} attend over cached tokens with no blocks")

    # ---------------------------------------------------------------- properties

    @property
    def num_sequences(self) -> int:
        return len(self.seq_ids)

    @property
    def total_tokens(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def is_pure_decode(self) -> bool:
        """Every sequence contributes exactly one token.

        The common case by far, and the one worth knowing about: it means the batch
        is a rectangle after all, so the decode kernel's `B x H x 1 x D` shape
        applies without any ragged handling.
        """
        return bool((self.seq_lens == 1).all())

    @property
    def num_prefill_tokens(self) -> int:
        """Tokens belonging to sequences contributing more than one — the chunk work."""
        return int(self.seq_lens[self.seq_lens > 1].sum().item())

    def slice_of(self, index: int) -> slice:
        """Where sequence `index`'s tokens live in the flattened axis."""
        return slice(int(self.cu_seqlens_q[index]), int(self.cu_seqlens_q[index + 1]))

    def describe(self) -> str:
        parts = []
        for i, seq_id in enumerate(self.seq_ids):
            length = int(self.seq_lens[i])
            phase = "decode" if length == 1 else f"prefill {length}"
            parts.append(f"{seq_id}:{phase}/{int(self.context_lens[i])}")
        return f"batch[{self.total_tokens} tokens] " + " ".join(parts)

    # ------------------------------------------------------------------- builders

    @classmethod
    def from_scheduled(
        cls,
        scheduled: Iterable[tuple[Sequence, int]],
        device: torch.device | str = "cpu",
        manager: object | None = None,
    ) -> ForwardBatch:
        """Build from `(sequence, tokens to compute now)` pairs, in one pass.

        The token count is the scheduler's decision, not the sequence's: a 2000-token
        prompt admitted under a 512-token budget contributes 512 here and remembers
        the rest through `num_computed_tokens`. That is the whole of chunked prefill
        as far as this object is concerned ([Step 4.4] changes the caller, not this).

        Positions start at each sequence's `num_computed_tokens`, which is why
        [Step 1.3] insisted RoPE take an explicit position tensor: a chunk's tokens
        are at positions 512..1023, and nothing in the tensor shapes says so.

        Pass `manager` (a `BlockManager`) to fill in the paging metadata as well. It
        is optional so that a scheduling test can build a batch without a pool, and
        typed loosely to keep `serve` from importing `block` — the dependency runs the
        other way, since the manager needs `Sequence`.
        """
        input_ids: list[int] = []
        positions: list[int] = []
        offsets = [0]
        seq_lens: list[int] = []
        context_lens: list[int] = []
        seq_ids: list[int] = []
        params: list[SamplingParams] = []
        slots: list[int] = []
        tables: list[tuple[int, ...]] = []

        for sequence, count in scheduled:
            if count < 1:
                raise ValueError(f"sequence {sequence.seq_id} was scheduled {count} tokens")
            start = sequence.num_computed_tokens
            if start + count > len(sequence):
                raise ValueError(
                    f"sequence {sequence.seq_id} has {len(sequence)} tokens; cannot schedule "
                    f"{count} starting at {start}"
                )

            tokens = sequence.token_ids
            input_ids.extend(tokens[start : start + count])
            positions.extend(range(start, start + count))
            offsets.append(offsets[-1] + count)
            seq_lens.append(count)
            context_lens.append(start + count)
            seq_ids.append(sequence.seq_id)
            params.append(sequence.sampling_params)

            if manager is not None:
                table = manager.table(sequence)
                slots.extend(int(slot) for slot in manager.slot_mapping(sequence, count))
                tables.append(table.block_ids)

        as_int64 = {"dtype": torch.int64, "device": device}
        as_int32 = {"dtype": torch.int32, "device": device}
        return cls(
            input_ids=torch.tensor(input_ids, **as_int64),
            positions=torch.tensor(positions, **as_int64),
            cu_seqlens_q=torch.tensor(offsets, **as_int32),
            seq_lens=torch.tensor(seq_lens, **as_int32),
            context_lens=torch.tensor(context_lens, **as_int32),
            seq_ids=tuple(seq_ids),
            sampling_params=tuple(params),
            slot_mapping=None if manager is None else torch.tensor(slots, **as_int32),
            block_tables=None if manager is None else _pad_tables(tables, device),
        )

    @classmethod
    def from_sequences(
        cls,
        sequences: SequenceABC[Sequence],
        device: torch.device | str = "cpu",
    ) -> ForwardBatch:
        """Every sequence contributes all of its uncomputed tokens. No chunking."""
        return cls.from_scheduled(
            ((sequence, sequence.num_uncomputed_tokens) for sequence in sequences), device
        )
