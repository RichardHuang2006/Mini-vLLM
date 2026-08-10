"""Step 4.7 — the physical KV storage the block ids index into.

[§6.1](../DESIGN.md#61-physical-layout): one pre-allocated pool, partitioned into
fixed-size pages of `P` tokens.

::

    keys:   num_layers x num_blocks x P x H_k x D
    values: num_layers x num_blocks x P x H_k x D

Allocated **once**, at startup, and never grown. That is the whole point of paging as
a memory strategy: a `cudaMalloc` in the middle of a decode step would stall every
sequence in flight, and a pool that can be exhausted but not fragmented is a
scheduling problem rather than a memory-manager problem.

Two operations, and the asymmetry between them is the design:

* :meth:`write` scatters this iteration's new keys and values into whatever slots the
  block tables say, in one indexed copy per layer. Physical order is irrelevant.
* :meth:`gather` collects one sequence's cache back into a contiguous tensor, which is
  **not** what the engine wants to do — it is the slow, obviously-correct path that
  [Step 4.8]'s kernel is diffed against. Doing this gather for real would copy the
  entire cache every iteration and defeat the purpose.
"""

from __future__ import annotations

import torch

__all__ = ["PagedKvPool"]


class PagedKvPool:
    """Pre-allocated paged storage for every layer's keys and values.

    The layer axis is part of one tensor rather than a list of per-layer tensors so
    the whole cache is a single allocation whose size is knowable up front —
    :meth:`bytes_for` is what an engine uses to pick `num_blocks` from a memory
    budget, and it would not be answerable if layers were allocated separately.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        for name, value in (
            ("num_layers", num_layers),
            ("num_blocks", num_blocks),
            ("block_size", block_size),
            ("num_kv_heads", num_kv_heads),
            ("head_dim", head_dim),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if block_size & (block_size - 1):
            raise ValueError(f"block_size must be a power of two, got {block_size}")

        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.keys = torch.zeros(shape, dtype=dtype, device=self.device)
        self.values = torch.zeros(shape, dtype=dtype, device=self.device)

    # ------------------------------------------------------------------- sizing

    @staticmethod
    def bytes_for(
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> int:
        """How much GPU memory a pool of this shape would take, keys and values."""
        elements = num_layers * num_blocks * block_size * num_kv_heads * head_dim
        return 2 * elements * torch.empty((), dtype=dtype).element_size()

    @property
    def num_slots(self) -> int:
        """Token slots per layer — the range `physical_slot` addresses."""
        return self.num_blocks * self.block_size

    # -------------------------------------------------------------------- access

    def layer_keys(self, layer: int) -> torch.Tensor:
        """One layer's key pool, `num_blocks x P x H_k x D`. A view, not a copy."""
        return self.keys[layer]

    def layer_values(self, layer: int) -> torch.Tensor:
        return self.values[layer]

    def flat(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's pools flattened to `num_slots x H_k x D`.

        The layout `slot_mapping` indexes into: `block_id * P + offset` is a flat slot
        number precisely because the block and offset axes are adjacent and
        contiguous, so this view is free.
        """
        shape = (self.num_slots, self.num_kv_heads, self.head_dim)
        return self.keys[layer].view(shape), self.values[layer].view(shape)

    # --------------------------------------------------------------------- write

    def write(
        self,
        layer: int,
        slot_mapping: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Scatter this iteration's keys and values into their slots.

        ::

            slot_mapping: int32 [T]     where each token goes, from the block tables
            key, value:   [T, H_k, D]   flattened across sequences

        `T` spans the whole ragged batch — a prefill chunk's tokens and a dozen
        decode steps' single tokens in one call — because the slots are already
        absolute. Nothing here needs to know which sequence a token belonged to,
        which is exactly the property that makes one launch enough.
        """
        self._check_layer(layer)
        if key.shape != value.shape:
            raise ValueError(
                f"key and value must match, got {tuple(key.shape)} and {tuple(value.shape)}"
            )
        expected = (slot_mapping.shape[0], self.num_kv_heads, self.head_dim)
        if tuple(key.shape) != expected:
            raise ValueError(f"expected keys shaped {expected}, got {tuple(key.shape)}")
        if slot_mapping.numel() and int(slot_mapping.max()) >= self.num_slots:
            raise ValueError(
                f"slot {int(slot_mapping.max())} is outside a pool of {self.num_slots} slots"
            )
        if slot_mapping.numel() and int(slot_mapping.min()) < 0:
            raise ValueError(f"negative slot {int(slot_mapping.min())} in the mapping")

        flat_keys, flat_values = self.flat(layer)
        index = slot_mapping.to(device=flat_keys.device, dtype=torch.int64)
        flat_keys.index_copy_(0, index, key.to(flat_keys.dtype))
        flat_values.index_copy_(0, index, value.to(flat_values.dtype))

    # -------------------------------------------------------------------- gather

    def gather(
        self,
        layer: int,
        block_ids: tuple[int, ...] | list[int],
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One sequence's cache, contiguous in logical order.

        ::

            returns: 1 x H_k x num_tokens x D, keys and values

        The shape is the one Phase 1's attention takes, so this is what makes a paged
        cache testable against a dense one. It is a **copy** — the gather every real
        paged kernel exists to avoid — and is here only as the oracle.
        """
        self._check_layer(layer)
        needed = -(-num_tokens // self.block_size)  # ceiling division
        if needed > len(block_ids):
            raise ValueError(
                f"{num_tokens} tokens need {needed} blocks, but the table holds {len(block_ids)}"
            )

        index = torch.tensor(list(block_ids[:needed]), dtype=torch.int64, device=self.device)
        gathered = []
        for pool in (self.keys[layer], self.values[layer]):
            blocks = pool.index_select(0, index)  # needed x P x H_k x D
            flat = blocks.reshape(-1, self.num_kv_heads, self.head_dim)[:num_tokens]
            gathered.append(flat.permute(1, 0, 2).unsqueeze(0).contiguous())
        return gathered[0], gathered[1]

    # ---------------------------------------------------------------------- copy

    def copy_block(self, source: int, destination: int) -> None:
        """Duplicate one page across every layer — the copy in copy-on-write.

        Every layer at once, because a block id means the same page in all of them:
        the block table is per sequence, not per sequence and layer. Copying one
        layer's page and not the others would leave a sequence attending over a
        prefix that is correct in layer 0 and stale in layer 1, which reads as a
        model that has subtly forgotten its prompt.
        """
        for block_id in (source, destination):
            if not 0 <= block_id < self.num_blocks:
                raise ValueError(f"block {block_id} is outside a pool of {self.num_blocks}")
        if source == destination:
            return

        self.keys[:, destination].copy_(self.keys[:, source])
        self.values[:, destination].copy_(self.values[:, source])

    def _check_layer(self, layer: int) -> None:
        if not 0 <= layer < self.num_layers:
            raise ValueError(f"layer {layer} is outside a pool of {self.num_layers} layers")

    def __repr__(self) -> str:
        return (
            f"PagedKvPool(layers={self.num_layers}, blocks={self.num_blocks}, "
            f"block_size={self.block_size}, heads={self.num_kv_heads}, dim={self.head_dim}, "
            f"{self.dtype}, {self.device.type})"
        )
