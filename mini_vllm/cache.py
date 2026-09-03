"""The KV-cache memory hierarchy: dense -> paged -> shared -> quantized.

Six layers, in file order: DenseKvCache, BlockPool, BlockTable, PagedKvPool,
PrefixCache, and BlockManager, which ties the other five together for
scheduler.py. Every physical block is accounted for at all times -- on the free
list at reference count zero, or held by a block table at a positive one -- and
check_no_leaks asserts it after every test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    # `cache.py` sits below `scheduler.py` in the module graph, but the manager's API
    # takes the scheduler's `Sequence`. At runtime only attributes are read
    # (`block_table`, `token_ids`, `num_computed_tokens`), so the import is for type
    # checkers only and the graph stays acyclic.
    from mini_vllm.scheduler import Sequence

__all__ = [
    "KvCache",
    "DenseKvCache",
    "Block",
    "BlockPool",
    "BlockPoolError",
    "OutOfBlocks",
    "BlockTable",
    "PagedKvPool",
    "RadixNode",
    "PrefixCache",
    "BlockManager",
]


# --- 1. Dense KV cache -------------------------------------------------------

class KvCache(ABC):
    """One layer's worth of cached keys and values. Attention at position t needs the
    keys and values of every position 0..t and those never change, so caching them
    turns a quadratic decode into a linear one. The interface is one method, so the
    paged implementation is indistinguishable to the model.
    """

    @property
    @abstractmethod
    def offset(self) -> int:
        """How many positions are currently cached."""

    @abstractmethod
    def update_and_fetch(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Append key/value [B, H_k, L, D], returning the full [B, H_k, S, D] pair plus the
        write offset, the length before this call. The caller builds a causal mask from
        that offset, and being off by L there lets a token attend to its own future.
        Keys must already have RoPE applied, which is what makes an entry reusable.
        """

    @abstractmethod
    def reset(self) -> None:
        """Forget everything, so the cache can serve a new sequence."""


class DenseKvCache(KvCache):
    """A cache that simply concatenates along the sequence dimension.

    The obvious implementation, with the two costs the rest of this file removes.
    Every decode step reallocates, since torch.cat cannot extend in place: appending
    one token to a cache of S copies all S positions. And one contiguous allocation
    per sequence means a batch pads to the longest, and a sequence that might reach
    40960 tokens must be budgeted as if it will. It is correct, which makes it the
    oracle for the paged version.
    """

    def __init__(self) -> None:
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self._offset = 0

    @property
    def offset(self) -> int:
        return self._offset

    def update_and_fetch(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                f"expected B x H_k x L x D keys and values, got {tuple(key.shape)} "
                f"and {tuple(value.shape)}"
            )
        if key.shape != value.shape:
            raise ValueError(
                f"key and value must have the same shape, got {tuple(key.shape)} "
                f"and {tuple(value.shape)}"
            )

        written_at = self._offset

        if self.keys is None:
            self.keys, self.values = key, value
        else:
            if key.shape[:2] != self.keys.shape[:2] or key.shape[3] != self.keys.shape[3]:
                raise ValueError(
                    f"cannot append {tuple(key.shape)} to a cache of "
                    f"{tuple(self.keys.shape)}: only the sequence dimension may differ"
                )
            self.keys = torch.cat([self.keys, key], dim=-2)
            self.values = torch.cat([self.values, value], dim=-2)

        self._offset += key.shape[-2]
        return self.keys, self.values, written_at

    def reset(self) -> None:
        self.keys = None
        self.values = None
        self._offset = 0


# --- 2. Physical block pool --------------------------------------------------

class BlockPoolError(RuntimeError):
    """Base for block pool misuse. These indicate bugs, not conditions."""


class OutOfBlocks(BlockPoolError):
    """The pool is exhausted. Unlike the other errors here this is an expected runtime
    condition rather than a bug: it signals to preempt or to leave a request
    waiting. The scheduler catches it on its own.
    """


class Block(NamedTuple):
    """A read-only view of one block's state, for tests and debugging. The pool stores
    a flat list of counts instead, since it holds tens of thousands of blocks and an
    object per block would be pure overhead.
    """

    block_id: int
    ref_count: int

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


class BlockPool:
    """A fixed pool of physical blocks, handed out by id and refcounted.

        allocate() -> id        take a free block, at ref_count 1
        incref(id)              add a holder (fork, prefix reuse)
        decref(id) -> bool      drop a holder; True if that freed the block

    The bottom of the hierarchy: it knows nothing about sequences, tokens, or
    tensors, so block tables, copy-on-write and admission control all build on those
    two operations without a GPU to test. The free list is FIFO, not LIFO --
    recycling the most recently freed block has better locality but makes a
    use-after-free read back what it just released and look correct.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        self._num_blocks = num_blocks
        self._ref_counts = [0] * num_blocks
        self._free: deque[int] = deque(range(num_blocks))

        # Prefix caching, inert unless a manager wires it up. `_cached[id]` is True while
        # a block is registered in the prefix tree, free or held, since a held block
        # stays matchable; `on_evict` is how the pool tells the tree to release a block
        # it is about to repurpose.
        self._cached = [False] * num_blocks
        self.on_evict: Callable[[int], None] | None = None

    # ------------------------------------------------------------ inspection
    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_allocated(self) -> int:
        return self._num_blocks - len(self._free)

    def ref_count(self, block_id: int) -> int:
        self._check_id(block_id)
        return self._ref_counts[block_id]

    def block(self, block_id: int) -> Block:
        return Block(block_id, self.ref_count(block_id))

    def allocated_ids(self) -> list[int]:
        """Ids currently held by someone. Sorted, so output is reproducible."""
        return [i for i, count in enumerate(self._ref_counts) if count]

    # ------------------------------------------------------------ allocation
    def allocate(self) -> int:
        """Take one block from the free list at reference count 1. Raises OutOfBlocks if
        nothing is free; the caller should preempt or stop admitting rather than retry.
        """
        if not self._free:
            raise OutOfBlocks(
                f"all {self._num_blocks} blocks are in use; "
                "the caller should preempt or stop admitting rather than retry"
            )

        block_id = self._free.popleft()
        # A block carrying cached KV is being repurposed: unlink it from the prefix tree
        # first, so a later match cannot return a page now holding another sequence's
        # tokens. FIFO over the free list is LRU over cache entries, so the free list
        # already supplies the right eviction order.
        if self._cached[block_id]:
            if self.on_evict is not None:
                self.on_evict(block_id)
            self._cached[block_id] = False
        self._ref_counts[block_id] = 1
        return block_id

    def allocate_many(self, count: int) -> list[int]:
        """Take count blocks, or none at all. A half-allocated sequence forces the caller
        to unwind, and the unwind path is where block leaks come from.
        """
        if count < 0:
            raise ValueError(f"cannot allocate {count} blocks")
        if count > self.num_free:
            raise OutOfBlocks(
                f"asked for {count} blocks with {self.num_free} free of {self._num_blocks}"
            )
        return [self.allocate() for _ in range(count)]

    def incref(self, block_id: int) -> int:
        """Add a holder. Used by fork and by prefix-cache hits."""
        self._check_id(block_id)
        if self._ref_counts[block_id] == 0:
            raise BlockPoolError(
                f"block {block_id} is free; it must be allocated before it can be shared"
            )

        self._ref_counts[block_id] += 1
        return self._ref_counts[block_id]

    def decref(self, block_id: int) -> bool:
        """Drop a holder, returning True if this call freed the block. Copy-on-write needs
        the distinction: a write to a block with holders left must copy, a write to one
        just released need not.
        """
        self._check_id(block_id)
        if self._ref_counts[block_id] == 0:
            raise BlockPoolError(
                f"block {block_id} is already free (double free); some holder released it twice"
            )

        self._ref_counts[block_id] -= 1
        if self._ref_counts[block_id]:
            return False

        self._free.append(block_id)
        return True

    def decref_many(self, block_ids: list[int]) -> int:
        """Drop a holder on each, returning how many blocks that freed."""
        return sum(self.decref(block_id) for block_id in block_ids)

    # ---------------------------------------------------------- prefix cache
    def mark_cached(self, block_id: int) -> None:
        """Record that a block is now registered in the prefix tree. The reference count is
        unchanged -- caching does not add a holder, it makes the page matchable while
        still reclaimable.
        """
        self._check_id(block_id)
        self._cached[block_id] = True

    def is_cached(self, block_id: int) -> bool:
        self._check_id(block_id)
        return self._cached[block_id]

    def acquire_cached(self, block_id: int) -> None:
        """Take a matched block for a new holder, from wherever it currently sits.

        A prefix-cache hit is not a fresh allocation: the block exists and already holds
        the right KV. One at reference count zero comes off the free list here, one held
        by a running sequence is a plain incref, and either way it stays matchable.
        """
        self._check_id(block_id)
        if self._ref_counts[block_id] == 0:
            # A cached free block: pull it off the free list by hand. It stays cached,
            # so a later reuse of the same page still evicts its node.
            self._free.remove(block_id)
            self._ref_counts[block_id] = 1
        else:
            self._ref_counts[block_id] += 1

    def is_free_cached(self, block_id: int) -> bool:
        """A cached block at reference count zero. Reusing it consumes a free page while
        matching a held one does not, a distinction admission control accounts for."""
        self._check_id(block_id)
        return self._cached[block_id] and self._ref_counts[block_id] == 0

    # ----------------------------------------------------------- consistency
    def check_consistency(self) -> None:
        """Assert the free list and the reference counts still agree. Cheap enough for test
        teardown, where it attributes a refcount bug to the test that introduced it
        rather than to an out-of-memory thousands of iterations later.
        """
        free = list(self._free)

        if len(set(free)) != len(free):
            duplicates = sorted({i for i in free if free.count(i) > 1})
            raise BlockPoolError(f"blocks appear twice in the free list: {duplicates}")

        expected_free = {i for i, count in enumerate(self._ref_counts) if count == 0}
        if set(free) != expected_free:
            raise BlockPoolError(
                f"free list {sorted(free)} disagrees with the zero-refcount blocks "
                f"{sorted(expected_free)}"
            )

        negative = [i for i, count in enumerate(self._ref_counts) if count < 0]
        if negative:
            raise BlockPoolError(f"negative reference counts at blocks {negative}")

    def _check_id(self, block_id: int) -> None:
        if not 0 <= block_id < self._num_blocks:
            raise BlockPoolError(
                f"block id {block_id} out of range for a pool of {self._num_blocks}"
            )

    def __repr__(self) -> str:
        return (
            f"BlockPool(num_blocks={self._num_blocks}, "
            f"free={self.num_free}, allocated={self.num_allocated})"
        )


# --- 3. Logical block table --------------------------------------------------

class BlockTable:
    """The physical block ids backing one sequence, in logical order.

    The whole paging indirection: a list of block ids, a shift, and a mask.
    Non-contiguous storage removes per-sequence reservation and its fragmentation
    for one division and one modulo per token, both bit operations because P is a
    power of two. num_slots is capacity and num_tokens occupancy; their difference
    is the internal fragmentation in the final block, bounded by P - 1 tokens per
    sequence however long it gets.
    """

    def __init__(
        self,
        block_size: int,
        block_ids: Iterable[int] | None = None,
        num_tokens: int = 0,
    ) -> None:
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError(f"block_size must be a positive power of two, got {block_size}")

        self._block_size = block_size
        # Power of two, so `p // block_size` is `p >> shift` and `p % block_size` is
        # `p & mask`. The only reason the block size is constrained.
        self._shift = block_size.bit_length() - 1
        self._mask = block_size - 1

        self._block_ids: list[int] = [] if block_ids is None else list(block_ids)
        for block_id in self._block_ids:
            if block_id < 0:
                raise ValueError(f"block ids must be non-negative, got {block_id}")

        if not 0 <= num_tokens <= self.num_slots:
            raise ValueError(
                f"num_tokens={num_tokens} does not fit in {len(self._block_ids)} "
                f"blocks of {block_size} ({self.num_slots} slots)"
            )
        self._num_tokens = num_tokens

    # -------------------------------------------------------------- geometry
    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def block_ids(self) -> tuple[int, ...]:
        """The table itself, logical order first. A copy, so callers cannot alias it."""
        return tuple(self._block_ids)

    @property
    def num_blocks(self) -> int:
        return len(self._block_ids)

    @property
    def num_slots(self) -> int:
        """Token capacity of this sequence. Distinct from the range of physical_slot, which
        indexes the pool's whole flat cache: a one-block table holding block id 900 has
        16 slots but addresses slot 14400.
        """
        return len(self._block_ids) * self._block_size

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def num_empty_slots(self) -> int:
        """Room left in the final block. Zero when the last block is full."""
        return self.num_slots - self._num_tokens

    def blocks_needed_for(self, num_new_tokens: int) -> int:
        """How many fresh blocks appending num_new_tokens would require. Space left in the
        partial block is used first, so this is often zero -- which is why a decode step
        usually allocates nothing.
        """
        if num_new_tokens < 0:
            raise ValueError(f"num_new_tokens must be non-negative, got {num_new_tokens}")

        deficit = num_new_tokens - self.num_empty_slots
        if deficit <= 0:
            return 0
        return (deficit + self._mask) >> self._shift

    # ---------------------------------------------------------------- growth
    def append_block(self, block_id: int) -> None:
        """Extend capacity by one block. The id comes from the pool."""
        if block_id < 0:
            raise ValueError(f"block ids must be non-negative, got {block_id}")
        self._block_ids.append(block_id)

    def append_tokens(self, count: int = 1) -> None:
        """Mark count more slots occupied. Capacity first, occupancy second: this raises
        rather than growing the table, since only the block manager may take blocks from
        the pool. A self-allocating table would be a second place blocks can leak.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count > self.num_empty_slots:
            raise ValueError(
                f"cannot append {count} tokens: {self.num_empty_slots} free slots in "
                f"{self.num_blocks} blocks; append_block first"
            )
        self._num_tokens += count

    def trim_tokens(self, count: int = 1) -> int:
        """Give back count occupied slots, returning how many blocks fell empty.

        One caller: speculative decoding writes k proposed tokens' KV before knowing
        whether the target accepts them. Only occupancy moves -- emptied blocks are
        reported rather than released, so exactly one place frees a block. KV in the
        trimmed slots is left in place, since nothing reads past num_tokens.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count > self._num_tokens:
            raise ValueError(f"cannot trim {count} tokens from a table holding {self._num_tokens}")

        blocks_before = self.num_blocks
        self._num_tokens -= count
        # Blocks still needed for what is left: the empty tail is everything past it.
        blocks_still_used = -(-self._num_tokens // self._block_size)
        return blocks_before - max(blocks_still_used, 0)

    def drop_last_block(self) -> int:
        """Remove the final block from the table and return its id, for the manager to
        decref. Raises if the block still holds tokens, since dropping an occupied block
        would unmap KV the sequence is still attending over.
        """
        if not self._block_ids:
            raise IndexError("no blocks to drop")
        if self._num_tokens > (self.num_blocks - 1) * self._block_size:
            raise ValueError(
                f"the last block still holds tokens: {self._num_tokens} tokens in "
                f"{self.num_blocks} blocks of {self._block_size}"
            )
        return self._block_ids.pop()

    def replace_block(self, index: int, block_id: int) -> int:
        """Repoint one table entry, returning the id it displaced. The copy-on-write step.
        The displaced id is returned rather than dropped so the caller cannot forget to
        decref it; a forgotten decref surfaces only as an out-of-memory much later.
        """
        if not 0 <= index < len(self._block_ids):
            raise IndexError(f"block index {index} out of range for {len(self._block_ids)} blocks")
        if block_id < 0:
            raise ValueError(f"block ids must be non-negative, got {block_id}")

        displaced = self._block_ids[index]
        self._block_ids[index] = block_id
        return displaced

    def copy(self) -> BlockTable:
        """An independent table over the same blocks: the fork half of copy-on-write. No
        block data is copied and no refcounts are touched; the caller increfs. The list
        is independent, so appending to one branch cannot alter the other's mapping.
        """
        return BlockTable(self._block_size, self._block_ids, self._num_tokens)

    # ------------------------------------------------------- the indirection
    def block_index(self, position: int) -> int:
        """Which entry of this table covers `position`."""
        self._check_position(position)
        return position >> self._shift

    def block_offset(self, position: int) -> int:
        """Where inside its block `position` sits."""
        self._check_position(position)
        return position & self._mask

    def physical_slot(self, position: int) -> int:
        """Flat index of a logical position in the pool's cache: block_id * block_size +
        offset, written as a shift and an or because the offset is strictly narrower.
        """
        self._check_position(position)
        block_id = self._block_ids[position >> self._shift]
        return (block_id << self._shift) | (position & self._mask)

    def slots(self, positions: Iterable[int]) -> list[int]:
        """The physical slots of positions, as plain integers -- the form the engine needs
        to concatenate several sequences' slots into one vector per iteration. Tensors
        would have to be read back off the device to do that, one synchronization per
        sequence per iteration, for numbers that were already Python integers.
        """
        return [self.physical_slot(position) for position in positions]

    def slot_mapping(
        self,
        positions: Iterable[int],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """The int32 slot vector the paged write path scatters through: one entry per token
        written this iteration, the whole chunk for a prefill and a single element for a
        decode. int32 is what the kernel indexes with, and a pool large enough to
        overflow it would not fit in any current GPU.
        """
        return torch.tensor(self.slots(positions), dtype=torch.int32, device=device)

    def _check_position(self, position: int) -> None:
        # Checked against occupancy rather than capacity: a slot that exists but holds
        # no token is not addressable, so a scheduler off-by-one fails here instead of
        # reading whichever sequence used the slot last.
        if not 0 <= position < self._num_tokens:
            raise IndexError(
                f"position {position} out of range for a sequence of {self._num_tokens} tokens"
            )

    def __repr__(self) -> str:
        return (
            f"BlockTable(block_size={self._block_size}, block_ids={self._block_ids}, "
            f"num_tokens={self._num_tokens}/{self.num_slots})"
        )


# --- 4. Paged KV storage -----------------------------------------------------

class PagedKvPool:
    """Pre-allocated paged storage for every layer's keys and values, each
    num_layers x num_blocks x P x H_k x D.

    Allocated once at startup and never grown: a cudaMalloc mid-decode would stall
    every sequence in flight. The layer axis is part of one tensor rather than a
    list, so the whole cache is a single allocation whose size is knowable up front
    -- which is what makes bytes_for answerable, and how an engine picks num_blocks
    from a memory budget.
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
        kv_dtype: torch.dtype | None = None,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
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
        # `dtype` is the activation dtype: what keys and values arrive as and what a
        # gather returns. `kv_dtype` is the storage dtype, identical unless the cache is
        # quantized. Separating them is what FP8 amounts to here: the model still
        # computes in bf16 and only the resident cache shrinks.
        self.dtype = dtype
        self.kv_dtype = kv_dtype or dtype
        self.is_fp8 = self.kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        # Static scales, one per tensor. A key or value is divided by its scale before
        # the cast down and multiplied back after, which is how e4m3's ±448 range is
        # made to cover activations outside it. Qwen3's post-norm, post-RoPE keys are
        # near unit scale, so 1.0 is a safe default; the hook exists for models whose
        # are not.
        self.k_scale = float(k_scale)
        self.v_scale = float(v_scale)
        self.device = torch.device(device)

        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.keys = torch.zeros(shape, dtype=self.kv_dtype, device=self.device)
        self.values = torch.zeros(shape, dtype=self.kv_dtype, device=self.device)

    # ---------------------------------------------------------------- sizing
    @staticmethod
    def bytes_for(
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> int:
        """How much GPU memory a pool of this shape would take, keys and values. dtype is
        the storage dtype: pass torch.float8_e4m3fn to size an FP8 pool, where a page of
        the same geometry costs half as much and the engine fits twice as many.
        """
        elements = num_layers * num_blocks * block_size * num_kv_heads * head_dim
        return 2 * elements * torch.empty((), dtype=dtype).element_size()

    @property
    def num_slots(self) -> int:
        """Token slots per layer: the range `physical_slot` addresses."""
        return self.num_blocks * self.block_size

    # ---------------------------------------------------------------- access
    def layer_keys(self, layer: int) -> torch.Tensor:
        """One layer's key pool, `num_blocks x P x H_k x D`. A view, not a copy."""
        return self.keys[layer]

    def layer_values(self, layer: int) -> torch.Tensor:
        return self.values[layer]

    def flat(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's pools flattened to num_slots x H_k x D, the layout slot_mapping
        indexes into. The view is free: block_id * P + offset is a flat slot number
        because the block and offset axes are adjacent and contiguous.
        """
        shape = (self.num_slots, self.num_kv_heads, self.head_dim)
        return self.keys[layer].view(shape), self.values[layer].view(shape)

    # ----------------------------------------------------------------- write
    def write(
        self,
        layer: int,
        slot_mapping: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Scatter this iteration's key/value [T, H_k, D] into the slots slot_mapping names.

        T spans the whole ragged batch because the slots are already absolute; nothing
        here needs to know which sequence a token belongs to, which is what makes one
        launch sufficient. Slot values are not checked: this is the innermost call in
        the engine, and slot_mapping.max() on a CUDA tensor waits for everything queued
        on the stream -- two per layer measured 7 ms per iteration, a third of the
        model's time, to re-check integers already bounds-checked on the host.
        """
        self._check_layer(layer)
        if key.shape != value.shape:
            raise ValueError(
                f"key and value must match, got {tuple(key.shape)} and {tuple(value.shape)}"
            )
        expected = (slot_mapping.shape[0], self.num_kv_heads, self.head_dim)
        if tuple(key.shape) != expected:
            raise ValueError(f"expected keys shaped {expected}, got {tuple(key.shape)}")

        flat_keys, flat_values = self.flat(layer)
        if self.is_fp8:
            # The fused kernel quantizes and scatters in one pass; off the GPU it falls
            # back to the two-pass PyTorch that serves as its oracle. Either way the
            # trailing dimensions must match the pool's, which the reshape enforces.
            from mini_vllm import kernels

            kernels.quantize_scatter(
                key.reshape(-1, self.num_kv_heads, self.head_dim),
                value.reshape(-1, self.num_kv_heads, self.head_dim),
                flat_keys,
                flat_values,
                slot_mapping,
                self.k_scale,
                self.v_scale,
                use_cuda=self.device.type == "cuda",
            )
        else:
            index = slot_mapping.to(device=flat_keys.device, dtype=torch.int64)
            flat_keys.index_copy_(0, index, key.to(flat_keys.dtype))
            flat_values.index_copy_(0, index, value.to(flat_values.dtype))

    # ---------------------------------------------------------------- gather
    def gather(
        self,
        layer: int,
        block_ids: tuple[int, ...] | list[int],
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One sequence's cache, contiguous as 1 x H_k x num_tokens x D keys and values:
        the shape the reference attention takes, which is what makes a paged cache
        testable against a dense one. It is a copy -- the gather the paged kernels exist
        to avoid -- and is present only as the oracle.
        """
        self._check_layer(layer)
        needed = -(-num_tokens // self.block_size)  # ceiling division
        if needed > len(block_ids):
            raise ValueError(
                f"{num_tokens} tokens need {needed} blocks, but the table holds {len(block_ids)}"
            )

        index = torch.tensor(list(block_ids[:needed]), dtype=torch.int64, device=self.device)
        gathered = []
        for pool, scale in ((self.keys[layer], self.k_scale), (self.values[layer], self.v_scale)):
            blocks = pool.index_select(0, index)  # needed x P x H_k x D
            flat = blocks.reshape(-1, self.num_kv_heads, self.head_dim)[:num_tokens]
            contiguous = flat.permute(1, 0, 2).unsqueeze(0).contiguous()
            if self.is_fp8:
                # Dequantize back to the activation dtype: the oracle attention runs in
                # the model's precision, so the cast and the scale are undone here rather
                # than leaving FP8 in the math.
                contiguous = (contiguous.float() * scale).to(self.dtype)
            gathered.append(contiguous)
        return gathered[0], gathered[1]

    # ------------------------------------------------------------------ copy
    def copy_block(self, source: int, destination: int) -> None:
        """Duplicate one page across every layer: the copy in copy-on-write. Every layer at
        once, because a block id names the same page in all of them. Copying one layer's
        page and not the others would leave a sequence attending over a prefix correct
        in layer 0 and stale in layer 1.
        """
        for block_id in (source, destination):
            if not 0 <= block_id < self.num_blocks:
                raise ValueError(f"block {block_id} is outside a pool of {self.num_blocks}")
        if source == destination:
            return

        # Copy raw bytes: a uint8 view is dtype-agnostic and works for every storage
        # type this pool can hold, FP8 included.
        keys, values = self.keys.view(torch.uint8), self.values.view(torch.uint8)
        keys[:, destination].copy_(keys[:, source])
        values[:, destination].copy_(values[:, source])

    def _check_layer(self, layer: int) -> None:
        if not 0 <= layer < self.num_layers:
            raise ValueError(f"layer {layer} is outside a pool of {self.num_layers} layers")

    def __repr__(self) -> str:
        stored = f"{self.kv_dtype} <- {self.dtype}" if self.is_fp8 else f"{self.dtype}"
        return (
            f"PagedKvPool(layers={self.num_layers}, blocks={self.num_blocks}, "
            f"block_size={self.block_size}, heads={self.num_kv_heads}, dim={self.head_dim}, "
            f"{stored}, {self.device.type})"
        )


# --- 5. Radix-tree prefix cache ----------------------------------------------

class RadixNode:
    """One cached block, and the edge of tokens that reaches it. The root is the only
    node with block_id None: it spells the empty prefix and owns no page.
    """

    __slots__ = ("parent", "token_key", "block_id", "children")

    def __init__(
        self,
        parent: RadixNode | None = None,
        token_key: tuple[int, ...] | None = None,
        block_id: int | None = None,
    ) -> None:
        self.parent = parent
        self.token_key = token_key
        self.block_id = block_id
        self.children: dict[tuple[int, ...], RadixNode] = {}


class PrefixCache:
    """A radix tree over block-aligned token prefixes, mapping them to blocks.

        match(tokens)          -> physical blocks for the longest cached prefix
        insert(tokens, blocks) -> register full blocks, returns the newly cached ones
        evict(block_id)        -> unlink a block the pool is reclaiming

    Edges are one block of tokens, and keys are exact token tuples, so unlike a hash
    cache there are no collisions. The tree owns no memory and never touches
    reference counts: a cached block sits on the free list at count zero, still
    referenced by its node, so prefix caching costs nothing when there are no hits.
    """

    def __init__(self, block_size: int) -> None:
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError(f"block_size must be a positive power of two, got {block_size}")
        self.block_size = block_size
        self.root = RadixNode()
        # Reverse index so eviction is O(depth) rather than a tree walk: the pool names
        # a block id, and this finds the node standing for it.
        self._node_of_block: dict[int, RadixNode] = {}

    # -------------------------------------------------------------- matching
    def match(self, token_ids: list[int]) -> list[int]:
        """The blocks of the longest cached prefix of token_ids, in order. Only whole
        blocks match -- a block still being written to is not immutable, so sharing it
        would be a data race. The caller increfs whatever comes back.
        """
        matched: list[int] = []
        node = self.root
        num_full = len(token_ids) // self.block_size
        for index in range(num_full):
            start = index * self.block_size
            key = tuple(token_ids[start : start + self.block_size])
            child = node.children.get(key)
            if child is None:
                break
            matched.append(child.block_id)  # type: ignore[arg-type]
            node = child
        return matched

    # ------------------------------------------------------------- insertion
    def insert(self, token_ids: list[int], block_ids: list[int]) -> list[int]:
        """Register block_ids as the cache of token_ids's full blocks, returning the newly
        cached ids for the pool to mark so it calls evict before reusing them. A block
        whose prefix another already caches is left out and freed normally: two
        sequences that prefilled the same prompt without a hit between them each
        computed it into their own page, and the tree keeps the first.
        """
        node = self.root
        newly: list[int] = []
        for index, block_id in enumerate(block_ids):
            start = index * self.block_size
            key = tuple(token_ids[start : start + self.block_size])
            if len(key) < self.block_size:
                break  # a partial block is not cacheable
            child = node.children.get(key)
            if child is None:
                child = RadixNode(parent=node, token_key=key, block_id=block_id)
                node.children[key] = child
                self._node_of_block[block_id] = child
                newly.append(block_id)
            node = child
        return newly

    # -------------------------------------------------------------- eviction
    def evict(self, block_id: int) -> None:
        """Unlink the block the pool is about to reuse, and orphan its subtree.

        Called by BlockPool as it pops a cached block off the free list. Since a held
        block also holds its ancestors, a free block's cached descendants are themselves
        free: dropping them loses nothing in use, and they keep their pages until the
        pool reuses them in turn, at which point their own eviction is a no-op.
        """
        node = self._node_of_block.pop(block_id, None)
        if node is None:
            return  # already orphaned by an ancestor's eviction
        if node.parent is not None and node.token_key is not None:
            node.parent.children.pop(node.token_key, None)
        self._orphan(node)

    def _orphan(self, node: RadixNode) -> None:
        """Drop a node's whole subtree from the reverse index."""
        stack = list(node.children.values())
        node.children.clear()
        while stack:
            child = stack.pop()
            self._node_of_block.pop(child.block_id, None)  # type: ignore[arg-type]
            stack.extend(child.children.values())
            child.children.clear()

    # ------------------------------------------------------------ inspection
    @property
    def num_cached_blocks(self) -> int:
        """How many blocks the tree currently points at. For tests and stats."""
        return len(self._node_of_block)


# --- 6. Block manager --------------------------------------------------------

class BlockManager:
    """Capacity, growth, sharing, and release: pages made usable by a scheduler.

        can_allocate(seq, n)   may this sequence take n more tokens?
        allocate(seq)          reserve blocks for everything uncomputed
        append_slot(seq)       grow by one token, adding a block only if needed
        fork(parent, child)    share the parent's blocks, copying nothing
        trim(seq, n)           roll back n tokens, releasing pages they emptied
        free(seq)              drop a holder on every block it owns

    It owns the pool, the KV tensors, and -- through the sequences -- their block
    tables, because copy-on-write touches a refcount, a table entry and a page at
    once. That copy takes only the last partial block, and only when shared.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        num_layers: int = 1,
        num_kv_heads: int = 1,
        head_dim: int = 1,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        enable_prefix_caching: bool = False,
        kv_dtype: torch.dtype | None = None,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        self.block_size = block_size
        self.pool = BlockPool(num_blocks)
        self.kv = PagedKvPool(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            kv_dtype=kv_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        # Prefix caching, off by default. When on, the pool calls back into the tree as
        # it reclaims a cached page, and `cached_tokens` counts tokens served from a hit
        # instead of recomputed.
        self.cache: PrefixCache | None = None
        self.cached_tokens = 0
        # Fresh pages taken from the free list over this manager's life. A prefix hit
        # reuses pages rather than taking new ones, so the saving shows up here and not
        # in the free count: a reused page still leaves the free list, it is just not a
        # newly reserved one.
        self.blocks_allocated = 0
        if enable_prefix_caching:
            self.cache = PrefixCache(block_size)
            self.pool.on_evict = self.cache.evict

    # --------------------------------------------------------- introspection
    @property
    def num_free_blocks(self) -> int:
        return self.pool.num_free

    @property
    def num_blocks(self) -> int:
        return self.pool.num_blocks

    def table(self, sequence: Sequence) -> BlockTable:
        """The sequence's table, raising if it was never allocated. A missing table means
        the scheduler ran a sequence the manager has not seen, which would otherwise
        surface as attention over an empty cache: fluent output unrelated to the prompt.
        """
        if sequence.block_table is None:
            raise ValueError(f"sequence {sequence.seq_id} has no block table; allocate it first")
        return sequence.block_table

    def has_table(self, sequence: Sequence) -> bool:
        return sequence.block_table is not None

    # ----------------------------------------------------- admission control
    def blocks_needed(self, sequence: Sequence, num_tokens: int) -> int:
        """Fresh blocks required to hold num_tokens more of this sequence, including the
        copy-on-write block: a shared partial page must be duplicated before it can be
        written. Omitting it lets admission control pass and the write fail
        mid-iteration, with half the batch already committed.
        """
        table = sequence.block_table
        if table is None:
            return -(-num_tokens // self.block_size)  # ceiling division

        needed = table.blocks_needed_for(num_tokens)
        if num_tokens and self._needs_copy(table):
            needed += 1
        return needed

    def can_allocate(self, sequence: Sequence, num_tokens: int | None = None) -> bool:
        """Whether num_tokens more tokens of this sequence would fit. Defaults to
        everything it has left to compute, the admission question for a new request; the
        scheduler passes a chunk size when chunking is on, and 1 before a decode step.
        """
        if num_tokens is None:
            num_tokens = sequence.num_uncomputed_tokens
        return self.blocks_needed(sequence, num_tokens) <= self.pool.num_free

    # -------------------------------------------------------------- lifetime
    def allocate(self, sequence: Sequence, num_tokens: int | None = None) -> None:
        """Reserve capacity for num_tokens of this sequence, creating its table. Calling it
        again extends the existing table rather than replacing it, which is what a
        chunked prefill does every iteration.
        """
        if num_tokens is None:
            num_tokens = sequence.num_uncomputed_tokens
        if num_tokens < 0:
            raise ValueError(f"cannot reserve {num_tokens} tokens")

        if sequence.block_table is None:
            sequence.block_table = BlockTable(self.block_size)

        table = sequence.block_table
        if num_tokens:
            self._resolve_copy_on_write(sequence)
        fresh = self.pool.allocate_many(table.blocks_needed_for(num_tokens))
        self.blocks_allocated += len(fresh)
        for block_id in fresh:
            table.append_block(block_id)
        table.append_tokens(num_tokens)

    def append_slot(self, sequence: Sequence) -> None:
        """Grow by exactly one token: the decode step. Usually one increment, since the
        last block has room. It costs an allocation once every P tokens, and a copy the
        first time a forked sequence writes into a shared page.
        """
        self.allocate(sequence, 1)

    def maybe_apply_prefix_cache(self, sequence: Sequence) -> int:
        """Match this sequence's prompt against the cache and reuse what hits, returning
        the tokens reused.

        Called once when a sequence is admitted and before the scheduler sizes its
        prefill, so a hit shows up in num_computed_tokens. The match never covers the
        whole sequence: a request hitting on every block would have nothing left to
        forward and no logits to sample from.
        """
        if self.cache is None or sequence.block_table is not None:
            return 0

        sequence.block_table = BlockTable(self.block_size)
        if sequence.num_computed_tokens != 0 or sequence.num_output_tokens != 0:
            return 0  # only a fresh prompt is a prefix worth matching

        matched = self.cache.match(sequence.token_ids)
        while matched and len(matched) * self.block_size >= len(sequence):
            matched.pop()
        if not matched:
            return 0

        table = sequence.block_table
        for block_id in matched:
            self.pool.acquire_cached(block_id)
            table.append_block(block_id)
        reused = len(matched) * self.block_size
        table.append_tokens(reused)
        sequence.num_computed_tokens = reused
        self.cached_tokens += reused
        return reused

    def _cache_full_blocks(self, sequence: Sequence) -> None:
        """Register a sequence's completed full blocks in the prefix tree, from free and
        before the blocks are decref'd. Only whole computed blocks are cached: flooring
        table.num_tokens to a block boundary gives exactly the immutable prefix.
        """
        if self.cache is None:
            return
        table = sequence.block_table
        if table is None:
            return
        num_full = table.num_tokens // self.block_size
        if num_full == 0:
            return
        block_ids = list(table.block_ids[:num_full])
        tokens = sequence.token_ids[: num_full * self.block_size]
        for block_id in self.cache.insert(tokens, block_ids):
            self.pool.mark_cached(block_id)

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """Point child at every one of parent's blocks, copying nothing. The refcount bump
        is the whole operation: a 4000-token prompt shared by eight samples costs 250
        blocks instead of 2000, and the branches diverge only when one writes.
        """
        table = self.table(parent)
        if child.block_table is not None:
            raise ValueError(f"sequence {child.seq_id} already has a block table")

        for block_id in table.block_ids:
            self.pool.incref(block_id)
        child.block_table = table.copy()

    def trim(self, sequence: Sequence, num_tokens: int) -> int:
        """Give back the last num_tokens slots, returning how many blocks reached the pool.

        The rollback half of speculative decoding: a step writes k proposed tokens into
        the cache before the target verifies them, because the verifying pass must
        attend over them. Trimming to a block boundary releases pages, trimming within
        the last partial block releases none.
        """
        if num_tokens < 0:
            raise ValueError(f"cannot trim {num_tokens} tokens")
        if num_tokens == 0:
            return 0

        table = self.table(sequence)
        emptied = table.trim_tokens(num_tokens)

        released = 0
        for _ in range(emptied):
            block_id = table.drop_last_block()
            released += int(self.pool.decref(block_id))
        return released

    def free(self, sequence: Sequence) -> int:
        """Drop a holder on every block the sequence owns, returning how many were actually
        released -- fewer than it holds when a fork is still alive. Safe to call twice:
        freeing a finished request again is a scheduler retry, whereas double-decrefing
        the pool would be corruption.
        """
        table = sequence.block_table
        if table is None:
            return 0

        # Cache the immutable prefix before releasing the blocks. The pages then sit on
        # the free list at reference count zero, still referenced by the tree, and are
        # reclaimed only when the pool runs short.
        self._cache_full_blocks(sequence)

        sequence.block_table = None
        return self.pool.decref_many(list(table.block_ids))

    # --------------------------------------------------------- copy-on-write
    def _needs_copy(self, table: BlockTable) -> bool:
        """Whether the next write into this table would land on a shared page. Only the
        last block is ever written to, and only while partial, since a full block is
        immutable -- which makes it the only page copy-on-write considers.
        """
        if not table.num_blocks or table.num_empty_slots == 0:
            return False
        return self.pool.ref_count(table.block_ids[-1]) > 1

    def _resolve_copy_on_write(self, sequence: Sequence) -> int | None:
        """Give the sequence a private copy of its last page if it is sharing one,
        returning the new block id or None. Ordering is load-bearing: allocate, copy,
        repoint, then decref. Decrefing first could return the source page to the free
        list and hand it to another sequence before its contents were read.
        """
        table = self.table(sequence)
        if not self._needs_copy(table):
            return None

        fresh = self.pool.allocate()
        source = table.block_ids[-1]
        self.kv.copy_block(source, fresh)
        displaced = table.replace_block(table.num_blocks - 1, fresh)
        self.pool.decref(displaced)
        return fresh

    # ------------------------------------------------------- the GPU handoff
    def slots(self, sequence: Sequence, num_tokens: int) -> list[int]:
        """Where this iteration's num_tokens tokens are written, as integers: the last
        num_tokens positions of the sequence, since allocate has already run for them.
        Occupancy leads computation by exactly this iteration's worth.
        """
        table = self.table(sequence)
        start = table.num_tokens - num_tokens
        if start < 0:
            raise ValueError(
                f"sequence {sequence.seq_id} holds {table.num_tokens} tokens; "
                f"cannot map {num_tokens}"
            )

        slots = table.slots(range(start, table.num_tokens))
        # Bounds-checked here, while the slots are still integers. Doing it in
        # `PagedKvPool.write` cost a device read per layer per iteration for the same
        # arithmetic; the only difference is host versus critical path.
        for slot in slots:
            if not 0 <= slot < self.kv.num_slots:
                raise ValueError(
                    f"sequence {sequence.seq_id} maps to slot {slot}, outside a pool of "
                    f"{self.kv.num_slots}"
                )
        return slots

    def slot_mapping(self, sequence: Sequence, num_tokens: int) -> torch.Tensor:
        """`slots`, as the int32 tensor the pool's write path scatters through."""
        return torch.tensor(
            self.slots(sequence, num_tokens), dtype=torch.int32, device=self.kv.device
        )

    def check_no_leaks(self) -> None:
        """Assert every block is back in the pool. For the end of a test or a run."""
        self.pool.check_consistency()
        if self.pool.num_free != self.pool.num_blocks:
            raise AssertionError(
                f"{self.pool.num_allocated} of {self.pool.num_blocks} blocks leaked: "
                f"{self.pool.allocated_ids()}"
            )

    def __repr__(self) -> str:
        return (
            f"BlockManager(blocks={self.pool.num_blocks}, free={self.pool.num_free}, "
            f"block_size={self.block_size})"
        )
