"""The paged KV cache: refcounted blocks, per-sequence block tables, the pages themselves,
copy-on-write, and the per-layer view the model attends through."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

import mlx.core as mx

from mini_vllm.attention import paged_attention
from mini_vllm.prefix_cache import PrefixCache
from mini_vllm.quantize import quantize_fp8

if TYPE_CHECKING:
    from mini_vllm.batch import ForwardBatch
    from mini_vllm.scheduler import Request

__all__ = ["BlockManager", "BlockPool", "BlockTable", "PagedKvCache", "PagedKvPool"]


class BlockPool:
    """A fixed set of physical block ids, handed out and refcounted."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.ref_counts = [0] * num_blocks
        # FIFO, not LIFO, so a use-after-free reads another sequence's page, not its own.
        self.free: deque[int] = deque(range(num_blocks))

        # Prefix caching: a cached block stays matchable on the free list until reused.
        self.cached = [False] * num_blocks
        self.on_evict: Callable[[int], None] | None = None

    @property
    def num_free(self) -> int:
        return len(self.free)

    def allocate(self) -> int:
        """Take the oldest free block at reference count 1."""
        block_id = self.free.popleft()
        # Repurposing a cached page: unlink it first, so no later match can return it.
        if self.cached[block_id]:
            self.cached[block_id] = False
            self.on_evict(block_id)
        self.ref_counts[block_id] = 1
        return block_id

    def incref(self, block_id: int) -> None:
        """Add a holder: a fork, or a prefix-cache hit on a block still in use."""
        self.ref_counts[block_id] += 1

    def decref(self, block_id: int) -> bool:
        """Drop a holder, returning True if that freed the block."""
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id]:
            return False
        self.free.append(block_id)
        return True

    def acquire_cached(self, block_id: int) -> None:
        """Hold a matched block, from wherever it sits: in use, or free but still cached."""
        if self.ref_counts[block_id] == 0:
            self.free.remove(block_id)
        self.ref_counts[block_id] += 1


class BlockTable:
    """The physical block ids backing one sequence, in logical order, and how many of
    their slots hold tokens."""

    def __init__(self, block_size: int, block_ids: list[int] | None = None, num_tokens: int = 0) -> None:
        self.block_size = block_size
        self.block_ids = [] if block_ids is None else list(block_ids)
        self.num_tokens = num_tokens

    @property
    def num_slots(self) -> int:
        """Token capacity, not a range of physical slot numbers."""
        return len(self.block_ids) * self.block_size

    @property
    def num_empty_slots(self) -> int:
        """Room left in the last block."""
        return self.num_slots - self.num_tokens

    def blocks_needed_for(self, num_new_tokens: int) -> int:
        """How many fresh blocks appending num_new_tokens would take; often zero."""
        deficit = num_new_tokens - self.num_empty_slots
        return max(0, -(-deficit // self.block_size))

    def append_block(self, block_id: int) -> None:
        self.block_ids.append(block_id)

    def append_tokens(self, count: int) -> None:
        self.num_tokens += count

    def replace_block(self, index: int, block_id: int) -> int:
        """Repoint one entry, returning the id it displaced for the caller to decref."""
        displaced = self.block_ids[index]
        self.block_ids[index] = block_id
        return displaced

    def copy(self) -> BlockTable:
        """An independent table over the same blocks; the caller increfs them."""
        return BlockTable(self.block_size, self.block_ids, self.num_tokens)

    def slots(self, start: int, stop: int) -> list[int]:
        """The flat pool slot of each position in [start, stop): block_id * P + offset."""
        size = self.block_size
        return [self.block_ids[p // size] * size + p % size for p in range(start, stop)]


class PagedKvPool:
    """Every layer's keys and values, allocated once and never grown. Each layer is stored
    flat, num_slots x H_k x D, so a write is one scatter through the slot mapping; the
    paged num_blocks x P x H_k x D view attention reads is a free reshape."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.bfloat16,
        fp8: bool = False,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # fp8 pages hold e4m3 bits in uint8 at static per-tensor scales: half the bytes of bf16.
        self.fp8 = fp8
        self.k_scale = k_scale
        self.v_scale = v_scale

        shape = (num_blocks * block_size, num_kv_heads, head_dim)
        storage = mx.uint8 if fp8 else dtype
        self.keys = [mx.zeros(shape, dtype=storage) for _ in range(num_layers)]
        self.values = [mx.zeros(shape, dtype=storage) for _ in range(num_layers)]

    def write(
        self, layer: int, slot_mapping: mx.array, key: mx.array, value: mx.array, use_metal: bool = False
    ) -> None:
        """Scatter key/value [T, H_k, D] into the slots slot_mapping names.

        MLX donates the pool's buffer to the scatter when nothing else holds it, so this
        costs the T slots written, not a copy of the pool. The Metal kernels write in place
        by construction, quantizing to fp8 in the same pass.
        """
        if use_metal:
            import mini_vllm_ext

            if self.fp8:
                scatter = mini_vllm_ext.kv_quantize_scatter
                self.keys[layer] = scatter(self.keys[layer], slot_mapping, key, self.k_scale)
                self.values[layer] = scatter(self.values[layer], slot_mapping, value, self.v_scale)
            else:
                self.keys[layer] = mini_vllm_ext.paged_cache_update(self.keys[layer], slot_mapping, key)
                self.values[layer] = mini_vllm_ext.paged_cache_update(self.values[layer], slot_mapping, value)
            return
        if self.fp8:
            key, value = quantize_fp8(key, self.k_scale), quantize_fp8(value, self.v_scale)
        self.keys[layer][slot_mapping] = key.astype(self.keys[layer].dtype)
        self.values[layer][slot_mapping] = value.astype(self.values[layer].dtype)

    def pages(self, layer: int) -> tuple[mx.array, mx.array]:
        """One layer's keys and values as num_blocks x P x H_k x D."""
        shape = (self.num_blocks, self.block_size, self.num_kv_heads, self.head_dim)
        return self.keys[layer].reshape(shape), self.values[layer].reshape(shape)

    def copy_block(self, source: int, destination: int) -> None:
        """Duplicate one page in every layer: the copy in copy-on-write. Raw values, so an
        fp8 page is copied bit for bit."""
        src = slice(source * self.block_size, (source + 1) * self.block_size)
        dst = slice(destination * self.block_size, (destination + 1) * self.block_size)
        for pool in (*self.keys, *self.values):
            pool[dst] = pool[src]

    def caches(self, batch: ForwardBatch) -> list[PagedKvCache]:
        """One view per layer for this batch, to pass to the model as its caches."""
        return [PagedKvCache(self, layer, batch) for layer in range(len(self.keys))]


class PagedKvCache:
    """One layer of the pool, as seen by one ragged batch. The model sees one row of
    T flattened tokens: q is 1 x H_q x T x D, k and v are 1 x H_k x T x D."""

    def __init__(self, pool: PagedKvPool, layer: int, batch: ForwardBatch) -> None:
        self.pool = pool
        self.layer = layer
        self.batch = batch

    def attend(self, q: mx.array, k: mx.array, v: mx.array, use_metal: bool = False) -> mx.array:
        """Write this batch's keys and values into their slots, then attend each sequence
        over its own pages."""
        pool, batch = self.pool, self.batch
        pool.write(self.layer, batch.slot_mapping, k[0].swapaxes(0, 1), v[0].swapaxes(0, 1), use_metal)
        key_pages, value_pages = pool.pages(self.layer)
        out = paged_attention(
            q[0].swapaxes(0, 1),
            key_pages,
            value_pages,
            batch.block_tables,
            batch.cu_seqlens_q,
            batch.context_lens,
            k_scale=pool.k_scale,
            v_scale=pool.v_scale,
            use_metal=use_metal,
        )
        return out.swapaxes(0, 1)[None]


class BlockManager:
    """Capacity, growth, sharing and release: the pool made usable by a scheduler.

    It owns the block pool and the KV pool together, and reaches each request's block
    table through the request, because copy-on-write changes a refcount, a table entry
    and a page at once.
    """

    def __init__(self, kv: PagedKvPool, enable_prefix_caching: bool = False) -> None:
        self.kv = kv
        self.block_size = kv.block_size
        self.pool = BlockPool(kv.num_blocks)
        self.cache = None
        if enable_prefix_caching:
            self.cache = PrefixCache(self.block_size)
            self.pool.on_evict = self.cache.evict

    def blocks_needed(self, request: Request, num_tokens: int) -> int:
        """Fresh blocks num_tokens more would take, counting a copy-on-write copy."""
        table = request.block_table
        if table is None:
            return -(-num_tokens // self.block_size)
        return table.blocks_needed_for(num_tokens) + self._needs_copy(table)

    def allocate(self, request: Request, num_tokens: int) -> None:
        """Reserve slots for num_tokens more tokens, creating or extending the table."""
        if request.block_table is None:
            request.block_table = BlockTable(self.block_size)
        table = request.block_table

        self._resolve_copy_on_write(table)
        for _ in range(table.blocks_needed_for(num_tokens)):
            table.append_block(self.pool.allocate())
        table.append_tokens(num_tokens)

    def slots(self, request: Request, num_tokens: int) -> list[int]:
        """Where the last num_tokens reserved tokens are written."""
        table = request.block_table
        return table.slots(table.num_tokens - num_tokens, table.num_tokens)

    def apply_prefix_cache(self, request: Request) -> int:
        """Reuse whatever of a fresh request's tokens the cache holds, returning how many."""
        if self.cache is None or request.block_table is not None:
            return 0

        matched = self.cache.match(request.token_ids)
        # A full match would leave nothing to forward, and so no logits to sample from.
        while matched and len(matched) * self.block_size >= len(request):
            matched.pop()
        if not matched:
            return 0

        for block_id in matched:
            self.pool.acquire_cached(block_id)
        reused = len(matched) * self.block_size
        request.block_table = BlockTable(self.block_size, matched, reused)
        request.num_computed_tokens = reused
        return reused

    def trim(self, request: Request, num_tokens: int) -> int:
        """Give back the last num_tokens slots, a rejected speculative tail, returning how
        many blocks that freed."""
        table = request.block_table
        table.num_tokens -= num_tokens
        blocks_used = -(-table.num_tokens // self.block_size)
        released = 0
        while len(table.block_ids) > blocks_used:
            released += self.pool.decref(table.block_ids.pop())
        return released

    def fork(self, parent: Request, child: Request) -> None:
        """Point child at every one of parent's blocks, copying nothing."""
        for block_id in parent.block_table.block_ids:
            self.pool.incref(block_id)
        child.block_table = parent.block_table.copy()

    def free(self, request: Request) -> int:
        """Drop a holder on every block the request owns, returning how many were freed."""
        table = request.block_table
        if table is None:
            return 0

        # Cache the full blocks first: they stay matchable while they sit on the free list.
        if self.cache is not None:
            num_full = table.num_tokens // self.block_size
            tokens = request.token_ids[: num_full * self.block_size]
            for block_id in self.cache.insert(tokens, table.block_ids[:num_full]):
                self.pool.cached[block_id] = True

        request.block_table = None
        return sum(self.pool.decref(block_id) for block_id in table.block_ids)

    def _needs_copy(self, table: BlockTable) -> bool:
        """Whether the next write into this table would land on a shared page."""
        if not table.block_ids or table.num_empty_slots == 0:
            return False
        return self.pool.ref_counts[table.block_ids[-1]] > 1

    def _resolve_copy_on_write(self, table: BlockTable) -> None:
        """Give the table a private copy of its last page if it is sharing one."""
        if not self._needs_copy(table):
            return
        # Allocate, copy, repoint, then decref: the other order could hand out the source.
        fresh = self.pool.allocate()
        self.kv.copy_block(table.block_ids[-1], fresh)
        self.pool.decref(table.replace_block(len(table.block_ids) - 1, fresh))
