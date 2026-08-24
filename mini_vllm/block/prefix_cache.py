"""Radix-tree prefix caching: reuse the KV of a shared prompt across requests.

Two requests beginning with the same tokens compute the same keys and values for that
prefix. Served workloads are full of this — a shared system prompt, a few-shot preamble,
a conversation replayed with one more turn — and recomputing it is the largest single
waste available to an inference engine.

The structure is a radix tree whose edges are one block of tokens. A path from the root
spells a token prefix in block-sized steps, and the node ending each edge names the
physical block holding that block's KV::

    root
     ├─ (t0..t15)  -> block 7
     │                 └─ (t16..t31) -> block 12
     └─ (u0..u15)  -> block 3

Matching a prompt walks the tree as far as its tokens agree, one block at a time, and
the blocks along the matched path are shared rather than recomputed. Keys are exact
token tuples, so unlike a hash cache there are no collisions: two prefixes share a node
exactly when their tokens are identical.

The tree does not own memory. A cached block sits on the block pool's free list at
reference count zero, still referenced by its node, and is reclaimed as soon as the pool
needs it — :class:`~mini_vllm.block.block_pool.BlockPool` calls
:meth:`PrefixCache.evict` as it hands the block out. Prefix caching therefore costs
nothing when there are no hits: a cached block is a free block that remembers its
contents until the page is needed.
"""

from __future__ import annotations

__all__ = ["PrefixCache", "RadixNode"]


class RadixNode:
    """One cached block, and the edge of tokens that reaches it.

    The root is the only node with ``block_id is None``: it spells the empty prefix and
    owns no page. Every other node stands for one physical block whose KV is the block's
    worth of tokens on the edge from its parent.
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

    ::

        match(tokens)          -> physical blocks for the longest cached prefix
        insert(tokens, blocks) -> register full blocks, returns the newly cached ones
        evict(block_id)        -> unlink a block the pool is reclaiming

    The cache never touches reference counts. Matching returns block ids to the block
    manager, which is what may incref them; inserting registers blocks a finishing
    sequence is about to release; eviction unlinks a node whose block the pool has
    already decided to reuse.
    """

    def __init__(self, block_size: int) -> None:
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError(f"block_size must be a positive power of two, got {block_size}")
        self.block_size = block_size
        self.root = RadixNode()
        # Reverse index so eviction is O(depth) rather than a tree walk: the pool names
        # a block id, and this finds the node standing for it.
        self._node_of_block: dict[int, RadixNode] = {}

    # ------------------------------------------------------------------- matching

    def match(self, token_ids: list[int]) -> list[int]:
        """The blocks of the longest cached prefix of ``token_ids``, in order.

        Only whole blocks match. A partial trailing block is never cached: a block still
        being written to is not immutable, so sharing it would be a data race. The caller
        increfs whatever comes back.
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

    # ------------------------------------------------------------------ insertion

    def insert(self, token_ids: list[int], block_ids: list[int]) -> list[int]:
        """Register ``block_ids`` as the cache of ``token_ids``'s full blocks.

        Returns the newly cached block ids, which the pool must mark so it calls
        :meth:`evict` before reusing them. A block whose prefix another block already
        caches is left out and freed normally: two sequences that prefilled the same
        prompt without a hit between them each computed it into their own page, and the
        tree keeps the first.
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

    # ------------------------------------------------------------------- eviction

    def evict(self, block_id: int) -> None:
        """Unlink the block the pool is about to reuse, and orphan its subtree.

        Called by :class:`BlockPool` as it pops a cached block off the free list. The
        block's descendants become unreachable once it is gone, and since a held block
        also holds its ancestors, a free block's cached descendants are themselves free:
        dropping them loses nothing in use. They keep their pages until the pool reuses
        them in turn, at which point their own eviction is a no-op.
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

    # ---------------------------------------------------------------- inspection

    @property
    def num_cached_blocks(self) -> int:
        """How many blocks the tree currently points at. For tests and stats."""
        return len(self._node_of_block)
