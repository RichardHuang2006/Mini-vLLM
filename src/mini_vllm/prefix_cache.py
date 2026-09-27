"""Radix-tree prefix caching: block-aligned token prefixes mapped to the pages that hold them."""

from __future__ import annotations

__all__ = ["PrefixCache", "RadixNode"]


class RadixNode:
    """One cached block and the edge of tokens that reaches it; the root owns no page."""

    __slots__ = ("block_id", "children", "parent", "token_key")

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
    """Edges are one block of tokens keyed by the exact token tuple, so there are no hash
    collisions. The tree owns no memory and never touches reference counts."""

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.root = RadixNode()
        # Reverse index, so eviction is O(depth) rather than a tree walk.
        self._node_of_block: dict[int, RadixNode] = {}

    def match(self, token_ids: list[int]) -> list[int]:
        """The blocks of the longest cached prefix of token_ids, whole blocks only."""
        matched = []
        node = self.root
        for start in range(0, len(token_ids) - self.block_size + 1, self.block_size):
            node = node.children.get(tuple(token_ids[start : start + self.block_size]))
            if node is None:
                break
            matched.append(node.block_id)
        return matched

    def insert(self, token_ids: list[int], block_ids: list[int]) -> list[int]:
        """Register block_ids as the pages of token_ids's full blocks, returning the newly
        cached ones; a block whose prefix is already cached is left out."""
        node = self.root
        newly = []
        for index, block_id in enumerate(block_ids):
            key = tuple(token_ids[index * self.block_size : (index + 1) * self.block_size])
            if len(key) < self.block_size:
                break  # a partial block is still being written, so it is not a prefix yet
            child = node.children.get(key)
            if child is None:
                child = RadixNode(parent=node, token_key=key, block_id=block_id)
                node.children[key] = child
                self._node_of_block[block_id] = child
                newly.append(block_id)
            node = child
        return newly

    def evict(self, block_id: int) -> None:
        """Unlink a page the pool is about to reuse, and orphan everything below it."""
        node = self._node_of_block.pop(block_id, None)
        if node is None:
            return  # already orphaned by an ancestor's eviction
        node.parent.children.pop(node.token_key)

        stack = list(node.children.values())
        while stack:
            child = stack.pop()
            self._node_of_block.pop(child.block_id)
            stack.extend(child.children.values())

    @property
    def num_cached_blocks(self) -> int:
        return len(self._node_of_block)
