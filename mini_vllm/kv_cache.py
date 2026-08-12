"""The KV cache, and the interface a paged one hides behind.

Attention at position `t` needs the keys and values of every position `0..t`, and
those do not change once computed. The uncached model recomputes them all anyway,
every single step, which is why generation runs at 1 token/s. Caching them turns
decode from quadratic into linear work.

:class:`KvCache` is abstract for one reason: the block manager provides a paged
implementation that stores the same tensors in fixed-size blocks scattered across
a pool, and the model must not be able to tell the difference. So the interface is
deliberately narrow — one method, taking the new keys and values and returning
everything accumulated so far — and any cache that satisfies it can be swapped in
without the model changing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

__all__ = ["KvCache", "DenseKvCache"]


class KvCache(ABC):
    """One layer's worth of cached keys and values.

    One cache per layer, not one per model: each layer's keys and values are
    independent, and the model holds a list of them.
    """

    @property
    @abstractmethod
    def offset(self) -> int:
        """How many positions are currently cached."""

    @abstractmethod
    def update_and_fetch(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Append ``key``/``value`` and return everything cached, plus the write offset.

        ::

            key, value (incoming):  B x H_k x L x D
            returns full_key/value: B x H_k x S x D     where S = offset + L
            returns offset:         the length *before* this call

        The returned offset is the position this update was written *at*, not the
        length afterwards — so ``S == offset + L`` holds as written above, and
        ``self.offset`` afterwards equals ``S``. The distinction matters because
        the caller uses it to build a causal mask, and being off by ``L`` there
        silently lets a token attend to its own future.

        Note that keys must already have RoPE applied before they get here.
        Positions are baked into the cached tensors, which is what makes a cache
        entry reusable at all.
        """

    @abstractmethod
    def reset(self) -> None:
        """Forget everything, so the cache can serve a new sequence."""


class DenseKvCache(KvCache):
    """A cache that simply concatenates along the sequence dimension.

    The obvious implementation, and a deliberately flawed one. Two costs, both of
    which the serving layer exists to remove:

    * **Every decode step reallocates.** ``torch.cat`` cannot extend a tensor in
      place, so appending one token to a cache of `S` copies all `S` positions to
      a new buffer. Over a full generation that is quadratic memory traffic to
      store linear data.
    * **One contiguous block per sequence.** A batch of sequences with different
      lengths has to be padded to the longest, and a sequence that might reach
      40960 tokens has to be budgeted for as if it will. That is the fragmentation
      problem PagedAttention solves.

    It is correct, though, and correct is what makes it the oracle for the paged
    version.
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
