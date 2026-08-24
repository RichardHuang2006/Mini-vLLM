"""The physical KV storage the block ids index into.

One pre-allocated pool, partitioned into fixed-size pages of `P` tokens.

::

    keys:   num_layers x num_blocks x P x H_k x D
    values: num_layers x num_blocks x P x H_k x D

Allocated once at startup and never grown: a `cudaMalloc` in the middle of a decode step
would stall every sequence in flight, and a pool that can be exhausted but not
fragmented turns memory pressure into a scheduling problem.

Two asymmetric operations:

* :meth:`write` scatters this iteration's new keys and values into the slots the block
  tables name, one indexed copy per layer. Physical order is irrelevant.
* :meth:`gather` collects one sequence's cache back into a contiguous tensor. This is
  the slow reference path the paged attention kernel is diffed against, not a serving
  path: doing it for real would copy the entire cache every iteration.
"""

from __future__ import annotations

import torch

__all__ = ["PagedKvPool"]


class PagedKvPool:
    """Pre-allocated paged storage for every layer's keys and values.

    The layer axis is part of one tensor rather than a list of per-layer tensors, so the
    whole cache is a single allocation whose size is knowable up front. That is what
    makes :meth:`bytes_for` answerable, which is how an engine picks `num_blocks` from a
    memory budget.
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
        """How much GPU memory a pool of this shape would take, keys and values.

        ``dtype`` is the storage dtype: pass ``torch.float8_e4m3fn`` to size an FP8 pool,
        where a page of the same geometry costs half as much and the engine fits twice
        as many.
        """
        elements = num_layers * num_blocks * block_size * num_kv_heads * head_dim
        return 2 * elements * torch.empty((), dtype=dtype).element_size()

    @property
    def num_slots(self) -> int:
        """Token slots per layer: the range `physical_slot` addresses."""
        return self.num_blocks * self.block_size

    # -------------------------------------------------------------------- access

    def layer_keys(self, layer: int) -> torch.Tensor:
        """One layer's key pool, `num_blocks x P x H_k x D`. A view, not a copy."""
        return self.keys[layer]

    def layer_values(self, layer: int) -> torch.Tensor:
        return self.values[layer]

    def flat(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's pools flattened to `num_slots x H_k x D`.

        The layout `slot_mapping` indexes into. `block_id * P + offset` is a flat slot
        number because the block and offset axes are adjacent and contiguous, so this
        view is free.
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

        `T` spans the whole ragged batch — a prefill chunk's tokens and a dozen decode
        steps' single tokens in one call — because the slots are already absolute. Nothing
        here needs to know which sequence a token belongs to, which is what makes one
        launch sufficient.

        Slot values are not checked here. This is the innermost call in the engine, 28
        layers per iteration, and `slot_mapping.max()` on a CUDA tensor is a
        device-to-host read that waits for everything queued behind it: two per layer
        measured 7 ms per iteration, a third of the model's time, to re-check integers
        `BlockManager.slots` already bounds-checked on the host. Shapes are checked
        because that is free.
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
            from mini_vllm.kernels import ops

            ops.quantize_scatter(
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

        The shape the reference attention implementation takes, which is what makes a
        paged cache testable against a dense one. It is a copy — the gather paged kernels
        exist to avoid — and is present only as the oracle.
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

    # ---------------------------------------------------------------------- copy

    def copy_block(self, source: int, destination: int) -> None:
        """Duplicate one page across every layer: the copy in copy-on-write.

        Every layer at once, because a block id names the same page in all of them — the
        block table is per sequence, not per sequence and layer. Copying one layer's page
        and not the others would leave a sequence attending over a prefix that is correct
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
