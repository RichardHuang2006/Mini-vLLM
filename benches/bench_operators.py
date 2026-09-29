"""Each Metal kernel against the pure-MLX branch of its operator and against MLX's own op, at
Qwen3-0.6B shapes in bf16.

    uv run python benches/bench_operators.py
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from matplotlib.figure import Figure
from utils import METAL_BUILT, evidence, grouped_bars, per_call_us, save_plot

from mini_vllm import PagedKvPool, RMSNorm, RoPE, paged_attention, swiglu
from mini_vllm import scaled_dot_product_attention_grouped as sdpa

BF16 = mx.bfloat16
HIDDEN, INTERMEDIATE, Q_HEADS, KV_HEADS, HEAD_DIM = 1024, 3072, 16, 8, 128
WRITES = 50


def normal(*shape: int) -> mx.array:
    return mx.random.normal(shape).astype(BF16)


def cases() -> list[tuple[str, int, object, object, object | None]]:
    """(name, bytes it must move at least, pure, metal, MLX's own or None)."""
    out = []
    for rows in (32, 512):
        x, w = normal(rows, HIDDEN), mx.ones((HIDDEN,), BF16)
        out.append((f"rms_norm [{rows}, {HIDDEN}]", 2 * x.nbytes,
                    lambda x=x, w=w: RMSNorm(HIDDEN, w)(x),
                    lambda x=x, w=w: RMSNorm(HIDDEN, w, use_metal=True)(x),
                    lambda x=x, w=w: mx.fast.rms_norm(x, w, 1e-6)))

        gate, up = normal(rows, INTERMEDIATE), normal(rows, INTERMEDIATE)
        out.append((f"swiglu [{rows}, {INTERMEDIATE}]", 3 * gate.nbytes,
                    lambda g=gate, u=up: swiglu(g, u),
                    lambda g=gate, u=up: swiglu(g, u, use_metal=True),
                    lambda g=gate, u=up: nn.silu(g) * u))

        pure, metal = RoPE(HEAD_DIM, 4096), RoPE(HEAD_DIM, 4096, use_metal=True)
        q, positions = normal(1, rows, Q_HEADS, HEAD_DIM), mx.arange(1000, 1000 + rows)
        out.append((f"rope [1, {rows}, {Q_HEADS}, {HEAD_DIM}]", 2 * q.nbytes,
                    lambda q=q, p=positions, rope=pure: rope(q, p),
                    lambda q=q, p=positions, rope=metal: rope(q, p),
                    lambda q=q: mx.fast.rope(q.swapaxes(1, 2), HEAD_DIM, traditional=False,
                                             base=1e6, scale=1.0, offset=1000)))

    for name, length, context in (("decode_attention", 1, 2048), ("flash_prefill", 512, 512)):
        q = normal(1, Q_HEADS, length, HEAD_DIM)
        k, v = normal(1, KV_HEADS, context, HEAD_DIM), normal(1, KV_HEADS, context, HEAD_DIM)
        out.append((f"{name} L={length} S={context}", 2 * q.nbytes + k.nbytes + v.nbytes,
                    lambda q=q, k=k, v=v: sdpa(q, k, v, mask="causal"),
                    lambda q=q, k=k, v=v: sdpa(q, k, v, mask="causal", use_metal=True),
                    lambda q=q, k=k, v=v: mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=HEAD_DIM**-0.5, mask="causal")))

    pool = PagedKvPool(1, 4096, 16, KV_HEADS, HEAD_DIM, dtype=BF16)
    key_pages, value_pages = pool.pages(0)
    batches = (("32 decodes, context 1024", [(1, 1024)] * 32), ("one 512-token chunk", [(512, 512)]))
    for label, shape in batches:
        starts, tables, contexts, used = [0], [], [], 0
        for length, context in shape:
            starts.append(starts[-1] + length)
            blocks = -(-context // 16)
            tables.append(list(range(used, used + blocks)))
            used += blocks
            contexts.append(context)
        tables = mx.array(tables, dtype=mx.int32)
        q = normal(starts[-1], Q_HEADS, HEAD_DIM)
        args = (q, key_pages, value_pages, tables, mx.array(starts, dtype=mx.int32),
                mx.array(contexts, dtype=mx.int32))
        read = 2 * sum(contexts) * KV_HEADS * HEAD_DIM * 2
        out.append((f"paged_attention, {label}", 2 * q.nbytes + read,
                    lambda args=args: paged_attention(*args),
                    lambda args=args: paged_attention(*args, use_metal=True), None))

    # A decode step's write: 32 tokens, one per sequence, each into its own page. Fifty writes
    # chain inside one graph, since holding each result would stop MLX writing in place.
    slots = mx.arange(0, 32 * 16, 16, dtype=mx.int32)
    kv = normal(32, KV_HEADS, HEAD_DIM)
    for fp8 in (False, True):
        pool = PagedKvPool(1, 4096, 16, KV_HEADS, HEAD_DIM, dtype=BF16, fp8=fp8)

        def writes(use_metal: bool, pool: PagedKvPool = pool) -> list[mx.array]:
            for _ in range(WRITES):
                pool.write(0, slots, kv, kv, use_metal)
            return [pool.keys[0], pool.values[0]]

        # Reads bf16 keys and values, writes them back at 1 byte (fp8) or 2 (bf16) an element.
        out.append((f"{'fp8 ' if fp8 else ''}cache write, 32 tokens into 4096 pages",
                    kv.nbytes * (3 if fp8 else 4),
                    lambda writes=writes: writes(False), lambda writes=writes: writes(True), None))
    return out


def main() -> None:
    mx.random.seed(0)
    title = "Operators: Metal kernel vs pure MLX vs MLX's own"
    with evidence(title):
        ceiling = normal(64 * 2**20)  # 128 MB: a copy's read + write is the bandwidth ceiling
        copy_us = per_call_us(lambda: ceiling * 1, calls=4)
        copy_gbps = 2 * ceiling.nbytes / copy_us / 1e3
        print(f"Copy ceiling: {copy_gbps:.0f} GB/s (read + write of 128 MB).\n")

        print("| op | pure | metal | speedup | metal GB/s | MLX's own |")
        print("|---|---|---|---|---|---|")
        names, pure_times, metal_times, own_times = [], [], [], []
        for name, nbytes, pure, metal, own in cases():
            # A write case makes its own WRITES calls per graph; everything else is batched here.
            calls, per = (1, WRITES) if "write" in name else (50, 1)
            pure_us = per_call_us(pure, calls) / per
            metal_us = per_call_us(metal, calls) / per if METAL_BUILT else None
            own_us = per_call_us(own) if own else None
            print(f"| {name} | {pure_us:.0f} µs | "
                  + (f"{metal_us:.0f} µs | {pure_us / metal_us:.1f}x | {nbytes / metal_us / 1e3:.0f} | "
                     if metal_us else "— | — | — | ")
                  + (f"{own_us:.0f} µs |" if own_us else "— |"))
            names.append(name)
            pure_times.append(pure_us)
            metal_times.append(metal_us if metal_us else math.nan)
            own_times.append(own_us if own_us else math.nan)
        print("\nGB/s counts the bytes the op must move at least, so it is a floor on what it moved."
              " Per-call times batch 50 calls per mx.eval, so the per-eval launch cost is amortized.")

        figure = Figure(figsize=(10, 8))
        figure.suptitle(title)
        axes = figure.subplots()
        series = {"pure": pure_times, "metal": metal_times, "MLX's own": own_times}
        grouped_bars(axes, names, series, horizontal=True)
        axes.invert_yaxis()
        axes.set_xscale("log")
        axes.set_xlabel("µs per call (log)")
        save_plot(figure, "bench_operators")


if __name__ == "__main__":
    main()
