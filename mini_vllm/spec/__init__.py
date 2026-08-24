"""Speculative decoding: spend a cheap model's tokens to save the expensive one's passes.

A decode step is memory-bound: reading a 0.6B model's weights to produce one token leaves
the arithmetic units idle. The fix is to extract more tokens per pass rather than to make
the pass faster — a draft model proposes `k` tokens cheaply and the target scores all
`k + 1` positions in a single forward, since scoring existing tokens is what attention
parallelizes.

This must not change what the model says. :mod:`mini_vllm.spec.rejection` enforces that:
proposals are accepted or rejected by a rule that provably preserves the target's output
distribution, making speculation a latency optimization rather than a quality trade.
"""

from __future__ import annotations

__all__ = ["rejection"]
