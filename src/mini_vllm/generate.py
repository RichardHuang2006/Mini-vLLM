"""Single-request generation loops: the oracles every batched and paged loop is checked against."""

from __future__ import annotations

from collections.abc import Collection

import mlx.core as mx

from mini_vllm.kv_cache import KvFullCache
from mini_vllm.qwen3 import Qwen3Model
from mini_vllm.sampler import SamplingParams, sample

__all__ = ["GREEDY", "generate_with_kv_cache", "simple_generate"]

GREEDY = SamplingParams(temperature=0.0)


def simple_generate(
    model: Qwen3Model,
    prompt: list[int],
    max_tokens: int,
    params: SamplingParams = GREEDY,
    eos_token_ids: Collection[int] = (),
) -> list[int]:
    """Recompute the whole sequence for every new token: quadratic, and needing no cache."""
    tokens = list(prompt)
    for _ in range(max_tokens):
        logits = model(mx.array([tokens]), mx.arange(len(tokens)))
        token = sample(logits[:, -1], params).item()
        tokens.append(token)
        if token in eos_token_ids:
            break
    return tokens[len(prompt) :]


def generate_with_kv_cache(
    model: Qwen3Model,
    prompt: list[int],
    max_tokens: int,
    params: SamplingParams = GREEDY,
    eos_token_ids: Collection[int] = (),
    prefill_chunk: int | None = None,
) -> list[int]:
    """Prefill the prompt, prefill_chunk tokens at a time, then forward one token per step."""
    caches = [KvFullCache() for _ in model.layers]

    chunk = prefill_chunk or len(prompt)
    for start in range(0, len(prompt), chunk):
        piece = prompt[start : start + chunk]
        logits = model(mx.array([piece]), mx.arange(start, start + len(piece)), caches)
        # MLX is lazy: evaluate each chunk's cache so one graph never spans the whole prompt.
        mx.eval([(cache.keys, cache.values) for cache in caches])

    generated = []
    while True:
        token = sample(logits[:, -1], params).item()
        generated.append(token)
        if token in eos_token_ids or len(generated) == max_tokens:
            return generated
        # The new token sits at the position right after everything cached.
        logits = model(mx.array([[token]]), mx.array([caches[0].offset]), caches)
