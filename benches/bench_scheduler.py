"""Chunked prefill under an adversarial mix: short requests arriving at random, with a long prompt
every twentieth, replayed through each chunk size. A long prompt run in one pass stalls every
decode in flight for that pass; chunking bounds the stall at one chunk.

    uv run python benches/bench_scheduler.py --metal
"""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from itertools import pairwise

from matplotlib.figure import Figure
from utils import engine, evidence, grouped_bars, load_mlx_lm, percentile, random_prompts, save_plot

from mini_vllm import DEFAULT_MODEL, LLM, Request


def replay(llm: LLM, prompts: list[list[int]], arrivals: list[float], max_tokens: int):
    """Add each prompt at its arrival time and step until every request is done. Returns each
    request's arrival and token times, in seconds from the start."""
    pending = deque(zip(arrivals, prompts, strict=True))
    arrived: dict[Request, float] = {}
    times: dict[Request, list[float]] = {}
    start = time.perf_counter()
    while pending or llm.scheduler.has_work:
        now = time.perf_counter() - start
        while pending and pending[0][0] <= now:
            at, prompt = pending.popleft()
            request = llm.add_request(prompt, max_tokens=max_tokens)
            arrived[request], times[request] = at, []
        if not llm.scheduler.has_work:
            time.sleep(pending[0][0] - now)
            continue
        emitted, _ = llm.step()
        now = time.perf_counter() - start
        for request, _ in emitted:
            times[request].append(now)
    return arrived, times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--metal", action="store_true", help="run on the Metal kernels in src/extensions/")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--rate", type=float, default=10.0, help="mean arrivals per second (Poisson)")
    parser.add_argument("--short-len", type=int, default=32)
    parser.add_argument("--long-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[128, 512])
    args = parser.parse_args()

    title = (f"Scheduler: {args.requests} requests at {args.rate:g}/s, {args.short_len}-token "
             f"prompts with a {args.long_len}-token one every 20th, {args.max_tokens} output "
             f"tokens each, {'metal' if args.metal else 'pure'}")
    with evidence(title):
        mlx_model, tokenizer = load_mlx_lm(args.model)
        vocab = len(tokenizer.vocab)
        lengths = [args.long_len if i % 20 == 19 else args.short_len for i in range(args.requests)]
        prompts = [random_prompts(1, [length], vocab, seed=i)[0] for i, length in enumerate(lengths)]
        rng, at, arrivals = random.Random(0), 0.0, []
        for _ in prompts:
            at += rng.expovariate(args.rate)
            arrivals.append(at)

        # Each chunk size, then the whole prompt in one pass: the budget leaves room for it
        # beside a full batch of decodes, so it runs as soon as it arrives rather than waiting.
        policies = [(f"chunks of {size}", {"chunk_size": size}) for size in args.chunk_sizes]
        policies.append(("one pass", {"enable_chunked_prefill": False,
                                      "max_batched_tokens": args.long_len + 32}))

        print("| prefill | TTFT p50 | TTFT p99 | inter-token p50 | inter-token p99 | inter-token max |")
        print("|---|---|---|---|---|---|")
        ttft_ms = {"p50": [], "p99": []}
        gap_ms = {"p50": [], "p99": [], "max": []}
        for label, config in policies:
            llm = engine(mlx_model, tokenizer, args.metal, num_blocks=4096, max_sequences=32, **config)
            arrived, times = replay(llm, prompts, arrivals, args.max_tokens)
            ttfts = [times[r][0] - arrived[r] for r in times]
            gaps = [(b - a) * 1e3 for t in times.values() for a, b in pairwise(t)]
            ttft_p50 = percentile(ttfts, 50) * 1e3
            ttft_p99 = percentile(ttfts, 99) * 1e3
            gap_p50 = percentile(gaps, 50)
            gap_p99 = percentile(gaps, 99)
            gap_max = max(gaps)
            print(f"| {label} | {ttft_p50:.0f} ms | {ttft_p99:.0f} ms"
                  f" | {gap_p50:.1f} ms | {gap_p99:.1f} ms | {gap_max:.1f} ms |")
            ttft_ms["p50"].append(ttft_p50)
            ttft_ms["p99"].append(ttft_p99)
            gap_ms["p50"].append(gap_p50)
            gap_ms["p99"].append(gap_p99)
            gap_ms["max"].append(gap_max)

        figure = Figure(figsize=(11, 4))
        figure.suptitle(title)
        ttft_axes, gap_axes = figure.subplots(1, 2)
        labels = [label for label, _ in policies]
        grouped_bars(ttft_axes, labels, ttft_ms)
        ttft_axes.set_ylabel("TTFT (ms)")
        grouped_bars(gap_axes, labels, gap_ms)
        gap_axes.set_ylabel("inter-token latency (ms)")
        save_plot(figure, "bench_scheduler")


if __name__ == "__main__":
    main()
