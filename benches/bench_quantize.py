"""FP8 against BF16 KV pages. At a fixed KV memory budget, fp8 pages cost half the bytes, so
twice as many fit: more requests in flight, fewer preemptions. What it costs is fidelity,
measured as greedy output that stops matching bf16's.

    uv run python benches/bench_quantize.py --metal
"""

from __future__ import annotations

import argparse
import statistics
import time

from matplotlib.figure import Figure
from utils import engine, evidence, load_mlx_lm, random_prompts, save_plot

from mini_vllm import DEFAULT_MODEL, LLM

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Paged attention stores the KV cache in",
    "Once upon a time, in a small village,",
    "The three laws of thermodynamics are",
    "SELECT name, COUNT(*) FROM users",
    "To make a cup of tea, first",
    "Continuous batching improves throughput because",
]


def serve(llm: LLM, prompts: list[list[int]], max_tokens: int) -> tuple[float, int]:
    """(output tokens per second, the most requests running at once) for the whole set."""
    requests = [llm.add_request(prompt, max_tokens=max_tokens) for prompt in prompts]
    peak, start = 0, time.perf_counter()
    while llm.scheduler.has_work:
        llm.step()
        peak = max(peak, len(llm.scheduler.running))
    elapsed = time.perf_counter() - start
    return sum(len(r.output_token_ids) for r in requests) / elapsed, peak


def first_divergence(a: list[int], b: list[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), len(a))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--metal", action="store_true", help="run on the Metal kernels in src/extensions/")
    parser.add_argument("--kv-mb", type=int, default=512, help="the KV memory budget both dtypes get")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    title = (f"FP8 KV: {args.requests} requests of {args.prompt_len} + {args.max_tokens} tokens "
             f"in {args.kv_mb} MB of KV, {'metal' if args.metal else 'pure'}")
    with evidence(title):
        mlx_model, tokenizer = load_mlx_lm(args.model)
        a = mlx_model.args
        prompts = random_prompts(args.requests, [args.prompt_len], len(tokenizer.vocab))

        print("| KV pages | bytes per token | pages in budget | most requests in flight | throughput |")
        print("|---|---|---|---|---|")
        greedy = {}
        dtypes, rates, peaks = [], [], []
        for fp8 in (False, True):
            per_token = 2 * a.num_hidden_layers * a.num_key_value_heads * a.head_dim * (1 if fp8 else 2)
            num_blocks = args.kv_mb * 2**20 // (16 * per_token)
            llm = engine(mlx_model, tokenizer, args.metal, num_blocks=num_blocks,
                         max_sequences=args.requests, fp8_kv_cache=fp8)
            rate, peak = serve(llm, prompts, args.max_tokens)
            dtypes.append("fp8" if fp8 else "bf16")
            rates.append(rate)
            peaks.append(peak)
            print(f"| {'fp8' if fp8 else 'bf16'} | {per_token:,} B | {num_blocks} | {peak} "
                  f"| {rate:,.0f} tok/s |")
            # Each prompt alone, so the only difference between the two runs is the page dtype.
            greedy[fp8] = [llm.generate(prompt, max_tokens=64)[0].token_ids for prompt in PROMPTS]

        divergences = [first_divergence(b, f) for b, f in zip(greedy[False], greedy[True], strict=True)]
        print(f"\nGreedy, 64 tokens on {len(PROMPTS)} text prompts: fp8 matches bf16 exactly on "
              f"{sum(d == 64 for d in divergences)}; the median first difference is at token "
              f"{statistics.median(divergences):.0f}.")

        figure = Figure(figsize=(9, 4))
        figure.suptitle(title)
        rate_axes, peak_axes = figure.subplots(1, 2)
        rate_axes.bar(dtypes, rates)
        rate_axes.set_ylabel("throughput (tok/s)")
        peak_axes.bar(dtypes, peaks)
        peak_axes.set_ylabel("most requests in flight")
        save_plot(figure, "bench_quantize")


if __name__ == "__main__":
    main()
