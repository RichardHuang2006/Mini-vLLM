"""One request at a time, where nothing batches and only the kernels and the cache are working:
time to first token and decode rate, against mlx_lm's generate_step.

    uv run python benches/bench_generate.py --prompt-lens 128 1024 --max-tokens 128
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm.generate import generate_step
from utils import engine, evidence, load_mlx_lm, random_prompts, token_times, variants

from mini_vllm import DEFAULT_MODEL


def rates(times: list[float]) -> tuple[float, float]:
    """(TTFT in ms, decode tokens per second after the first)."""
    return times[0] * 1e3, (len(times) - 1) / (times[-1] - times[0])


def median_rates(runs: list[list[float]]) -> tuple[float, float]:
    ttfts, decodes = zip(*(rates(times) for times in runs), strict=True)
    return statistics.median(ttfts), statistics.median(decodes)


def mlx_lm_times(model, prompt: list[int], max_tokens: int) -> list[float]:
    """generate_step is the loop under stream_generate, which adds ~47 ms of setup per call
    (wiring the model's memory) that a server pays once, not per request."""
    start = time.perf_counter()
    steps = generate_step(mx.array(prompt), model, max_tokens=max_tokens)
    return [time.perf_counter() - start for _ in steps]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-lens", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3, help="the median of this many runs")
    args = parser.parse_args()

    with evidence(f"Single request: {args.max_tokens} output tokens, median of {args.repeats}"):
        mlx_model, tokenizer = load_mlx_lm(args.model)
        prompts = random_prompts(len(args.prompt_lens), args.prompt_lens, len(tokenizer.vocab))

        results = {}
        for label, use_metal in variants():
            llm = engine(mlx_model, tokenizer, use_metal, num_blocks=512)
            for prompt in prompts:
                runs = [token_times(llm.generate_stream(prompt, max_tokens=args.max_tokens))[0]
                        for _ in range(args.repeats)]
                results[label, len(prompt)] = median_rates(runs)

        mlx_lm_times(mlx_model, [1] * 32, 4)  # warm-up
        for prompt in prompts:
            runs = [mlx_lm_times(mlx_model, prompt, args.max_tokens) for _ in range(args.repeats)]
            results["mlx_lm generate_step", len(prompt)] = median_rates(runs)

        print("| engine | prompt tokens | TTFT | decode |")
        print("|---|---|---|---|")
        for (label, length), (ttft, decode) in results.items():
            print(f"| {label} | {length} | {ttft:.1f} ms | {decode:.0f} tok/s |")


if __name__ == "__main__":
    main()
