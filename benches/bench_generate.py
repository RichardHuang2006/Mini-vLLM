"""One request at a time, where nothing batches and only the kernels and the cache are working:
time to first token and decode rate, against mlx_lm's generate_step.

    uv run python benches/bench_generate.py --prompt-lens 128 1024 --max-tokens 128
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from matplotlib.figure import Figure
from mlx_lm.generate import generate_step
from utils import (
    engine,
    evidence,
    grouped_bars,
    load_mlx_lm,
    random_prompts,
    save_plot,
    token_times,
    variants,
)

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

    title = f"Single request: {args.max_tokens} output tokens, median of {args.repeats}"
    with evidence(title):
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

        labels = list(dict.fromkeys(label for label, _ in results))
        lengths = args.prompt_lens
        groups = [str(length) for length in lengths]
        figure = Figure(figsize=(11, 4))
        figure.suptitle(title)
        ttft_axes, decode_axes = figure.subplots(1, 2)
        ttft_series = {label: [results[label, n][0] for n in lengths] for label in labels}
        decode_series = {label: [results[label, n][1] for n in lengths] for label in labels}
        grouped_bars(ttft_axes, groups, ttft_series)
        grouped_bars(decode_axes, groups, decode_series)
        ttft_axes.set_ylabel("TTFT (ms)")
        decode_axes.set_ylabel("decode (tok/s)")
        for axes in (ttft_axes, decode_axes):
            axes.set_xlabel("prompt tokens")
        save_plot(figure, "bench_generate")


if __name__ == "__main__":
    main()
