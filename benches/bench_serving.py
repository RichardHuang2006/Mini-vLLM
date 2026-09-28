"""Throughput by concurrency: output tokens per second over a whole request set, against
mlx_lm.batch_generate given the same prompts, batch width and output length.

    uv run python benches/bench_serving.py --concurrency 1 4 16 32
"""

from __future__ import annotations

import argparse
import time

import mlx_lm
from utils import engine, evidence, load_mlx_lm, random_prompts, variants

from mini_vllm import DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--prompt-lens", type=int, nargs="+", default=[32, 128, 256, 512])
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    with evidence(f"Throughput: 2 x concurrency requests, prompts of {args.prompt_lens} tokens, "
                  f"{args.max_tokens} output tokens each"):
        mlx_model, tokenizer = load_mlx_lm(args.model)
        rows = {concurrency: {} for concurrency in args.concurrency}

        for label, use_metal in variants():
            for concurrency in args.concurrency:
                # Twice the requests there are slots, so finished requests are replaced mid-run.
                prompts = random_prompts(2 * concurrency, args.prompt_lens, len(tokenizer.vocab))
                llm = engine(mlx_model, tokenizer, use_metal, num_blocks=2048, max_sequences=concurrency)
                start = time.perf_counter()
                completions = llm.generate(prompts, max_tokens=args.max_tokens)
                tokens = sum(len(c.token_ids) for c in completions)
                rows[concurrency][label] = tokens / (time.perf_counter() - start)

        mlx_lm.batch_generate(mlx_model, tokenizer, [[1] * 32] * 2, max_tokens=4)  # warm-up
        for concurrency in args.concurrency:
            prompts = random_prompts(2 * concurrency, args.prompt_lens, len(tokenizer.vocab))
            start = time.perf_counter()
            response = mlx_lm.batch_generate(
                mlx_model, tokenizer, prompts, max_tokens=args.max_tokens,
                completion_batch_size=concurrency, prefill_batch_size=min(8, concurrency),
            )
            rows[concurrency]["mlx_lm.batch_generate"] = (
                response.stats.generation_tokens / (time.perf_counter() - start)
            )

        labels = list(rows[args.concurrency[0]])
        print("| concurrency | " + " | ".join(labels) + " |")
        print("|---" * (len(labels) + 1) + "|")
        for concurrency, row in rows.items():
            print(f"| {concurrency} | " + " | ".join(f"{row[label]:,.0f} tok/s" for label in labels) + " |")


if __name__ == "__main__":
    main()
