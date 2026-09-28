"""Prefix caching on and off: requests behind one long shared preamble, run one after another,
so each can reuse the pages the ones before it cached. A hit removes prompt tokens from the
prefill and leaves decoding alone, so time to first token is the metric it moves.

    uv run python benches/bench_prefix_cache.py --metal
"""

from __future__ import annotations

import argparse
import statistics
import time

from utils import engine, evidence, load_mlx_lm, random_prompts

from mini_vllm import DEFAULT_MODEL, LLM, RequestStatus


def run_alone(llm: LLM, prompt: list[int], max_tokens: int) -> tuple[float, int]:
    """(seconds to the first token, prompt tokens the prefix cache served) for one request."""
    start = time.perf_counter()
    request = llm.add_request(prompt, max_tokens=max_tokens)
    ttft = None
    while request.status is not RequestStatus.FINISHED:
        emitted, _ = llm.step()
        if ttft is None and emitted:
            ttft = time.perf_counter() - start
    return ttft, request.num_cached_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--metal", action="store_true", help="run on the Metal kernels in src/extensions/")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--preamble-len", type=int, default=2000)
    parser.add_argument("--question-len", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    with evidence(f"Prefix caching: {args.requests} requests, one after another, behind a shared "
                  f"{args.preamble_len}-token preamble, {'metal' if args.metal else 'pure'}"):
        mlx_model, tokenizer = load_mlx_lm(args.model)
        vocab = len(tokenizer.vocab)
        preamble = random_prompts(1, [args.preamble_len], vocab)[0]
        questions = random_prompts(args.requests, [args.question_len], vocab, seed=1)
        prompts = [preamble + question for question in questions]

        print("| prefix caching | first request TTFT | later requests' mean TTFT | prompt tokens from cache "
              "| total |")
        print("|---|---|---|---|---|")
        for enabled in (False, True):
            llm = engine(mlx_model, tokenizer, args.metal, num_blocks=1024, enable_prefix_caching=enabled)
            start = time.perf_counter()
            ttfts, cached = zip(*(run_alone(llm, prompt, args.max_tokens) for prompt in prompts), strict=True)
            total = time.perf_counter() - start
            print(f"| {'on' if enabled else 'off'} | {ttfts[0] * 1e3:.0f} ms | "
                  f"{statistics.mean(ttfts[1:]) * 1e3:.0f} ms | {sum(cached):,} of "
                  f"{sum(map(len, prompts)):,} | {total:.2f} s |")


if __name__ == "__main__":
    main()
