"""Serve many prompts at once through the engine and report throughput.

    uv run python batch-main.py --concurrency 32 --max-tokens 128
"""

import argparse
import time

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

parser = argparse.ArgumentParser(description="Serve many prompts at once.")
parser.add_argument("--model", default=DEFAULT_MODEL)
parser.add_argument("--concurrency", type=int, default=32, help="requests in flight at once")
parser.add_argument("--max-tokens", type=int, default=128)
parser.add_argument("--num-blocks", type=int, default=2048, help="KV pages of 16 tokens")
parser.add_argument("--prefix-caching", action="store_true")
parser.add_argument("--fp8-kv-cache", action="store_true")
parser.add_argument("--metal", action="store_true", help="run on the Metal kernels in src/extensions/")
args = parser.parse_args()

llm = LLM.from_pretrained(
    args.model,
    use_metal=args.metal,
    num_blocks=args.num_blocks,
    max_sequences=args.concurrency,
    enable_prefix_caching=args.prefix_caching,
    fp8_kv_cache=args.fp8_kv_cache,
)
prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.concurrency)]
llm.generate(prompts[:2], max_tokens=4)  # warm-up: the first pass pays one-time setup

started = time.perf_counter()
# ignore_eos, so every request runs to max_tokens and the batch measures the engine, not the prompts.
completions = llm.generate(prompts, max_tokens=args.max_tokens, ignore_eos=True)
elapsed = time.perf_counter() - started

for completion in completions[:3]:
    print(f"{completion.prompt!r} -> {completion.text[:80]!r}")
tokens = sum(len(c.token_ids) for c in completions)
print(f"\n{len(completions)} requests, {tokens} tokens in {elapsed:.2f}s: {tokens / elapsed:.0f} tok/s")
