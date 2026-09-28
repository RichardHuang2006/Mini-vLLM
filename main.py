"""Generate from one prompt with the Mini-vLLM engine.

    uv run python main.py "The capital of France is" --stream
"""

import argparse
import time

from mini_vllm import DEFAULT_MODEL, LLM, SamplingParams

parser = argparse.ArgumentParser(description="Generate from one prompt.")
parser.add_argument("prompt", nargs="?", default="The capital of France is")
parser.add_argument("--model", default=DEFAULT_MODEL)
parser.add_argument("--max-tokens", type=int, default=64)
parser.add_argument("--temperature", type=float, default=0.0, help="0 is greedy")
parser.add_argument("--stream", action="store_true", help="print each token as it arrives")
parser.add_argument("--num-blocks", type=int, default=1024, help="KV pages of 16 tokens")
parser.add_argument("--fp8-kv-cache", action="store_true")
args = parser.parse_args()

llm = LLM.from_pretrained(args.model, num_blocks=args.num_blocks, fp8_kv_cache=args.fp8_kv_cache)
params = SamplingParams(temperature=args.temperature)

started = time.perf_counter()
if args.stream:
    print(args.prompt, end="", flush=True)
    token_ids = []
    for update in llm.generate_stream(args.prompt, params, args.max_tokens):
        print(update.text, end="", flush=True)
        token_ids.append(update.token_id)
    print()
    count = len(token_ids)
else:
    completion = llm.generate(args.prompt, params, args.max_tokens)[0]
    print(args.prompt + completion.text)
    count = len(completion.token_ids)
elapsed = time.perf_counter() - started

print(f"\n{count} tokens in {elapsed:.2f}s ({count / elapsed:.1f} tok/s)")
