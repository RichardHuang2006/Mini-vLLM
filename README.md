# Mini-vLLM

A paged-attention LLM inference engine for Qwen3 on Apple silicon, written in MLX and hand-written
Metal. It serves continuously batched requests out of a paged KV cache, re-forming the batch every
iteration so ragged sequences run in one forward pass, and every Metal kernel has a pure-MLX
counterpart that serves as its test oracle.

## Key features

- **Paged KV cache** of 16-token pages, with copy-on-write sharing for parallel sampling.
- **Radix-tree prefix caching**, so requests behind a shared preamble reuse its pages.
- **Continuous batching** with chunked prefill and preemption.
- **Metal kernels** for RMSNorm, RoPE, SwiGLU, dense and paged attention, and the cache writes.
- **FP8 KV cache**: half the bytes per page.
- **Speculative decoding**, verified by rejection sampling so the output distribution is unchanged.

## Usage

```bash
uv sync
uv run python src/extensions/build.py    # the Metal kernels, optional
```

```python
from mini_vllm import LLM, SamplingParams

llm = LLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    use_metal=True,               # needs the Metal kernels built
    enable_prefix_caching=True,
    fp8_kv_cache=True,
    num_speculative_tokens=4,
)

for completion in llm.generate(["The capital of France is"], max_tokens=32):
    print(completion.text)

for update in llm.generate_stream(["Once upon a time,"], SamplingParams(temperature=0.8), max_tokens=64):
    print(update.text, end="", flush=True)
```

```bash
uv run python main.py "Explain paged attention." --stream --metal
uv run python batch-main.py --concurrency 32 --metal
uv run pytest
```
