# Mini-vLLM

A single-GPU LLM inference engine for Qwen3-0.6B, written in Python, PyTorch and CUDA C++. It
serves continuously batched requests out of a paged KV cache: fixed-size 16-token pages that any
sequence can hold in any order, addressed through a per-sequence block table that the attention
kernels walk in-kernel, so a batch of ragged sequences is one forward pass rather than one per
sequence. The scheduler re-forms the batch every iteration and splits long prompts into chunks so a
2048-token prefill cannot monopolize an iteration. Pages are reference-counted and shared with
copy-on-write, and indexed by a radix tree so requests behind a common preamble reuse its keys and
values. Seven hand-written CUDA kernels replace the PyTorch expressions they are diffed against,
each reachable through a dispatch table that can be flipped back to the reference implementation at
runtime. The cache can optionally be stored in FP8, and speculative decoding with a draft model is
verified by rejection sampling so it changes latency rather than output distribution.

## Key features

- **Paged KV cache with block-table indirection.** Fixed 16-token pages in a single pre-allocated
  pool; a per-sequence block table maps token position to page id and offset. Internal
  fragmentation is bounded to one partial page per sequence by construction.
- **Copy-on-write block sharing.** Reference-counted pages let a forked sequence share its parent's
  blocks until one of them writes, at which point only the written page is copied. This is the path
  `SamplingParams(n=k)` takes: one prefill, `k` branches.
- **Radix-tree prefix caching.** A radix tree over block-aligned token hashes lets a prompt that
  begins with one already served reuse those pages. Cached blocks sit at reference count zero on
  the free list, still referenced by the tree, and are reclaimed only under pressure.
- **Continuous batching scheduler** with iteration-level scheduling and preemption, chunked prefill,
  and piggyback decoding: every iteration re-forms the batch under a token budget, admitting decodes
  first, then prefill chunks in the remaining budget, then new requests. A sequence whose next step
  the pool cannot back is preempted and recomputed later.
- **Fused CUDA kernels** for RMSNorm, RoPE and SwiGLU, plus FlashAttention-style attention: decode
  attention using online softmax so the score row is never materialized, tiled causal prefill with
  K/V tiles staged through shared memory, and paged attention for both shapes that gathers K and V
  through the block table inside the kernel.
- **FP8 KV quantization.** `kv_cache_dtype="fp8"` stores the cache in e4m3 with per-tensor scales,
  halving the bytes a page costs. A fused quantize-and-scatter kernel writes it in one pass, and
  because the scales are scalars the paged attention kernel needs no change to its inner loops: the
  key scale folds into the softmax scale at the launch site and the value scale rides on the output,
  so an FP8 element costs only its conversion on load.
- **Speculative decoding with a draft model,** verified by rejection sampling. A proposal is
  accepted with probability `min(1, p/q)`; the first rejection ends the step with one draw from the
  residual `norm(relu(p - q))`; a fully accepted run gets a bonus token from `p_{k+1}`, which the
  same target pass already computed. The emitted tokens are distributed exactly as the target
  model's own.

Every fast path has a slow, obviously correct counterpart, and the slow one serves as a test oracle
rather than an unexercised fallback. Each kernel is diffed against the PyTorch expression it
replaces, the paged model against the dense model, and the engine against `transformers.generate`
with greedy output matching token for token. No architectural novelty is claimed; this is a correct,
measured single-GPU implementation of the ideas in vLLM's PagedAttention and continuous batching.

## Installation

Requires Python 3.13 and a CUDA GPU with a couple of GB free. The KV pool sizes itself to the memory
left after the weights, so a smaller card serves fewer concurrent requests rather than failing.

```bash
make setup     # create .venv and install pinned deps (CUDA 13 torch)
```

The pinned `torch` is a CUDA 13 build, which is not on the default PyPI index; `make setup` adds
`https://download.pytorch.org/whl/cu130`. The CUDA major version of torch must match the nvcc that
compiles `csrc/`, and on a machine whose system nvcc is older (12.8 here) `torch.utils.cpp_extension`
refuses to build. The four `nvidia-cuda-*` wheels in `requirements.txt` supply a CUDA 13 toolchain
instead, and must be pinned to the same minor; `requirements.txt` documents the two failure modes
that come from mixing them.

### Building the CUDA extension

`csrc/` is compiled on first use by PyTorch's JIT loader, so there is no separate build step.
`mini_vllm/kernels/extension.py` locates a matching toolchain, assembling a `CUDA_HOME` out of the
pinned nvcc wheels when no system toolkit matches, and caps build parallelism so nvcc's peak memory
does not exhaust a 16-core, 15 GB WSL2 allocation.

```bash
make ext       # force a full rebuild and print the resolved toolchain
```

A stale JIT cache is the first thing to suspect when a kernel edit appears to have no effect, which
is what this target rules out.

## Quickstart

```python
from mini_vllm import LLM
from mini_vllm.sampler import SamplingParams

llm = LLM("Qwen/Qwen3-0.6B")

for completion in llm.generate(["The capital of France is"], max_tokens=32):
    print(completion.text)

# streaming, interleaved across requests as the engine produces them
for update in llm.generate_stream(prompts, max_tokens=128):
    print(update.index, update.text, end="", flush=True)
```

The optional features are off by default and each is one argument:

```python
llm = LLM(
    "Qwen/Qwen3-0.6B",
    enable_prefix_caching=True,   # reuse pages behind a shared preamble
    kv_cache_dtype="fp8",         # half the bytes per page, dequantized in-kernel
    num_speculative_tokens=4,     # propose 4, verify by rejection sampling
    use_cuda_kernels=True,        # route ops through csrc/ where a kernel is faster
)
llm.generate(prompt, sampling_params=SamplingParams(temperature=0.8, n=4))  # 4 branches, one prefill
```

`use_cuda_kernels=False` runs the same code path against the PyTorch references, which is how the
tests establish that the kernels changed the speed and not the output.

## Architecture

### Model and reference implementations (`mini_vllm/`)

- `basics.py`, `layer_norm.py`, `positional_encoding.py`, `attention.py`, `embedding.py` — reference
  implementations kept as oracles: `linear`/`silu`/`softmax`, RMSNorm, the RoPE tables and rotation,
  grouped-query attention, and the embedding table used in both directions.
- `model/loader.py` — `ModelConfig` read from `config.json`, HuggingFace-to-local weight name mapping
  with an expected shape for every tensor, and safetensors loading. A checkpoint that is missing,
  extra, or misshapen fails at load with the name of the offending tensor.
- `model/qwen3.py` — the dense model: 28 layers, 16 query heads over 8 key/value heads, per-head
  QK-norm before RoPE, SwiGLU MLP, RoPE at base 1e6. Recomputes the whole prefix every step, making
  it the slowest and most trustworthy implementation here.
- `model/qwen3_cached.py` — the same model over a KV cache, with the position `offset` machinery that
  lets a forward pass start partway into a sequence, which is what chunked prefill needs.
- `model/qwen3_paged.py` — the model the engine runs: one ragged token axis, paged attention, and no
  per-sequence loop in the forward pass.
- `kv_cache.py`, `paged_attention.py` — the dense per-sequence cache, and a gather-then-dense paged
  attention written to be obviously correct. The latter is the oracle the paged kernels are diffed
  against; performing its gather in production would copy every sequence's cache once per iteration.
- `kernels/extension.py`, `kernels/ops.py` — the JIT extension loader and the dispatch table. Every
  op takes `use_cuda`, so kernels on and kernels off are the same code path with a different callee,
  and `dispatch_report` prints which ops ran as kernels.
- `sampler.py`, `generate.py` — `SamplingParams` with greedy, temperature, top-k and top-p sampling
  batched per request in one call, and the reference generation loop the engine is checked against.
- `bench.py` — six measurement modes: single-request latency, per-kernel bandwidth, throughput
  against `transformers.generate`, a scheduler stress test, prefix caching on against off, and a
  speculation sweep over draft depth. It samples GPU clocks during a run and voids its own output if
  the card was throttled.

### CUDA kernels (`csrc/`)

- `kernel_utils.cuh` — launch geometry, warp and block reductions, and vectorized load helpers.
  Every kernel accumulates in fp32 regardless of storage dtype.
- `rmsnorm.cu`, `rope.cu`, `swiglu.cu` — the memory-bound elementwise trio, each a single pass over
  its data at 73–81% of theoretical bandwidth.
- `decode_attention.cu` — one query against a whole context by online softmax: running max and sum
  updated per key tile, so the score vector is never materialized. Split over the key axis when the
  context is long enough that one block per head leaves the GPU idle.
- `flash_prefill.cu` — tiled causal attention for query length > 1, K/V tiles staged through shared
  memory, tiles above the diagonal skipped rather than computed and masked.
- `paged_attention.cu` — the same two shapes, gathering K and V through a block table inside the
  kernel instead of from a contiguous tensor. The decode path splits over the key axis; the prefill
  path scores eight keys per step so the cross-lane reductions have independent work to overlap.
- `kv_quantize.cu` — quantize-and-scatter in one pass: divide by the scale, cast to FP8, and store
  straight to the slot the token maps to. The PyTorch it replaces builds a whole quantized temporary
  and then scatters it, two reads and two writes for a few flops an element.
- `bindings.cpp`, `hello.cu` — the pybind11 surface, and an `axpby` smoke kernel that exercises the
  toolchain before anything depends on it.

Kernels pairing a query with an FP8 cache are instantiated only where they are reachable
(`if constexpr`, e4m3 only), holding `paged_attention.cu` to five template instantiations instead of
nine. nvcc's peak memory on this file decides whether the build survives on this machine, which is
also why build parallelism is capped in `kernels/extension.py`.

### Paged KV cache (`mini_vllm/block/`)

- `block_pool.py` — allocation and reference counting over physical block ids. Reference counts are
  what make sharing possible; the free count drives admission control and preemption.
- `block_table.py` — one sequence's logical-to-physical mapping: token position to block id and
  offset within it.
- `kv_pool.py` — the physical storage the block ids index into: a single pre-allocated tensor with
  the layer axis inside it, so the whole cache's size is known up front and can be chosen from the
  memory left after the weights. Written by slot mapping rather than by sequence. Optionally stored
  in FP8 with per-tensor scales, which halves the bytes a page costs and so doubles the pages a
  given budget buys.
- `block_manager.py` — capacity, growth, sharing and release, plus copy-on-write: a forked sequence
  shares its parent's blocks until one writes, and only the written block is copied. Also `trim`,
  which returns the pages a rejected speculative proposal no longer needs, the one place a page is
  released mid-sequence.
- `prefix_cache.py` — the radix tree over block-aligned token hashes: `match` for the longest cached
  prefix of a prompt, `insert` for the blocks a completed prefill filled, and `evict` for when the
  pool reclaims a cached page. A cached block sits at reference count zero on the free list and stays
  matchable until it is actually reused.

### Serving (`mini_vllm/serve/`)

- `sequence.py` — per-request state: prompt and output tokens, how many are computed, and the status
  transitions. The distinction between computed and present is what makes chunked prefill and
  preemption expressible. Speculative proposals live here as well, outside the output, so a guessed
  end-of-text token cannot finish a request the target model was about to disagree with.
- `batch.py` — the ragged `ForwardBatch`: sequences of different lengths flattened onto one token
  axis with offsets, carrying the slot mapping and block tables the kernels need. Built entirely from
  host integers, since reading a device tensor here costs more than the model does.
- `scheduler.py` — continuous batching: every iteration re-forms the batch under a token budget,
  piggybacking decodes onto prefill chunks, preempting the newest sequence when the pool cannot back
  its next step and recomputing it later. Also holds a dense per-sequence runner used to show what
  scheduling without paging buys.
- `runner.py`, `engine.py` — the paged model runner, and the `LLM` API: admission, `step()`,
  `generate` and `generate_stream`. The batch call is the streaming call, drained, so the two cannot
  disagree.

### Speculative decoding (`mini_vllm/spec/`)

- `rejection.py` — the acceptance rule and the residual distribution. At temperature zero it reduces
  to keeping the prefix where the two models' argmaxes agree, with no randomness anywhere, which is
  why greedy speculative output is identical to greedy non-speculative output.
- `proposer.py` — the draft model, its own KV pool, and one shadow sequence per request holding the
  draft's block table. Resynchronization is the subtle part: after verification the draft's cache
  holds tokens the sequence no longer contains, so the divergence point is found and everything past
  it is returned. An error there is silent, collapsing acceptance while the output stays correct.
- `spec_decode.py` — one iteration: propose before the scheduler runs (proposals need reserved
  pages), verify with a single `all_rows=True` target pass, accept, and roll back the rejected tail in
  the same iteration so no page is held by a token that does not exist.

## Benchmarks

Measured on an RTX 5070 Laptop (8 GB, Blackwell `sm_120`, 384 GB/s theoretical) with Qwen3-0.6B in
bf16 and CUDA 13.

```bash
make bench               # single-request TTFT and decode tok/s, against transformers
make bench-throughput    # output tok/s vs transformers, by concurrency
make bench-scheduler     # decode-latency tails: chunked prefill vs prefill-first
make bench-kernels       # achieved bandwidth per kernel vs the torch it replaced
make bench-prefix-cache  # TTFT with and without radix-tree prefix caching
make bench-spec          # speculative decoding: acceptance and wall clock by draft depth
```

### Throughput

Output tokens per second over a whole request set, EOS ignored so both engines do identical work,
prompts cycling through 32–512 tokens (`make bench-throughput`):

| Concurrency | Mini-vLLM | `transformers.generate` | Speedup |
|---|---|---|---|
| 1 | 98 tok/s | 38 tok/s | 2.5x |
| 4 | 349 tok/s | 130 tok/s | 2.7x |
| 16 | 1062 tok/s | 292 tok/s | 3.6x |
| 32 | **1688 tok/s** | 152 tok/s | **11.1x** |

The comparison is asymmetric by construction, and that asymmetry is the result: `generate` takes one
padded rectangle, so prompts of 32 to 512 tokens all run for 512 and its rate falls past batch 16 as
padding grows, while the engine gives each sequence its own length and admits a replacement in the
iteration a request finishes. Single-request latency, where there is nothing to batch and only the
kernels and the cache are working, is 19.3 ms to first token and 107 tok/s decoding, against 25.6 ms
and 40 tok/s.

### Kernels

RMSNorm and SwiGLU reach 81% of theoretical bandwidth and RoPE 73%, against 79% for a bare `copy_`,
which is the practical ceiling for an op that touches memory once. Paged decode attention is 38x the
PyTorch it replaces at batch 16, almost all of it from not copying every sequence's cache per
iteration. Prefill attention is the one regression: paged prefill measures 0.54x a gather-and-cuBLAS
oracle at L=512/S=2048, and dense `flash_prefill` 0.4x cuBLAS at L=512, because the oracle's matmuls
run on tensor cores while these kernels' inner loops run on the FP32 pipes. `flash_prefill` is
therefore listed in `ops.NOT_YET_FASTER`, which keeps the dispatch on the PyTorch path with the
measured reason attached and is what `dispatch_report` prints. An `mma.sync`-based inner loop is the
largest remaining performance item and nothing else depends on it.

The fused FP8 quantize-and-scatter is 14x the two-pass PyTorch path for a decode step and 6.8x past
L2, and it runs once per layer per iteration, so 28 launches per step. An FP8 cache costs exactly
half the bytes per page, so a given memory budget holds twice the pages and therefore twice the
context or twice the concurrent requests.

### Scheduler

Under an adversarial mix — many 32-token requests with a 2048-token prompt dropped in every
twentieth, Poisson arrivals near capacity, 2000 requests replayed through both policies
(`make bench-scheduler`) — chunked prefill cuts the worst decode gap 2.4x (259 ms against 613 ms) at
identical throughput and with no leaked blocks. Total waiting is unchanged: chunking redistributes
the stall, many decodes waiting one chunk each instead of a few waiting an entire prompt, and what it
removes is the unbounded case. Whether the improvement also appears at P99 depends on how many
decodes are in flight when a long prompt lands, so the benchmark reports both tails rather than
asserting a direction.

### Prefix caching

Over six requests behind one long shared preamble (`make bench-prefix-cache`), prefix caching cuts
mean TTFT 1.26x, 69 ms against 88 ms, with 3088 prompt tokens served from cache. Whole-batch time
barely moves, which is the expected result: a cache hit removes prompt tokens from the prefill and
leaves decode unchanged, so a run whose cost is decode cannot show it. TTFT is the metric the feature
targets.

### Speculative decoding

`make bench-spec`, one greedy request, 64 output tokens:

| Draft | Acceptance | Tokens per target pass | Seconds |
|---|---|---|---|
| none | — | 1.00 | 5.67 |
| 4 of 28 layers | 0.000 | 1.00 | 8.42 |
| 20 of 28 layers | 0.161 | 1.57 | 11.74 |
| 28 of 28 layers (self-draft) | 0.982 | 4.71 | 5.30 |

The mechanism behaves as designed — at full depth the draft is accepted 98% of the time and each
expensive pass yields 4.7 tokens instead of 1 — and still buys almost nothing, because a draft deep
enough to be accepted costs what the target costs, while truncating it to four layers makes it cheap
and it is then rejected essentially always. What speculative decoding needs is a separate, genuinely
smaller checkpoint. Qwen3-0.6B is the smallest of its family, so on 8 GB the configuration that would
pay — a 0.6B draft against a 1.7B target, via `LLM(..., draft_model=...)` — is supported but not
measured here. The self-draft row is retained because acceptance near 1.0 with identical output is
the strongest available correctness statement for the verification path.

## Testing and validation

```bash
make test        # 927 tests; GPU and checkpoint tests skip when unavailable
make test-cpu    # the subset needing neither a GPU nor weights
```

927 tests, differential at the core. Each kernel is pinned to the PyTorch expression it replaces on
value (fp32 exactly; bf16 and fp16 to dtype-appropriate tolerances defined once in `conftest.py`) and
on the property that matters more than the value: the kernel changed the speed, not the text. Above
the kernels, the cached model is diffed against the recomputing model, the paged model against the
cached one, paged attention against a gather-then-dense oracle, and the engine against
`transformers.generate` on sixteen varied-length prompts. In fp32 that identity is exact; in bf16 it
usually is, and where it is not, a separate test establishes that the divergent token was a numerical
tie — the second choice in fp32, one rounding from the first — rather than a bug.

Behavioral gates cover what numerics cannot: the block pool must leak nothing over three thousand
requests, a pool small enough to force preemption must still finish every request, scheduler policies
must not change the tokens produced, and head-of-line stalls are asserted in tokens the engine
computed between two of a sequence's tokens rather than in milliseconds, which makes the claim
deterministic rather than dependent on machine load.

Three of the newer features are checked by equality. Prefix caching must produce the same tokens as
an engine with it switched off. Parallel sampling's `n` greedy branches must each equal a solo run.
Greedy speculative decoding must be token-identical to greedy non-speculative decoding, including
with a deliberately poor draft, since rejection sampling is what makes the draft's quality irrelevant
to correctness. Distribution preservation, which no equality can show, is tested by chi-square over
120k samples against a draft whose distribution is the target's reversed: a correct implementation
scores about 1.8 against a threshold of 18.5, and one that simply trusted the draft scores 368,000.

Because the GPU of record has 8 GB, real-weight engines go through one context manager
(`conftest.real_engine`) that pins `num_blocks` and empties the caching allocator on exit. Two
engines do not fit at once, and that failure appears as an out-of-memory abort with no failing
assertion, so the discipline is structural rather than remembered.

## License

MIT. See [LICENSE](LICENSE).
