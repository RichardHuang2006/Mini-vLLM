# Mini-vLLM

A single-GPU LLM inference engine for Qwen3-0.6B in Python, PyTorch, and CUDA C++, built from the
model outwards. It serves continuously batched requests out of a paged KV cache: fixed-size
16-token pages that any sequence can hold in any order, addressed through a per-sequence block
table that the attention kernels walk in-kernel, so a batch of ragged sequences is one forward
pass rather than one per sequence. The scheduler re-forms the batch every iteration — decodes
first, then prefill chunks in whatever token budget is left, then new admissions — and splits long
prompts into chunks so a 2048-token prefill cannot monopolize an iteration and stall everyone
else's decode. Six hand-written CUDA kernels (RMSNorm, RoPE, SwiGLU, decode attention with online
softmax, tiled causal prefill, and paged attention for both decode and prefill) replace the
PyTorch expressions they were diffed against, each reachable through a dispatch table that can be
flipped back to the readable version at runtime.

The method throughout: every fast path has a slow, obviously-correct twin, and the slow one is a
test oracle rather than a fallback that happens to exist. Each kernel is diffed against the
PyTorch expression it replaces, the paged model against the dense model, and the engine against
`transformers.generate` — greedy output matching token for token. No architectural novelty is
claimed; the point is a correct, measured, single-GPU implementation of the ideas in vLLM's
PagedAttention and continuous batching.

```python
from mini_vllm import LLM

llm = LLM("Qwen/Qwen3-0.6B")

for completion in llm.generate(["The capital of France is"], max_tokens=32):
    print(completion.text)

# streaming, interleaved across requests as the engine produces them
for update in llm.generate_stream(prompts, max_tokens=128):
    print(update.index, update.text, end="", flush=True)
```

## Architecture

### The model and its references (`mini_vllm/`)

- `basics.py`, `layer_norm.py`, `positional_encoding.py`, `attention.py`, `embedding.py` — the
  readable reference implementations, correct by inspection and kept as oracles: `linear`/`silu`/
  `softmax`, RMSNorm, the RoPE tables and rotation, grouped-query attention, and the embedding
  table used in both directions.
- `model/loader.py` — the `ModelConfig` read from `config.json`, HuggingFace-to-local weight name
  mapping with an expected shape for every tensor, and safetensors loading. A checkpoint that is
  missing, extra, or misshapen fails at load with the name of the offending tensor.
- `model/qwen3.py` — the dense model: 28 layers, 16 query heads over 8 key/value heads, per-head
  QK-norm before RoPE, SwiGLU MLP, RoPE at base 1e6. Recomputes the whole prefix every step, which
  makes it the slowest and most trustworthy thing in the repository.
- `model/qwen3_cached.py` — the same model over a KV cache, with the position `offset` machinery
  that lets a forward pass start partway into a sequence. That is what chunked prefill later
  needs.
- `model/qwen3_paged.py` — the model the engine runs: one ragged token axis, paged attention, and
  no per-sequence loop anywhere in the forward pass.
- `kv_cache.py`, `paged_attention.py` — the dense per-sequence cache, and a gather-then-dense
  paged attention written to be obviously correct. The latter is the oracle the paged kernels are
  diffed against; doing its gather for real would copy every sequence's cache once per iteration.
- `kernels/extension.py`, `kernels/ops.py` — the JIT extension loader (it assembles a CUDA 13
  toolchain out of the pinned nvcc wheels, since the system nvcc is too old for torch) and the
  dispatch table. Every op takes `use_cuda`, so kernels on and kernels off are the same code path
  with a different callee, and `dispatch_report` prints which ops actually ran as kernels.
- `sampler.py`, `generate.py` — `SamplingParams` with greedy and top-p/temperature sampling,
  batched per request in one call, and the reference generation loop the engine is checked
  against.
- `bench.py` — four measurement modes: single-request latency, per-kernel bandwidth, throughput
  against `transformers.generate`, and a scheduler stress test. It samples GPU clocks during a run
  and voids its own output if the card was asleep.

### CUDA kernels (`csrc/`)

- `kernel_utils.cuh` — launch geometry, warp and block reductions, and the vectorized load
  helpers. Every kernel accumulates in fp32 regardless of storage dtype.
- `rmsnorm.cu`, `rope.cu`, `swiglu.cu` — the memory-bound elementwise trio, each a single pass
  over its data at 73–81% of theoretical bandwidth.
- `decode_attention.cu` — one query against a whole context by online softmax: running max and sum
  updated per key block, so the score vector is never materialized. Split over the key axis when
  the context is long enough that one block per head leaves the GPU idle.
- `flash_prefill.cu` — tiled causal attention for query length > 1, K/V tiles staged through
  shared memory, entire tiles skipped when the causal mask excludes them.
- `paged_attention.cu` — the same two shapes, but gathering K and V through a block table inside
  the kernel instead of from a contiguous tensor. The decode path splits over the key axis; the
  prefill path scores eight keys per step so the cross-lane reductions have something to overlap
  with.
- `bindings.cpp`, `hello.cu` — the pybind11 surface, and an `axpby` smoke test whose only job is
  to prove the toolchain works before anything depends on it.

### Paged KV cache (`mini_vllm/block/`)

- `block_pool.py` — allocation and reference counting over physical block ids. Reference counts
  are what make sharing possible; the free count is what drives admission control and preemption.
- `block_table.py` — one sequence's logical-to-physical mapping, and the whole paging indirection:
  token position → block id and offset within it. Fragmentation is bounded by one partial block
  per sequence by construction.
- `kv_pool.py` — the physical storage the block ids index into: a single pre-allocated tensor with
  the layer axis inside it, so the whole cache's size is knowable up front and can be chosen from
  the memory left after the weights. Written by slot mapping rather than by sequence.
- `block_manager.py` — capacity, growth, sharing, and release, plus copy-on-write: a forked
  sequence shares its parent's blocks until one of them writes, and only the block being written
  is copied.

### Serving (`mini_vllm/serve/`)

- `sequence.py` — per-request state: prompt and output tokens, how many are computed, and the
  status transitions. The distinction between "computed" and "present" is what makes chunked
  prefill and preemption expressible.
- `batch.py` — the ragged `ForwardBatch`: sequences of different lengths flattened onto one token
  axis with offsets, carrying the slot mapping and block tables the kernels need. Built entirely
  from host integers, because reading a device tensor here costs more than the model does.
- `scheduler.py` — continuous batching: every iteration re-forms the batch under a token budget,
  piggybacking decodes onto prefill chunks, preempting the newest sequence when the pool cannot
  back the next step and recomputing it later. Also holds a dense per-sequence runner used to
  demonstrate that scheduling without paging buys fairness and no throughput at all.
- `runner.py`, `engine.py` — the paged model runner, and the `LLM` API: admission, `step()`,
  `generate`, and `generate_stream`. The batch call is the streaming call, drained, so the two
  cannot disagree.

### Validation (`tests/`)

858 tests, differential at the core. Each kernel is pinned to the PyTorch expression it replaces
on value (fp32 exactly; bf16 and fp16 to dtype-appropriate tolerances defined once in
`conftest.py`) and on the property that matters more than the value: the kernel changed the speed,
not the text. Above the kernels, the cached model is diffed against the recomputing model, the
paged model against the cached one, paged attention against a gather-then-dense oracle, and the
engine against `transformers.generate` on sixteen varied-length prompts. In fp32 that identity is
exact; in bf16 it usually is, and where it is not, a separate test proves the divergent token was
a genuine numerical tie — the second choice in fp32, one rounding from the first — rather than a
bug.

Behavioral gates cover what numerics cannot: the block pool must leak nothing over three thousand
requests, a pool small enough to force preemption must still finish every request, scheduler
policies must not change the tokens produced, and head-of-line stalls are asserted in *tokens the
engine computed between two of a sequence's tokens* rather than in milliseconds, so the claim is
deterministic instead of dependent on a quiet machine.

## Performance

Measured on an RTX 5070 Laptop (8 GB, Blackwell `sm_120`, 384 GB/s theoretical) with Qwen3-0.6B in
bf16 and CUDA 13. Output tokens per second over a whole request set, EOS ignored so both engines
do identical work, prompts cycling through 32–512 tokens (`make bench-throughput`):

| Concurrency | Mini-vLLM | `transformers.generate` | Speedup |
|---|---|---|---|
| 1 | 98 tok/s | 38 tok/s | 2.5x |
| 4 | 349 tok/s | 130 tok/s | 2.7x |
| 16 | 1062 tok/s | 292 tok/s | 3.6x |
| 32 | **1688 tok/s** | 152 tok/s | **11.1x** |

The asymmetry is the result rather than a handicap invented for it: `generate` takes one padded
rectangle, so prompts of 32 to 512 tokens all run for 512 and its rate *falls* past batch 16 as
padding grows, while the engine gives each sequence its own length and admits a replacement in the
iteration a request finishes. Single-request latency, where there is nothing to batch and only the
kernels and the cache are working, is 19.3 ms to first token and 107 tok/s decoding, against
25.6 ms and 40 tok/s.

RMSNorm and SwiGLU reach 81% of theoretical bandwidth and RoPE 73%, which is the practical ceiling
for anything that touches memory once — a bare `copy_` gets 79%. Paged decode attention is 38x the
PyTorch it replaces at batch 16, almost all of it from not copying every sequence's cache per
iteration. Paged prefill is the one loss on the board, 0.54x a gather-and-cuBLAS oracle at
L=512/S=2048, because the oracle's matmuls run on tensor cores and this kernel's inner loop runs
on the FP32 pipes; an `mma`-based inner loop is the largest single performance item left and
nothing else depends on it.

Under an adversarial mix — many 32-token requests with a 2048-token prompt dropped in every
twentieth, Poisson arrivals near capacity, 2000 requests replayed through both policies
(`make bench-scheduler`) — chunked prefill cuts the worst decode gap 2.4x (259 ms against 613 ms) at
identical throughput and with no leaked blocks. The honest shape of that result is that total
waiting is unchanged: chunking *redistributes* the stall, many decodes waiting one chunk each
instead of a few waiting an entire prompt, and what it removes is the unbounded case. Whether the
improvement also shows up at P99 depends on how many decodes are in flight when a long prompt
lands, so the benchmark reports both tails rather than asserting a direction.

## Build and run

Requires a CUDA GPU with a couple of GB free — the KV pool sizes itself to what is left after the
weights, so a smaller card serves fewer concurrent requests rather than failing — and Python 3.13.
The pinned `torch` is a CUDA 13 build; `requirements.txt` explains why the four nvcc wheels beside
it must be pinned to the same minor.

```bash
make setup             # create .venv and install pinned deps (CUDA 13 torch)
make test              # 858 tests; GPU and checkpoint tests skip when unavailable
make test-cpu          # the subset needing neither a GPU nor weights
make bench             # single-request TTFT and decode tok/s, against transformers
make bench-throughput  # output tok/s vs transformers, by concurrency
make bench-scheduler   # decode-latency tails: chunked prefill vs prefill-first
make bench-kernels     # achieved bandwidth per kernel vs the torch it replaced
make ext               # force-rebuild the csrc/ extension and print the toolchain
make clean             # remove .venv, build/, and caches
```

The CUDA extension builds itself on first use through PyTorch's JIT loader, so there is no
separate compile step; `make ext` is for when a stale cache is the suspect. Every kernel can be
turned off at runtime (`LLM(..., use_cuda_kernels=False)`), which is how the tests prove the
kernels changed the speed and not the answer.
