# Mini-vLLM

A single-GPU LLM inference engine for Qwen3-0.6B in ~15 Python files and 7 CUDA
kernels, written to be read. It serves continuously batched requests out of a paged KV
cache — fixed 16-token pages addressed through per-sequence block tables that the
attention kernels walk in-kernel — with copy-on-write page sharing, radix-tree prefix
caching, chunked prefills with piggybacked decoding, optional FP8 cache storage, and
speculative decoding verified by rejection sampling. Every fast path has a slow,
obviously correct counterpart that serves as its test oracle.

## 1. What Mini-vLLM teaches

Each file is one systems concept, sized to be read in a sitting:

| File | The concept |
|---|---|
| `mini_vllm/config.py` | every knob, in one place |
| `mini_vllm/ops.py` | the nine operations inside an inference engine, as readable references |
| `mini_vllm/model.py` | Qwen3 three ways: dense oracle, dense-cached, paged |
| `mini_vllm/cache.py` | the memory hierarchy: pool → block table → paged storage → radix tree → block manager |
| `mini_vllm/scheduler.py` | continuous batching, chunked prefill, piggyback decoding, preemption |
| `mini_vllm/kernels.py` | how hand-written CUDA reaches PyTorch: JIT build + transparent dispatch |
| `mini_vllm/engine.py` | admission, the iteration loop, and the `LLM` API |
| `mini_vllm/speculative.py` | draft, verify by rejection sampling, roll back |
| `mini_vllm/benchmark.py` | measuring a GPU without fooling yourself |
| `csrc/*.cu` | one kernel per file: fused elementwise ops and FlashAttention-style attention |

Shape symbols, used throughout the code: `B` batch, `L` query length, `S` source
(context) length, `D` head dim, `H_q` / `H_k` query and KV heads, `E` hidden size, `V`
vocabulary, `T` total tokens of a ragged batch, `P` tokens per page.

No architectural novelty is claimed; this is a correct, measured, single-GPU
implementation of the ideas in vLLM's PagedAttention paper and Orca's continuous
batching, written so each idea can be traced from the paper to the line of code.

## 2. Feature summary

- **Paged KV cache with block-table indirection** (`cache.py`, `csrc/paged_attention.cu`)
  — fixed 16-token pages in one pre-allocated pool; a per-sequence block table maps
  token position → page and offset; the attention kernel resolves that indirection
  inside its inner loop, so a ragged batch is one launch and internal fragmentation is
  bounded to one partial page per sequence.
- **Copy-on-write block sharing** (`cache.BlockManager`) — reference-counted pages let
  a forked sequence share its parent's blocks until one writes; only the written page
  is copied. This is the path `SamplingParams(n=k)` takes: one prefill, `k` branches.
- **Radix-tree prefix caching** (`cache.PrefixCache`) — a radix tree over
  block-aligned token prefixes lets a prompt beginning with one already served reuse
  its pages. Cached blocks sit free at reference count zero, still matchable, and are
  reclaimed only under pressure.
- **Continuous-batching scheduler** (`scheduler.py`) — the batch is re-formed every
  iteration under a token budget: decodes first, prefill chunks in what remains, then
  admissions. Iteration-level preemption evicts the newest sequence when the pool
  cannot back a decode, and recomputes it later.
- **Fused CUDA kernels** (`csrc/`) — RMSNorm, RoPE and SwiGLU as single-pass
  elementwise kernels; decode attention by online softmax with split-key execution for
  long contexts; tiled causal flash prefill; paged attention for both shapes with the
  block-table walk in-kernel.
- **FP8 KV quantization** (`kv_cache_dtype="fp8"`) — e4m3 storage with explicit
  per-tensor scales halves the bytes per page; a fused quantize-and-scatter kernel
  writes it in one pass, and the paged attention kernel dequantizes in registers.
- **Speculative decoding** (`speculative.py`) — a draft model proposes `k` tokens; the
  target verifies all of them in one pass; rejection sampling (accept with
  `min(1, p/q)`, resample rejections from `norm(relu(p−q))`, bonus token on full
  acceptance) keeps the output distribution exactly the target's.

## 3. Request lifecycle

One request, front to back:

1. `LLM.add_request` tokenizes the prompt and enqueues a `Sequence` (waiting).
2. Each `step()`, the `Scheduler` re-forms the batch: running decodes take one token
   each, running prefills take a chunk of the remaining budget, and new requests are
   admitted if pages and budget remain. A prefix-cache hit at admission marks the
   matched prefix as already computed.
3. The `BlockManager` reserves pages for every scheduled token and produces the slot
   mapping (where each new K/V is written) and block tables (where old K/V lives).
4. `ForwardBatch` flattens the scheduled sequences onto one ragged token axis, and
   `Qwen3Paged` runs all 28 layers once for the whole iteration: embed → project
   Q/K/V → QK-norm → RoPE → scatter K/V into the pool → paged attention through the
   block tables → output projection → SwiGLU → final norm → logits at each sequence's
   last position.
5. One batched sample draws a token per sequence; `commit` appends tokens, retires
   finished sequences (freeing their pages, optionally registering them in the prefix
   tree), and the loop repeats.

## 4. Repository structure

```
mini_vllm/           the engine, one file per concept (see §1)
csrc/                bindings.cpp, kernel_utils.cuh, and one .cu per kernel
tests/               conftest + four suites: model, cache/scheduler,
                     CUDA kernels, engine/speculative
Makefile             setup · ext · test · test-cpu · bench-*
requirements.txt     pinned deps, including the CUDA 13 toolchain wheels
pytest.ini           markers (cuda / oracle / slow) and pythonpath
```

## 5. Recommended reading order

1. `config.py` — the vocabulary of knobs everything else uses.
2. `ops.py` — the math, in its slow, correct form.
3. `model.py` — the ops assembled into Qwen3, three ways.
4. `cache.py` — pages, tables, sharing, and the radix tree.
5. `scheduler.py` — who runs, and for how many tokens.
6. `kernels.py` — how the fast paths are built and dispatched.
7. `engine.py` — the loop that ties it together, and the public API.
8. `speculative.py` — the draft/verify/rollback cycle.
9. `benchmark.py` — how each claim is measured.
10. `tests/` — the differential evidence.
11. `csrc/` — the kernels, in the same order as their wrappers.

Each file opens with a one-line statement of what it holds; the reasoning behind the
design lives in this README, and the code carries only short notes on what is not
obvious from it.

## 6. Dense versus paged KV caching

Attention at position `t` re-reads the keys and values of positions `0..t`, which do
not change. A dense cache (`cache.DenseKvCache`) concatenates them into one contiguous
tensor per sequence: correct, but every append reallocates, batches must be padded to
the longest member, and capacity must be reserved for the longest possible sequence.
The paged cache stores the same tensors in fixed 16-token pages that any sequence can
hold in any order, so memory is committed page by page as sequences actually grow.
The dense implementation is kept as the oracle: `tests/test_cache_scheduler.py` proves
the paged engine token-identical to it.

## 7. Block tables and physical pages

```
sequence A (23 tokens)                     physical pool (any order)
logical blocks:  [0] [1]  ...partial       ┌────┬────┬────┬────┬────┬────┐
block table:      7   2   9                │ p0 │ p1 │ p2 │ .. │ p7 │ .. │
                  │   │   │                └────┴──▲─┴──▲─┴────┴──▲─┴────┘
                  │   │   └── tokens 16..22 ───────┘   │          │
                  │   └────── tokens  8..15 ───────────┘          │
                  └────────── tokens  0..7  ──────────────────────┘

position p  →  page = table[p // 16],  slot = page·16 + (p % 16)
```

Five words that must not blur together: a **token position** indexes one sequence; its
**logical block** is `position // block_size`; the **physical block** is whichever pool
page the table maps that logical block to; the **slot offset** is `position %
block_size`; and the **reference count** is how many block tables currently name that
physical page. The same arithmetic reappears inside `csrc/paged_attention.cu`, which
is what "the kernel walks the block table" means.

## 8. Copy-on-write

Forking a sequence (`SamplingParams(n=4)`, one prompt → four branches) increments the
reference count on each of the parent's pages and copies nothing. Full pages are
immutable — no one writes into the middle of a sequence — so they stay shared for both
lifetimes. Only the last, partial page can be written, and the first write by either
branch triggers the copy: allocate a fresh page, copy 16 tokens of K/V across every
layer, repoint one table entry, decref the original. The copy costs one page, not a
prefix of arbitrary length. `test_cache_scheduler.py` pins all of it, down to the
copied bytes.

## 9. Radix-tree prefix caching

Two prompts that begin with the same tokens compute identical K/V for that prefix. A
radix tree whose edges are one full block of token ids maps block-aligned prefixes to
the physical pages that already hold their KV: matching walks the tree as far as the
tokens agree, and matched pages are acquired by reference rather than recomputed. Keys
are exact token tuples, so there are no hash collisions. The tree owns no memory — a
cached page sits on the free list at reference count zero, still matchable, and the
pool evicts its tree node only when it actually reuses the page (FIFO over the free
list, which is LRU over cache entries). Enable with `LLM(enable_prefix_caching=True)`.

## 10. Continuous batching

Static batching runs a fixed batch to completion, so every request waits for the
longest. Here the batch is a per-iteration decision (`Scheduler.schedule`): a finished
sequence's slot is refilled the same iteration, and a 20-token request admitted beside
a 500-token one leaves 480 iterations earlier. The governing invariant, asserted
token-for-token in the tests: a scheduling decision may change *timing*, never
*output*.

```
every iteration, in order, under max_batched_tokens:
  1. decodes        one token for every running sequence past its prompt
  2. prefill chunks up to chunk_size for sequences mid-prompt
  3. admissions     new requests, FCFS, if pages and budget remain
  (any step may preempt the newest running sequence if pages run out)
```

## 11. Chunked prefill and preemption

A 2048-token prompt in one iteration monopolizes the GPU and every streaming caller
waits for it. Chunked prefill bounds each iteration: the prompt runs as
`min(remaining, chunk_size, budget)` tokens per iteration, resuming from
`num_computed_tokens`. Decodes are scheduled *before* the chunk, so they ride along in
the same forward pass rather than queueing behind the prompt — piggyback decoding.
What "stall-free" means precisely: a decode is never displaced by prompt work within
an iteration's budget, so its worst wait is one bounded chunk rather than an unbounded
whole prompt; it is a scheduling-order guarantee, not an absolute latency guarantee.

When the pool cannot back a running decode's next token, the scheduler preempts the
newest running sequence: its pages return to the pool immediately and it re-prefills
later over its prompt *plus* the output already emitted (recompute, not swap). The
tests force this with a deliberately tiny pool and require identical tokens.

## 12. Ragged forward batches

One iteration is one forward pass, however mixed its contents. `ForwardBatch` flattens
a 300-token prefill chunk and a dozen single-token decodes into 312 rows with offsets
(`cu_seqlens_q`), per-sequence lengths (`seq_lens` = L) and context sizes
(`context_lens` = S), plus the slot mapping and block tables. No padding: a padded
rectangle would waste the compute that paging saved in memory. RoPE takes explicit
per-token positions for the same reason — a chunk's tokens may sit at positions
512..1023, and nothing in the tensor shapes records that.

## 13. CUDA kernels

Seven kernels, one file each, reachable only through the dispatch in `kernels.py`:

| Kernel | Replaces | Notes |
|---|---|---|
| `rmsnorm.cu` | `ops.rms_norm` | fused normalize+scale, fp32 accumulation |
| `rope.cu` | `ops.apply_rope` | fused rotation at explicit (ragged) positions |
| `swiglu.cu` | `silu(gate) * up` | one pass instead of two over the wide MLP activation |
| `decode_attention.cu` | grouped attention, L=1 | online softmax, split-key for long contexts |
| `flash_prefill.cu` | grouped causal attention | tiled, K/V staged through shared memory |
| `paged_attention.cu` | the gather oracle | block-table walk in-kernel, decode + prefill, FP8 reads |
| `kv_quantize.cu` | quantize + `index_copy_` | FP8 quantize-and-scatter in one pass |

Every op takes `use_cuda`, so kernels-on and kernels-off are the same code path with a
different callee, and `kernels.dispatch_report(use_cuda)` prints which implementation
each op would use and why. A kernel that measures slower than PyTorch is listed in
`kernels.NOT_YET_FASTER` with the measured reason and is *not* routed to —
`flash_prefill` sits there today, since its scalar inner loop loses to cuBLAS on
tensor cores; an `mma.sync` inner loop is the largest remaining performance item.

## 14. Online softmax

Softmax needs the row maximum before it can exponentiate anything, which seems to
force two passes over the scores. The online-softmax recurrence makes it one pass:
carry a running maximum `m`, running normalizer `l`, and running output `O`; when a
new tile raises the maximum, rescale `l` and `O` by `exp(m_old − m_new)` before
accumulating. The score row is never materialized. `decode_attention.cu` is this
recurrence per key tile; its split-key path repeats the same merge across blocks for
long contexts. The tests target the recurrence directly: a maximum arriving in the
last tile (every accumulator must be rescaled), the mirror case, and logits that would
overflow a non-shifted `exp`.

## 15. FP8 KV-cache quantization

`LLM(kv_cache_dtype="fp8")` stores the cache in e4m3 — 4 exponent bits, 3 mantissa
bits, range ±448 — with explicit per-tensor scales for keys and values: divide by the
scale before the cast down, multiply after the cast up. A page costs exactly half its
bf16 bytes, so the same memory budget holds twice the pages (measured: 2655 vs 1335
pages for the same fraction of free VRAM). Because the scales are scalars, the paged
attention kernel's inner loops are unchanged: the key scale folds into the softmax
scale at the launch site and the value scale rides on the output, so an FP8 element
costs only its conversion on load. The fused `kv_quantize.cu` writes quantized K/V
straight to their slots in one pass, bit-identical to the two-pass PyTorch reference.

## 16. Speculative decoding and rejection sampling

A decode step reads all 1.2 GB of weights to produce one token; scoring `k+1` tokens
costs nearly the same. So a cheap draft proposes `k` tokens, and one target pass
verifies all of them:

```
draft (k passes)            target (1 pass, all_rows)      verdict, per position i
q1..qk, x1..xk       ──▶    p1..p{k+1} at every            accept xi with min(1, pi(xi)/qi(xi))
(distributions + tokens)    proposed position       ──▶    first rejection: draw from
                                                           norm(relu(pi − qi)), discard rest
                                                           all accepted: bonus from p{k+1}
                            rejected tail's cache slots are trimmed the same iteration
```

The emitted tokens are distributed exactly as the target's own — the acceptance step
emits token `t` with probability `min(p(t), q(t))`, and the residual draw restores
precisely the shortfall — so draft quality affects only latency, never output. At
temperature zero the same code path degenerates to keeping the prefix where the two
argmaxes agree, with no randomness, so greedy speculative output is token-identical to
greedy non-speculative output; the tests assert this even with a deliberately bad
draft. The draft keeps its own KV pool (a different model has different K/V) and is
resynchronized after every verdict: pages past the first divergence are trimmed and
recomputed. Use `LLM(num_speculative_tokens=4, draft_model=...)` for a separate
checkpoint, or `num_draft_layers=k` for a self-draft built from the target's first `k`
layers (shares all weights; validates the mechanism on one model's memory budget).

## 17. Correctness strategy

Differential testing at every seam, with the slow implementation as the oracle:

- each CUDA kernel ⟷ the `ops.py` expression it replaces (value parity on boundary
  shapes, then greedy token identity through the whole model);
- `Qwen3` (dense) ⟷ HuggingFace transformers, on a tiny random model and the real
  checkpoint;
- `Qwen3Cached` ⟷ `Qwen3`, including chunk-boundary positions; `Qwen3Paged` ⟷
  `Qwen3Cached`, through shuffled block tables;
- the engine ⟷ `transformers.generate`, sixteen varied prompts, token for token in
  fp32 (`transformers` appears *only* as an oracle — the serving path never calls it);
- scheduling policies ⟷ single-sequence runs: batching, chunking, piggybacking and
  preemption may change timing, never tokens;
- speculation ⟷ no speculation, greedy-exact; distribution preservation by chi-square
  over 120,000 draws against a deliberately adversarial draft;
- behavioral gates numerics cannot express: no leaked pages after every scenario,
  including a 2,000-request prefix-cache soak and a preemption-forcing stress run.

Tolerances are shared in `tests/conftest.py` so no test picks its own. Single operators
are compared elementwise at 1e-5 (fp32) or 1e-2 (bf16/fp16). Accumulated bf16 error is
compared by relative norm instead, because one bf16 ULP at magnitude 512 is an absolute
difference of 4, which says nothing about whether either side is right.
`BF16_DRIFT_LIMIT` is 5% for a full 28-layer pass against HuggingFace: measured on real
text, a correct model sits at 1.7% and a model given Qwen2's `rope_theta` in place of
Qwen3's at 14%, so the limit is about 3x above the first and 3x below the second.
`KERNEL_DRIFT_LIMIT` is 1% for the CUDA path against the PyTorch path — one bf16 ULP
with headroom rather than an error budget, the kernels being the more accurate of the
two by about one rounding. Where an exact check is available — greedy token ids, FP8
bytes — it is used instead.

195 tests. `cuda` tests skip without a GPU, `oracle` tests skip without the
downloaded checkpoint, so the suite degrades cleanly on any machine.

## 18. Benchmarking

`benchmark.py` has seven modes, one per claim: `single` (TTFT and decode tok/s),
`kernels` (achieved bandwidth vs the PyTorch each kernel replaced, with a `copy_`
ceiling), `throughput` (output tok/s by concurrency vs `transformers.generate`),
`scheduler` (decode-latency tails, chunked prefill vs prefill-priority on one Poisson
schedule), `prefix-cache` (TTFT on vs off), `fp8` (pages per budget and greedy
matching prefix vs bf16), and `spec` (acceptance and wall clock by draft depth).
Every mode prints one evidence header — hardware, torch/CUDA versions, model, dtype,
batch, prompt lengths, output tokens, warmup, metric — and ends with a throttle
verdict: GPU clocks are sampled during the run, and results are declared invalid if
the card ran below half its clocks.

### Measured results

Every number below was rerun on this branch after the consolidation, with the exact
command shown. Conditions: RTX 5070 Laptop GPU (8 GB, sm_120, 384 GB/s theoretical),
WSL2, torch 2.11.0+cu130, Qwen3-0.6B bf16, CUDA kernels on; the clock sampler observed
SM ~2.0 GHz / memory ~11.0 GHz during every run (no throttling, laptop power limits in
effect at ~23 W). Numbers published for the pre-refactor implementation — measured on
the same card under different power conditions — are preserved in the `full-engine`
branch's README and are historical: they were not produced by this code.

**Throughput** (`make bench-throughput`): output tokens/s over a whole request set,
EOS ignored so both engines do identical work, prompts cycling 32–512 tokens, 128
output tokens each.

| Concurrency | Mini-vLLM | `transformers.generate` | Speedup |
|---|---|---|---|
| 1 | 58 tok/s | 33 tok/s | 1.8x |
| 4 | 242 tok/s | 105 tok/s | 2.3x |
| 16 | 774 tok/s | 272 tok/s | 2.8x |
| 32 | **1342 tok/s** | 148 tok/s | **9.1x** |

The comparison is asymmetric by construction, and the asymmetry is the finding:
`generate` takes one padded rectangle, so prompts of 32–512 tokens all run for 512 and
its rate collapses past batch 16 as padding grows, while the engine gives each
sequence its own length and refills slots the iteration a request finishes — 23x
scaling from batch 1 to 32 on the engine side.

**Single request** (`make bench`): TTFT 21.3 ms and 88.6 tok/s decode, against
55.0 ms and 30.3 tok/s for `transformers.generate` (2.6x TTFT, 2.9x decode).

**Kernels** (`make bench-kernels`), DRAM-bound regime against a 303.7 GB/s `copy_`
ceiling (79% of theoretical peak — the practical ceiling for an op that touches memory
once): SwiGLU 302 GB/s (79% of peak, 2.6x torch), RoPE 265 GB/s (69%, 12.0x), RMSNorm
251 GB/s (65%, 7.7x), fused FP8 quantize-and-scatter 6.2x past L2 and 8.3x at decode
shape. Paged decode attention: 40x the gather oracle at batch 16 and 58x at batch 64 —
the removal of a full cache copy per iteration, not a faster loop over the same data.
The honest losses, kept in the table and off the dispatch path (`NOT_YET_FASTER`):
dense flash prefill 0.39x cuBLAS at L=512 and paged prefill 0.57x the oracle, both
because their scalar inner loops lose to tensor cores.

**Scheduler** (`make bench-scheduler`): 2,000 requests, Poisson 16/s, one 2048-token
prompt every 20 requests, the same arrival schedule replayed through both policies on
one engine. Chunked prefill against prefill-priority: worst decode gap 285 ms vs
645 ms (2.26x) and P99 273 ms vs 599 ms (2.19x), at identical throughput (498 tok/s
both ways) and no leaked blocks over either 2,000-request run. Total decode stall is
unchanged (0.99x): chunking redistributes the waiting — many decodes waiting one
bounded chunk instead of a few waiting an entire prompt — and what it removes is the
unbounded worst case, whose bound becomes the configured budget rather than the
longest prompt to arrive.

**Prefix caching** (`make bench-prefix-cache`): six requests behind one shared
preamble: mean TTFT 19.9 ms with the radix tree on vs 27.1 ms off (1.36x), with 3,088
prompt tokens served from cache. Whole-batch time barely moves, as expected: a hit
removes prefill work only, and this run's cost is decode.

**FP8 KV cache** (`make bench-fp8`): 2,655 pages vs 1,335 for the same memory budget
(1.99x measured; 2.0x is the arithmetic), 32,768 vs 65,536 bytes per 16-token page per
layer. Greedy matching prefixes of 0–17 tokens (of 64) before a bf16 near-tie flips:
the per-element tolerance is pinned by the kernel tests, and greedy divergence
compounds after the first flip, so the matching prefix is the honest end-to-end
measure.

**Speculative decoding** (`make bench-spec`, one greedy request, 128 output tokens):

| Draft | Acceptance | Tokens per target pass | Seconds |
|---|---|---|---|
| none | — | 1.00 | 1.95 |
| 4 of 28 layers | 0.004 | 1.02 | 4.78 |
| 28 of 28 (self-draft) | 0.991 | 4.81 | 2.47 |

The mechanism behaves as designed — at full depth 99.1% of proposals are accepted and
each expensive target pass yields 4.81 tokens instead of 1 — and self-drafting still
buys nothing on the wall clock, because a draft deep enough to be accepted costs what
the target costs, while a 4-layer draft is cheap and rejected essentially always.
What speculation needs is a genuinely smaller separate checkpoint
(`LLM(draft_model=...)`, supported but not measurable on 8 GB with this family's
smallest model as the target). The full-depth row is retained because acceptance ≈1.0
with token-identical output is the strongest available correctness statement for the
verification path.

## 19. Installation and commands

Requires Python 3.13. CPU-only machines can run the reference implementations and
most of the test suite; the engine itself wants a CUDA GPU with a couple of GB free.

```bash
make setup     # create .venv and install pinned deps (CUDA 13 torch)
```

The pinned `torch` is a CUDA 13 build from `https://download.pytorch.org/whl/cu130`.
Its CUDA major version must match the nvcc that compiles `csrc/`; on machines whose
system nvcc is older, the four `nvidia-cuda-*` wheels in `requirements.txt` supply a
CUDA 13 toolchain and `kernels.py` assembles them into a usable `CUDA_HOME`.
Pin all four `nvidia-cuda-*` wheels to the same CUDA minor and keep them there: they
are separate wheels but one compiler, and pip will mix minors because the nvcc wheel's
own bounds are loose. Two failures come from that — an nvvm or crt newer than nvcc makes
ptxas report `Unsupported .version`, and a cccl that does not match nvcc gives
`CUDA compiler and CUDA toolkit headers are incompatible`. `nvidia-cuda-cccl` is not
optional: it provides the `<nv/target>` header that `cuda_fp16.h` includes and the nvcc
wheel omits.

```bash
# tests
make test-cpu                      # no GPU, no weights: ~110 tests in seconds
make test                          # full suite [GPU + downloaded weights: ~2 min]

# CUDA extension (JIT-compiled on first use; this forces a rebuild)  [GPU]
make ext                           # prints the toolchain, verifies rmsnorm vs ops.py

# model weights (auto-downloaded on first use; explicit form:)
python3 -c "from mini_vllm.model import resolve_model_path; print(resolve_model_path())"

# generation  [GPU + weights]
python3 -m mini_vllm.engine "The capital of France is" --max-tokens 32
python3 -m mini_vllm.engine "Explain KV caching" --stream --temperature 0.8
python3 -m mini_vllm.engine "The capital of France is" --prefix-caching
python3 -m mini_vllm.engine "The capital of France is" --kv-cache-dtype fp8
python3 -m mini_vllm.engine "The capital of France is" --speculative-tokens 4

# the Python API
python3 - <<'EOF'
from mini_vllm import LLM, SamplingParams
llm = LLM("Qwen/Qwen3-0.6B")                      # batched generation
for completion in llm.generate(["The capital of France is", "2 + 2 ="], max_tokens=32):
    print(completion.text)
for update in llm.generate_stream("Once upon a time", max_tokens=32):  # streaming
    print(update.text, end="", flush=True)
llm.generate("A story:", SamplingParams(temperature=0.8, n=4))  # 4 branches, one prefill
EOF

# benchmarks  [GPU + weights; each prints its evidence header and throttle verdict]
make bench               # single-request TTFT + decode tok/s vs transformers [~3 min]
make bench-throughput    # output tok/s by concurrency vs transformers [~10 min]
make bench-scheduler     # decode-latency tails under both policies [~10 min, 2000 requests x2]
make bench-kernels       # per-kernel bandwidth vs the torch it replaced [~3 min]
make bench-prefix-cache  # TTFT with the radix tree on and off [~2 min]
make bench-fp8           # FP8 vs BF16: pages per budget, greedy matching prefix [~1 min]
make bench-spec          # speculation: acceptance and wall clock by draft depth [~10 min]
```

Optional features are one argument each:

```python
llm = LLM(
    "Qwen/Qwen3-0.6B",
    enable_prefix_caching=True,   # reuse pages behind a shared preamble
    kv_cache_dtype="fp8",         # half the bytes per page, dequantized in-kernel
    num_speculative_tokens=4,     # propose 4, verify by rejection sampling
    use_cuda_kernels=True,        # route ops through csrc/ where a kernel is faster
    seed=0,                       # replayable stochastic sampling
)
```

`use_cuda_kernels=False` runs the identical code path against the PyTorch references,
which is how the tests establish that the kernels changed the speed and not the text.
Configuration can also be passed as one object: `LLM(config=EngineConfig(...))`.

## 20. Historical full implementation

This branch is a consolidation of a larger implementation (71 Python files, 927
tests) into a learning-first layout, preserving every capability and its tests. The
original is kept intact on the `full-engine` branch, including its README with
the originally measured benchmark tables. Nothing was removed in the consolidation:
features were merged file-by-file with their differential tests, and the full suite
above re-validates each one on this branch.

## License

MIT. See [LICENSE](LICENSE).
