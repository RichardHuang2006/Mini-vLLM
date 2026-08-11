<div align="center">

# Mini-vLLM

**A paged-attention LLM inference engine, built bottom-up**

`Python` · `PyTorch` · `CUDA C++`

A single-GPU serving engine for Qwen3-0.6B, written from the model outwards:<br/>
a readable reference implementation, then hand-written CUDA kernels, then a paged KV cache<br/>
and a continuous-batching scheduler — with the slow version kept at every step as the oracle.

**11x the throughput of `transformers.generate` at 32 concurrent requests**,<br/>
with greedy output verified token-for-token against it.

[Design](./DESIGN.md) · [Build it yourself](./PLAN.md)

</div>

---

```python
from mini_vllm import LLM

llm = LLM("Qwen/Qwen3-0.6B")

for completion in llm.generate(["The capital of France is"], max_tokens=32):
    print(completion.text)
```

Streaming, interleaved across requests as the engine produces them:

```python
for update in llm.generate_stream(prompts, max_tokens=128):
    print(update.index, update.text, end="", flush=True)
```

---

## Contents

| Section | What is in it |
|---|---|
| [Results](#results) | Throughput, latency, kernel bandwidth, scheduler tails |
| [Quickstart](#quickstart) | Clone to reproduced benchmark |
| [How it works](#how-it-works) | The five pieces and one iteration through them |
| [What I learned](#what-i-learned) | The findings that changed the code |
| [Known limitations](#known-limitations) | What is slow, what is missing, and why |
| [Repository layout](#repository-layout) | Where everything lives |

---

## Results

Hardware of record: **RTX 5070 Laptop, 8 GB, Blackwell `sm_120`**, Qwen3-0.6B in bf16, CUDA 13.
Every number below is reproducible with one `make` target, and the harness refuses to be quiet about a
throttled GPU: it samples the clocks during the run and prints a warning that invalidates its own output
if the card was asleep.

### Throughput — the headline

Output tokens per second over a whole request set, submitted at once. Prompts cycle through 32–512 tokens,
128 output tokens each, EOS ignored so both engines do identical work. `make bench-throughput`:

| Concurrency | Mini-vLLM | `transformers.generate` | Speedup |
|---|---|---|---|
| 1 | 98 tok/s | 38 tok/s | 2.5x |
| 4 | 349 tok/s | 130 tok/s | 2.7x |
| 16 | 1062 tok/s | 292 tok/s | 3.6x |
| 32 | **1688 tok/s** | 152 tok/s | **11.1x** |

**17.2x from batch 1 to 32.** Decode is bound by reading the weights, and every sequence in an iteration
reads them once *between them*, so concurrency is nearly free until the arithmetic runs out.

The comparison is not symmetric, and the asymmetry is the result rather than a handicap invented for it:
`generate` takes one padded rectangle, so sixteen prompts of 32 to 512 tokens all run for 512 and its rate
*falls* past batch 16 as padding grows. The engine gives each sequence its own length and admits a
replacement in the iteration a request finishes. That is paging plus continuous batching, and there is no
other trick in the table.

### Single-request latency

The case with nothing to batch, where the engine's only advantages are its kernels and its cache.
`make bench`:

| | TTFT (128-token prompt) | Decode |
|---|---|---|
| Mini-vLLM | 19.3 ms | 107 tok/s (9.3 ms/token) |
| `transformers` | 25.6 ms | 40 tok/s (25.1 ms/token) |
| | 1.3x | **2.7x** |

### Kernels

Achieved bandwidth against the PyTorch expression each kernel replaced. For a memory-bound op the honest
score is its share of peak bandwidth, so the table reports GB/s and sizes one case past this card's 32 MB
L2 — a benchmark whose working set fits in L2 reports numbers *above* the DRAM peak. `make bench-kernels`:

| Kernel | Shape | Ours | vs PyTorch | Share of 384 GB/s peak |
|---|---|---|---|---|
| RMSNorm | past L2 | 312 GB/s | 8.8x | 81% |
| SwiGLU | past L2 | 310 GB/s | 2.6x | 81% |
| RoPE | past L2 | 280 GB/s | 12.3x | 73% |
| Decode attention | S=8192 | 161 GB/s | 4.3x | 42% |
| Paged decode attention | B=16, S=1024 | 223 GB/s | 38x | 58% |
| Paged prefill attention | L=512, S=2048 | 7.6 ms/layer | **0.54x** | compute-bound |

A bare `copy_` of a large buffer reaches 303 GB/s on this card, so the elementwise kernels at 81% of
theoretical peak are at the practical ceiling for anything that touches memory once. The 38x on paged decode
is not a better loop over the same data — it is the removal of a full copy of every sequence's cache per
iteration, which is what the PyTorch oracle has to do to use dense attention at all. Attention is also where
GB/s stops being the right score, which is why the prefill row is in milliseconds.

The last row is a loss and it is in the table on purpose: see [known limitations](#known-limitations).

### Scheduler under load

An adversarial mix — many 32-token requests with a 2048-token prompt dropped in every twentieth, arriving on
a Poisson schedule near this card's capacity — replayed twice through one engine, once with chunked prefill
and once with the prefill-prioritized policy vLLM shipped before it. 2000 requests each way,
`make bench-scheduler`:

| | Chunked prefill | Prefill-first | |
|---|---|---|---|
| Worst decode gap | 259 ms | 613 ms | **2.4x** |
| P99 decode gap | 249 ms | 564 ms | **2.3x** |
| Total decode stall | 906 s | 841 s | 0.93x |
| Output rate | 500 tok/s | 500 tok/s | 1.00x |

4000 requests served, **zero leaked blocks, no OOM**, and every one of them completed.

The last two rows are the finding, and they are not what I expected to write. Chunked prefill does not
*remove* the waiting — total stall and throughput are unchanged. It **redistributes** it: many decodes
waiting one chunk each instead of a few waiting an entire prompt. What it removes is the unbounded case. An
iteration cannot exceed the token budget with chunking on, so the longest a decode can wait is set by
configuration rather than by whichever prompt happens to arrive.

Which means whether the win shows up in *P99* depends on load, and the benchmark reports both tails instead
of asserting a direction. Run it at `--rate 4 --num-requests 300`, where the engine keeps up with arrivals
and few decodes are ever in flight, and a long prompt catches only a handful of them: P99 becomes a tie
(25 ms against 28), because 99% of decode gaps never coincide with a prompt at all, while the worst gap
stays 2.4x apart. The bound is the claim that survives the arrival rate; P99 is a claim about the queue.

---

## Quickstart

```bash
git clone <this repo> && cd Mini-vLLM
make setup                  # .venv + pinned deps (CUDA 13 torch)
source .venv/bin/activate

make bench-throughput       # downloads Qwen3-0.6B, prints the headline table
```

The CUDA extension builds itself on first use through PyTorch's JIT loader, so there is no separate compile
step; `make ext` forces a rebuild when a stale cache is the suspect. `make test` runs the suite — GPU tests
skip without a GPU, and tests that need the real checkpoint skip until it is downloaded:

```bash
make test                   # 858 tests, ~70 s on the hardware of record
make test-cpu               # the subset needing neither a GPU nor weights
```

Requirements: a CUDA GPU with a couple of GB free (the KV pool sizes itself to what is left after the
weights, so a smaller card serves fewer concurrent requests rather than failing), and Python 3.13. The
pinned `torch` is a CUDA 13 build; `requirements.txt` explains why the four nvcc wheels beside it are pinned
to the same minor, which is the one setup detail that will otherwise cost you an afternoon.

---

## How it works

Five pieces, and one `step()` that advances every request in flight by exactly one iteration. There is no
per-request loop anywhere in the engine, because a request is not a unit of execution — an iteration is.

```
   prompts ──────►  ┌──────────────────────────────────────────────┐
                    │  Scheduler        which sequences run, and   │
                    │                   how many tokens each       │
                    └───────────────────────┬──────────────────────┘
                                            │  (sequence, count) pairs
                    ┌───────────────────────▼──────────────────────┐
                    │  BlockManager     pages to back that, or a   │
                    │                   refusal and a preemption   │
                    └───────────────────────┬──────────────────────┘
                                            │  slot mapping + block tables
                    ┌───────────────────────▼──────────────────────┐
                    │  ForwardBatch     one ragged batch: a 512-   │
                    │                   token chunk beside eleven  │
                    │                   single-token decodes       │
                    └───────────────────────┬──────────────────────┘
                                            │  one forward pass
                    ┌───────────────────────▼──────────────────────┐
                    │  Qwen3Paged       28 layers, hand-written    │
                    │                   kernels, paged attention   │
                    └───────────────────────┬──────────────────────┘
                                            │  one logits row per sequence
                    ┌───────────────────────▼──────────────────────┐
                    │  Sampler          greedy or top-p, per       │
                    │                   request, one batched call  │
                    └───────────────────────┬──────────────────────┘
                                            ▼
                                          tokens
```

* **The model** ([`model/`](./mini_vllm/model)) is Qwen3 reconstructed from the paper and the checkpoint:
  grouped-query attention with 16 query heads over 8 KV heads, QK-norm, SwiGLU, RoPE at base 1e6. Two
  versions exist and both are kept — a readable dense one that is the oracle, and a paged one that the
  engine runs. [DESIGN §3](./DESIGN.md#3-the-qwen3-model)
* **The kernels** ([`csrc/`](./csrc)) are hand-written CUDA C++: RMSNorm, RoPE, SwiGLU, and three attention
  kernels including the online-softmax decode path and paged attention that walks a block table inside the
  kernel. [DESIGN §5](./DESIGN.md#5-cuda-kernels)
* **The paged cache** ([`block/`](./mini_vllm/block)) stores K and V in fixed 16-token pages that any
  sequence can hold in any order, addressed through a per-sequence block table, with copy-on-write for
  forked sequences. Fragmentation is near-zero by construction.
  [DESIGN §6](./DESIGN.md#6-paged-kv-cache)
* **The scheduler** ([`serve/scheduler.py`](./mini_vllm/serve/scheduler.py)) re-forms the batch every
  iteration: decodes first, then prefill chunks in whatever budget is left, then new admissions. Under
  memory pressure it preempts the newest sequence and recomputes it later.
  [DESIGN §7](./DESIGN.md#7-continuous-batching-scheduler)
* **The engine** ([`serve/engine.py`](./mini_vllm/serve/engine.py)) is the wiring and the public API. The
  batch call is the streaming call, drained, so the two cannot disagree.

The full request lifecycle, with shapes at every hop, is [DESIGN §8](./DESIGN.md#8-data--control-flow).

### Correctness

Every fast path has a slow one behind it, and the slow one is a test oracle rather than a fallback that
happens to exist. The chain closes at both ends: each kernel is diffed against the PyTorch expression it
replaces, the paged model against the dense model, the engine against `transformers.generate`.

The milestone test is token identity: greedy output for sixteen varied-length prompts, matching
`transformers` token for token. In fp32 it matches exactly. In bf16 it usually does, and where it does not,
a separate test proves the disagreement is a genuine numerical tie — the divergent token is the *second*
choice in fp32, one rounding apart from the first — rather than a bug.
[DESIGN §9](./DESIGN.md#9-testing--validation)

### Build it yourself

[`PLAN.md`](./PLAN.md) is the course this repository was built from: 33 steps across six phases, each with
what to write, what to test it against, and a "done when" you can check. It is ordered so that something
works at every step — a model that generates text at Step 1.9, an engine that serves at Step 4.9 — and it
says where the walls are before you hit them, including the one it tells you to walk into.

---

## What I learned

The things that changed the code, rather than the things I expected to write:

**Reading a device tensor is the most expensive line in a Python inference loop.** Two places were doing it
per iteration, both to re-learn numbers that had been Python integers moments earlier: batch construction
converted its slot mapping element by element, and the KV write bounds-checked its slots with `.max()` and
`.min()` *per layer* — 56 pipeline drains per iteration, a third of the model's time. Moving both checks
onto host integers took a batch of 16 from 32.4 ms per iteration to 22.9, and batch construction from 8.8 ms
to 0.6. Timing the loop stage by stage is what found it; the code reads innocently, because a bounds check
is exactly the sort of thing you are supposed to write.

**Chunked prefill redistributes latency, it does not remove it.** I set out to reproduce "chunking lowers
P99 decode latency" and found that the total time callers spend waiting is the same under both policies to
within 8%, and that whether the improvement appears at P99 depends entirely on the arrival rate. The real
guarantee is narrower and better: an iteration cannot exceed the token budget, so the worst case is bounded
by configuration. Reporting the number I expected would have been reporting a coincidence.

**Latency claims should be tested in tokens, not milliseconds.** A test asserting on wall-clock fails on a
busy laptop, so the scheduler tests count *tokens the engine computed between two of a sequence's tokens* —
deterministic, and what the clock is a noisy view of. It also separates two effects a milliseconds test
blurs: the pass order is what protects a decode (16x less waiting), and chunking is what keeps an iteration
inside its budget.

**Tensor cores are not an optimization, they are the arithmetic.** My hand-written prefill kernel does
everything the textbook says — shared-memory tiling, online softmax, coalesced gathers, register blocking —
and still loses to a PyTorch oracle that materializes an entire L×S score matrix, because the oracle's two
matmuls run on tensor cores and my inner loop runs on the FP32 pipes. Optimizing the loop closed some of
the gap (13.3 ms to 7.6 ms per layer, by scoring eight keys before touching the running softmax so the
cross-lane reductions have something to overlap with) and could not close the rest.

**A cache that cannot be batched makes continuous batching worthless.** Phase 4 deliberately builds the
scheduler *before* paging, on a dense per-sequence cache, and measures the result: `n` scheduled sequences
become `n` forward passes, fairness improves, and GPU efficiency does not move at all. Paging is what turns
a scheduling decision into one forward pass. The wall was worth walking into on purpose.

**In bf16, greedy decoding has ties.** Two correct implementations diverge a few tokens into a generation
because the top two logits are one rounding apart, and chasing that as a bug cost real time. The fix was not
to loosen a tolerance: it was to make the ambiguity visible — run the identity test in fp32, where there is
a right answer, and add a separate test asserting that each bf16 disagreement is a near-tie.

---

## Known limitations

Measured, not hypothetical:

* **Prefill attention is scalar.** The paged prefill kernel is 0.54x the gather-and-cuBLAS oracle at
  L=512/S=2048 (7.6 ms against 4.1 per layer), because the oracle uses tensor cores. This is the largest
  single performance item in the repository, and it is
  [Step 3.6](./PLAN.md#step-36--tensor-core-prefill-inner-loop-optional), deferred on purpose: an
  `mma`-based inner loop is a project of its own and nothing else depends on it. Dense prefill routes to
  cuBLAS for the same reason, which the dispatch report prints on every run rather than hiding.
* **~15 ms of Python per iteration.** Several hundred kernel launches for a model whose weights take 5 ms
  to read, so at low concurrency the engine is dispatch-bound rather than GPU-bound — visible as a decode
  iteration that costs the same at batch 1 and batch 8. CUDA graph capture is the fix and is out of scope
  by design ([DESIGN §11](./DESIGN.md#11-future-work)).
* **Preemption is recompute-only.** An evicted sequence re-prefills its prompt and its output so far; there
  is no swap-to-host path. Simpler, and it costs compute already spent under memory pressure.
* **One GPU, one model family, Python API only.** No tensor parallelism, no HTTP frontend, no radix-tree
  prefix caching, no FP8 KV, no speculative decoding. Each is a
  [deliberate non-goal](./DESIGN.md#1-goals--non-goals) rather than an oversight, and each has a ready-made
  correctness test the moment someone wants it.

---

## Repository layout

```
mini_vllm/
  model/          Qwen3: readable, cached, and paged
  kernels/        extension loading and the op dispatch table
  block/          block pool, block table, block manager
  serve/          sequence, batch, scheduler, runner, engine
  bench.py        four benchmark modes: single, kernels, throughput, scheduler
csrc/             CUDA kernels (2.3k lines): rmsnorm, rope, swiglu, attention x3
tests/            858 tests (10k lines), differential against PyTorch oracles
DESIGN.md         the architecture, with the shape contract and the tolerances
PLAN.md           the 33-step course this was built from
```

Roughly 7k lines of Python, 2.3k of CUDA, and 10k of tests — the ratio is the point.
