"""Phase 4 — the serving layer: sequences, batches, the scheduler, the engine.

Everything above the model. The model computes one forward pass over whatever it
is handed; this package decides *what* to hand it, which requests are in flight,
and how their memory is accounted for.

The invariant that shapes all of it: a scheduling decision may change **timing**,
never **output**. Every test here compares a batched, interleaved, chunked run
against the same requests run one at a time and asserts the tokens are identical.
"""

from __future__ import annotations
