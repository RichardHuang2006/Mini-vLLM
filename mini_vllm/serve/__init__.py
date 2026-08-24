"""The serving layer: sequences, batches, the scheduler, the engine.

Everything above the model. The model runs one forward pass over whatever it is
handed; this package decides what to hand it, which requests are in flight, and how
their memory is accounted for.

Governing invariant: a scheduling decision may change timing, never output. The tests
here compare batched, interleaved, chunked runs against the same requests run one at a
time and assert the token sequences are identical.
"""

from __future__ import annotations
