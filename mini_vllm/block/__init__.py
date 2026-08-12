"""Paged KV cache: pools, block tables, and the manager over them.

Everything in this package is integer bookkeeping — no tensors, no floats, no GPU.
That is deliberate: the hard part of paging is refcounts and index arithmetic, and
keeping it in plain Python means the subtle bugs are found by tests that run in
milliseconds.
"""

from __future__ import annotations
