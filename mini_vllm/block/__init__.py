"""Paged KV cache: pools, block tables, and the manager over them.

This package is integer bookkeeping only: no tensors, no floats, no GPU. The hard part
of paging is reference counting and index arithmetic, so keeping it in plain Python
means the subtle failures are caught by tests that run in milliseconds.
"""

from __future__ import annotations
