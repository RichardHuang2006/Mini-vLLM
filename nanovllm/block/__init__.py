"""Paged KV cache bookkeeping: block pool, block tables, prefix cache.

Everything in this package is pure Python over integers -- no tensors, no GPU.
That is deliberate. This is where the subtle bugs live (refcount leaks,
aliasing across a fork, evicting a block someone still holds), and keeping it
free of GPU state makes those bugs debuggable in milliseconds.
"""
