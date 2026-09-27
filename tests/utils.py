"""Comparison helpers shared by every test."""

import mlx.core as mx

DTYPES = [mx.float32, mx.float16, mx.bfloat16]

# The shared numerical tolerances, keyed by dtype so no test picks its own threshold.
OP_TOLERANCES = {
    mx.float32: 1e-5,
    mx.float16: 1e-2,
    mx.bfloat16: 1e-2,
}


def assert_allclose(actual: mx.array, expected: mx.array) -> None:
    """Compare two arrays of the same shape and dtype at the tolerance that dtype implies."""
    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"dtype {actual.dtype} != {expected.dtype}"

    tolerance = OP_TOLERANCES[actual.dtype]
    actual, expected = actual.astype(mx.float32), expected.astype(mx.float32)
    if not mx.allclose(actual, expected, rtol=tolerance, atol=tolerance).item():
        worst = mx.max(mx.abs(actual - expected)).item()
        raise AssertionError(f"max abs difference {worst:.3g} exceeds tolerance {tolerance:g}")
