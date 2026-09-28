"""Comparison helpers shared by every test."""

import mlx.core as mx
import pytest

try:
    import mini_vllm_ext  # noqa: F401

    METAL_BUILT = True
except ImportError:
    METAL_BUILT = False

DTYPES = [mx.float32, mx.float16, mx.bfloat16]

# The shared numerical tolerances, keyed by dtype so no test picks its own threshold.
OP_TOLERANCES = {
    mx.float32: 1e-5,
    mx.float16: 1e-2,
    mx.bfloat16: 1e-2,
}


# Runs a test on the pure-MLX path and, once the extension is built, on the kernels too.
_UNBUILT = pytest.mark.skipif(not METAL_BUILT, reason="build src/extensions first")
with_and_without_metal = pytest.mark.parametrize(
    "use_metal", [False, pytest.param(True, id="metal", marks=[pytest.mark.metal, _UNBUILT])]
)


def requires_metal(test):
    """A test that runs a Metal kernel: marked `metal`, and skipped until the extension is built."""
    return pytest.mark.metal(_UNBUILT(test))


def assert_allclose(actual: mx.array, expected: mx.array) -> None:
    """Compare two arrays of the same shape and dtype at the tolerance that dtype implies."""
    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"dtype {actual.dtype} != {expected.dtype}"

    tolerance = OP_TOLERANCES[actual.dtype]
    actual, expected = actual.astype(mx.float32), expected.astype(mx.float32)
    if not mx.allclose(actual, expected, rtol=tolerance, atol=tolerance).item():
        worst = mx.max(mx.abs(actual - expected)).item()
        raise AssertionError(f"max abs difference {worst:.3g} exceeds tolerance {tolerance:g}")
