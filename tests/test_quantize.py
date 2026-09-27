"""quantize.py: the e4m3 error bound, saturation, and the scale."""

import mlx.core as mx
import pytest

from mini_vllm.quantize import FP8_MAX, dequantize_fp8, quantize_fp8


@pytest.mark.parametrize("scale", [1.0, 0.05])
def test_round_trip_error_is_within_the_e4m3_bound(scale):
    x = (scale * 50 * mx.random.normal((4096,))).astype(mx.float32)
    back = dequantize_fp8(quantize_fp8(x, scale), scale, mx.float32)

    # 3 mantissa bits round to within 2^-4 relative; subnormals to half their 2^-9 step.
    bound = 2**-4 * mx.abs(x) + scale * 2**-10
    assert mx.all(mx.abs(back - x) <= bound).item()


def test_out_of_range_values_saturate():
    x = mx.array([1e6, -1e6, 500.0, FP8_MAX])
    back = dequantize_fp8(quantize_fp8(x), 1.0, mx.float32)
    assert back.tolist() == [FP8_MAX, -FP8_MAX, FP8_MAX, FP8_MAX]


def test_the_scale_divides_before_the_cast():
    x = mx.random.normal((256,)).astype(mx.bfloat16)
    bits = quantize_fp8(x, 0.25)

    assert bits.dtype == mx.uint8 and bits.nbytes == x.size
    assert mx.array_equal(bits, quantize_fp8(x.astype(mx.float32) / 0.25)).item()
