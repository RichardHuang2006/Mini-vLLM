"""Test environment and fixtures shared by every test."""

import os

# The M5's neural accelerators run fp32 matmuls in TF32 by default, which puts fp32 results
# ~1e-3 off; that is 100x the fp32 tolerance. It must be set before MLX is imported.
os.environ["MLX_ENABLE_TF32"] = "0"

import mlx.core as mx
import pytest

SEED = 1234


@pytest.fixture(autouse=True)
def seeded():
    """Seed MLX's global RNG before each test so a failure is reproducible."""
    mx.random.seed(SEED)
