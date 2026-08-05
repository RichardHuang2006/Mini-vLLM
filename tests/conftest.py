"""Shared fixtures and collection hooks for the whole test suite.

Three jobs:

* **Reproducibility.** Every test starts from the same RNG state, so a failure
  can always be reproduced. This is not optional for a project whose tests
  compare float tensors -- a test that fails one run in twenty is worse than no
  test at all.
* **Graceful degradation.** ``pytest`` with no arguments should do something
  sensible on a machine with no GPU and no downloaded weights. The marker hooks
  below skip what cannot run, so you get one clear failure from
  ``test_cuda_is_available`` rather than a wall of errors.
* **Hypothesis setup**, used from Phase 1 onward for property tests.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, settings

SEED = 0

# Property tests get no per-example deadline: the first example of a GPU test
# pays for kernel compilation and would otherwise trip the default timeout. The
# health-check suppression is for `seeded` below -- Hypothesis warns about
# function-scoped fixtures because they are not re-run between examples, which
# is exactly the behaviour we want from a fixture that only seeds RNGs.
settings.register_profile(
    "nanovllm",
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.load_profile("nanovllm")


@pytest.fixture(autouse=True)
def seeded():
    """Reset every RNG before each test. Applied automatically, everywhere."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)  # seeds CPU and all CUDA devices


@pytest.fixture(scope="session")
def device() -> torch.device:
    """The CUDA device under test. Only request this from a `gpu`-marked test."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device("cuda:0")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip tests whose prerequisites are absent on this machine."""
    no_gpu = pytest.mark.skip(reason="no CUDA device available")
    no_oracle = pytest.mark.skip(
        reason="needs HuggingFace weights; set NANOVLLM_RUN_ORACLE=1 to allow the download"
    )

    gpu_missing = not torch.cuda.is_available()
    oracle_disabled = os.environ.get("NANOVLLM_RUN_ORACLE") != "1"

    for item in items:
        if gpu_missing and "gpu" in item.keywords:
            item.add_marker(no_gpu)
        if oracle_disabled and "oracle" in item.keywords:
            item.add_marker(no_oracle)
