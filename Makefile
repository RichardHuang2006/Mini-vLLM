# Mini-vLLM — a paged-attention LLM inference engine, built bottom-up
#
# Targets: setup (venv + deps) · ext (build CUDA extension) · test · bench · clean · help
#
# `make test` uses whatever `python` is on PATH, so it works against an
# already-installed torch without `make setup` first. `make setup` creates a
# local venv for a clean-room install.

PYTHON ?= python3
VENV    = .venv
VENVBIN = $(VENV)/bin
TORCH_INDEX = https://download.pytorch.org/whl/cu130

.PHONY: setup ext test test-cpu bench bench-kernels clean help
.DEFAULT_GOAL := help

# ------------------------------------------------------------------- setup ---
# Create a local venv and install pinned dependencies for a fresh clone. The
# extra index is where the CUDA 13 torch build lives.
setup:
	$(PYTHON) -m venv $(VENV)
	$(VENVBIN)/pip install --upgrade pip
	$(VENVBIN)/pip install -r requirements.txt --extra-index-url $(TORCH_INDEX)
	@echo "setup: activate with 'source $(VENVBIN)/activate'"

# --------------------------------------------------------------------- ext ---
# Force a full rebuild of the csrc/ extension and print the resolved toolchain.
# A stale JIT cache is the first thing to suspect when a kernel edit appears to
# do nothing, so this target exists to rule it out in one command.
ext:
	$(PYTHON) -m mini_vllm.kernels.extension --rebuild

# -------------------------------------------------------------------- test ---
# Run the suite. Pure-CPU tests always run; `cuda` tests skip without a GPU and
# `oracle` tests skip until the Qwen3-0.6B weights are downloaded.
test:
	$(PYTHON) -m pytest -q

# Everything that needs neither a GPU nor downloaded weights.
test-cpu:
	$(PYTHON) -m pytest -q -m "not cuda and not oracle and not slow"

# ------------------------------------------------------------------- bench ---
# TTFT and decode tok/s for the cached model. Warns if it caught the GPU running
# below its clocks, which would make every number meaningless.
bench:
	$(PYTHON) -m mini_vllm.bench --mode single

# Achieved memory bandwidth for each hand-written kernel, next to the PyTorch
# expression it replaced. The score for a memory-bound op is its share of peak
# bandwidth, not its wall-clock (Phase 3).
bench-kernels:
	$(PYTHON) -m mini_vllm.bench --mode kernels

# --------------------------------------------------------------- housekeeping
clean:
	rm -rf $(VENV) build .pytest_cache .hypothesis
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

help:
	@echo "Mini-vLLM targets:"
	@echo "  setup     create .venv and install pinned requirements (CUDA 13 torch)"
	@echo "  ext       force-rebuild the csrc/ CUDA extension, print the toolchain"
	@echo "  test      run the pytest suite (cuda/oracle tests skip when unavailable)"
	@echo "  test-cpu  run only the tests needing neither a GPU nor model weights"
	@echo "  bench     TTFT and decode tok/s (Step 2.3 onward)"
	@echo "  bench-kernels  achieved bandwidth per kernel vs the torch it replaced"
	@echo "  clean     remove .venv, build/, caches, and __pycache__"
