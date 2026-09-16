# Mini-vLLM — a paged-attention LLM inference engine. `make help` lists the targets.

PYTHON ?= python3
VENV    = .venv
VENVBIN = $(VENV)/bin
TORCH_INDEX = https://download.pytorch.org/whl/cu130

.PHONY: setup ext test test-cpu bench bench-throughput bench-scheduler bench-kernels \
        bench-prefix-cache bench-fp8 bench-spec clean help
.DEFAULT_GOAL := help

# A local venv for a fresh clone; `make test` uses whatever python is on PATH instead.
setup:
	$(PYTHON) -m venv $(VENV)
	$(VENVBIN)/pip install --upgrade pip
	$(VENVBIN)/pip install -r requirements.txt --extra-index-url $(TORCH_INDEX)
	@echo "setup: activate with 'source $(VENVBIN)/activate'"

# A stale JIT cache is the first thing to suspect when a kernel edit appears to do nothing.
ext:
	$(PYTHON) -m mini_vllm.kernels --rebuild

test:
	$(PYTHON) -m pytest -q

test-cpu:
	$(PYTHON) -m pytest -q -m "not cuda and not oracle and not slow"

bench:
	$(PYTHON) -m mini_vllm.benchmark --mode single --compare hf --use-cuda-kernels

bench-throughput:
	$(PYTHON) -m mini_vllm.benchmark --mode throughput --batch-sizes 1,4,16,32 \
		--compare hf --use-cuda-kernels --kv-fraction 0.35

bench-scheduler:
	$(PYTHON) -m mini_vllm.benchmark --mode scheduler --num-requests 2000 --use-cuda-kernels

bench-kernels:
	$(PYTHON) -m mini_vllm.benchmark --mode kernels

bench-prefix-cache:
	$(PYTHON) -m mini_vllm.benchmark --mode prefix-cache --use-cuda-kernels

bench-fp8:
	$(PYTHON) -m mini_vllm.benchmark --mode fp8 --use-cuda-kernels

bench-spec:
	$(PYTHON) -m mini_vllm.benchmark --mode spec --draft-layers 4,14,28 --use-cuda-kernels

clean:
	rm -rf $(VENV) build .pytest_cache .hypothesis
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

help:
	@echo "Mini-vLLM targets:"
	@echo "  setup     create .venv and install pinned requirements (CUDA 13 torch)"
	@echo "  ext       force-rebuild the csrc/ CUDA extension, print the toolchain"
	@echo "  test      run the pytest suite (cuda/oracle tests skip when unavailable)"
	@echo "  test-cpu  run only the tests needing neither a GPU nor model weights"
	@echo "  bench     TTFT and decode tok/s for one request, against transformers"
	@echo "  bench-throughput  the headline: output tok/s vs transformers, by concurrency"
	@echo "  bench-scheduler   decode-latency tails: chunked prefill vs prefill-first"
	@echo "  bench-kernels  achieved bandwidth per kernel vs the torch it replaced"
	@echo "  bench-prefix-cache  TTFT with and without radix-tree prefix caching"
	@echo "  bench-fp8      FP8 vs BF16 KV cache: capacity per budget and greedy agreement"
	@echo "  bench-spec     speculative decoding: acceptance rate and wall clock by draft depth"
	@echo "  clean     remove .venv, build/, caches, and __pycache__"
