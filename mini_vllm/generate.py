"""Greedy generation, with and without a KV cache.

:func:`generate_ids` is the naive loop. Every step re-runs all 28 layers over the
entire prefix, so generating token 100 redoes the work of tokens 0-99 for the
hundredth time — quadratic total cost for a job that should be linear.

:func:`generate_ids_cached` is the cached version, and it is kept side by side
with the naive one on purpose: they must produce **identical tokens**, which is
the only convincing evidence that the cache is a pure optimization rather than a
subtle change of behaviour. The `--no-cache` flag exists so the difference can be
felt rather than described.

Run it::

    python -m mini_vllm.generate --prompt "The capital of France is"
    python -m mini_vllm.generate --prompt "The capital of France is" --no-cache
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from typing import NamedTuple

import torch

from mini_vllm.model.loader import DEFAULT_MODEL_ID, resolve_model_path
from mini_vllm.model.qwen3 import Qwen3
from mini_vllm.model.qwen3_cached import Qwen3Cached

__all__ = [
    "Loaded",
    "eos_token_ids_for",
    "generate",
    "generate_ids",
    "generate_ids_cached",
    "load",
]

DEFAULT_MAX_TOKENS = 32


class Loaded(NamedTuple):
    """A model, its tokenizer, and the stop tokens that go with them."""

    model: Qwen3 | Qwen3Cached
    tokenizer: object
    eos_token_ids: tuple[int, ...]
    pad_token_id: int


def eos_token_ids_for(model_path) -> tuple[int, ...]:
    """The stop tokens, read from `generation_config.json`.

    Qwen3 lists *two* (`<|im_end|>` and `<|endoftext|>`), which is why this
    returns a tuple rather than a single id. Honouring only `tokenizer.eos_token_id`
    would miss one and generate past the end of a turn.
    """
    config_path = model_path / "generation_config.json"
    if not config_path.is_file():
        return ()

    stated = json.loads(config_path.read_text()).get("eos_token_id")
    if stated is None:
        return ()
    return (stated,) if isinstance(stated, int) else tuple(stated)


def load(
    model: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    cached: bool = True,
    use_cuda_kernels: bool = False,
) -> Loaded:
    """Load the model, tokenizer and stop tokens together.

    ``cached=False`` gives the uncached model, which exists to be compared
    against rather than used.
    """
    from transformers import AutoTokenizer

    path = resolve_model_path(model)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if cached:
        loaded_model = Qwen3Cached.from_pretrained(
            path, device=device, use_cuda=use_cuda_kernels
        )
    else:
        loaded_model = Qwen3.from_pretrained(path, device=device)

    return Loaded(
        model=loaded_model,
        tokenizer=AutoTokenizer.from_pretrained(path),
        eos_token_ids=eos_token_ids_for(path),
        pad_token_id=json.loads((path / "generation_config.json").read_text()).get(
            "pad_token_id", 0
        ),
    )


@torch.no_grad()
def generate_ids(
    model: Qwen3,
    input_ids: torch.Tensor,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    eos_token_ids: Sequence[int] = (),
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """Greedy decode, returning the prompt with the generated tokens appended.

    ::

        input_ids: B x L      ->  B x (L + generated)

    Stops once every row has produced a stop token. Rows that finish early are
    filled with ``pad_token_id`` so the batch stays rectangular, which is what
    HuggingFace does too — and is a hint at why the serving layer abandons
    rectangular batches entirely: with 16 sequences of wildly different lengths,
    most of a padded batch is wasted work.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"expected B x L input ids, got shape {tuple(input_ids.shape)}")

    stop_tokens = set(eos_token_ids)
    if pad_token_id is None:
        pad_token_id = next(iter(stop_tokens), 0)

    tokens = input_ids
    finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)

    for _ in range(max_tokens):
        # The waste: the whole prefix, every single step.
        logits = model(tokens)[:, -1, :]
        next_tokens = logits.argmax(dim=-1)

        next_tokens = torch.where(
            finished, torch.full_like(next_tokens, pad_token_id), next_tokens
        )
        tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)

        for stop in stop_tokens:
            finished |= next_tokens == stop
        if bool(finished.all()):
            break

    return tokens


@torch.no_grad()
def generate_ids_cached(
    model: Qwen3Cached,
    input_ids: torch.Tensor,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    eos_token_ids: Sequence[int] = (),
    pad_token_id: int | None = None,
    caches: list | None = None,
) -> torch.Tensor:
    """Greedy decode with a KV cache: prefill once, then one token per step.

    ::

        prefill: the whole prompt at offset 0   -> first token, cache holds len(prompt)
        decode:  one token at offset = prev_len -> next token, cache grows by 1

    The loop looks almost the same as the naive one, and that is the point — the
    only structural difference is that it feeds `next_tokens` rather than the whole
    sequence back in. The caches carry the position, so nothing here tracks an
    offset by hand.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"expected B x L input ids, got shape {tuple(input_ids.shape)}")

    stop_tokens = set(eos_token_ids)
    if pad_token_id is None:
        pad_token_id = next(iter(stop_tokens), 0)

    if caches is None:
        caches = model.create_kv_cache()

    tokens = input_ids
    finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
    step_input = input_ids

    for _ in range(max_tokens):
        # Only the tokens the model has not seen: the whole prompt on the first
        # pass, then exactly one per step.
        logits = model(step_input, caches, last_only=True)[:, -1, :]
        next_tokens = logits.argmax(dim=-1)

        next_tokens = torch.where(
            finished, torch.full_like(next_tokens, pad_token_id), next_tokens
        )
        tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)
        step_input = next_tokens.unsqueeze(1)

        for stop in stop_tokens:
            finished |= next_tokens == stop
        if bool(finished.all()):
            break

    return tokens


def generate(
    loaded: Loaded,
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    skip_special_tokens: bool = True,
) -> str:
    """Greedily continue ``prompt``, returning only the newly generated text.

    Uses whichever loop matches the loaded model, so the two are interchangeable
    from here up.
    """
    device = loaded.model.embedding.weight.device
    input_ids = loaded.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    loop = (
        generate_ids_cached if isinstance(loaded.model, Qwen3Cached) else generate_ids
    )
    tokens = loop(
        loaded.model,
        input_ids,
        max_tokens=max_tokens,
        eos_token_ids=loaded.eos_token_ids,
        pad_token_id=loaded.pad_token_id,
    )

    generated = tokens[0, input_ids.shape[1] :]
    return loaded.tokenizer.decode(generated, skip_special_tokens=skip_special_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy generation, with or without a KV cache.")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="use the uncached loop, which recomputes the whole prefix every step",
    )
    arguments = parser.parse_args()

    print(f"loading {arguments.model} ...")
    loaded = load(arguments.model, arguments.device, cached=not arguments.no_cache)
    device = loaded.model.embedding.weight.device
    prompt_length = loaded.tokenizer(arguments.prompt, return_tensors="pt").input_ids.shape[1]

    started = time.perf_counter()
    text = generate(loaded, arguments.prompt, arguments.max_tokens)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    generated_count = len(loaded.tokenizer(text).input_ids) or 1

    print(f"\n{arguments.prompt}\033[1m{text}\033[0m\n")
    print(f"device {device}, prompt {prompt_length} tokens, generated {generated_count}")
    print(f"{elapsed:.2f}s total, {generated_count / elapsed:.1f} tokens/s")

    if arguments.no_cache:
        print(
            "no KV cache: every step recomputes the whole prefix, so this rate falls "
            "as the sequence grows. Drop --no-cache to compare."
        )
    else:
        print("with a KV cache: each step forwards a single token.")


if __name__ == "__main__":
    main()
