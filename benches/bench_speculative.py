"""Speculative decoding by draft depth: one greedy request at a time, the draft proposing k
tokens per step for the target to verify. It pays only when the draft is accepted often and
costs much less than the target, and a self-draft cannot be both.

    uv run python benches/bench_speculative.py --metal
    uv run python benches/bench_speculative.py --metal --model Qwen/Qwen3-1.7B --draft-model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import time

from bench_quantize import PROMPTS
from matplotlib.figure import Figure
from utils import engine, evidence, load_mlx_lm, save_plot

from mini_vllm import DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--metal", action="store_true", help="run on the Metal kernels in src/extensions/")
    parser.add_argument("--draft-model", help="a separate, smaller checkpoint sharing the tokenizer")
    parser.add_argument("--draft-layers", type=int, nargs="+", default=[4, 14, 28],
                        help="self-draft depths: the target's first N layers")
    parser.add_argument("-k", type=int, default=4, help="tokens proposed per step")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    title = (f"Speculative decoding: k={args.k}, {len(PROMPTS)} greedy requests of "
             f"{args.max_tokens} tokens, one at a time, {'metal' if args.metal else 'pure'}")
    with evidence(title):
        mlx_model, tokenizer = load_mlx_lm(args.model)
        layers = mlx_model.args.num_hidden_layers
        drafts = [("none", {})]
        for n in args.draft_layers:
            drafts.append((f"first {n} of {layers} layers", {"num_speculative_tokens": args.k,
                                                             "num_draft_layers": n}))
        if args.draft_model:
            drafts.append((args.draft_model, {"num_speculative_tokens": args.k,
                                              "draft_model": args.draft_model}))

        print("| draft | acceptance | tokens per target pass | seconds | identical to no draft |")
        print("|---|---|---|---|---|")
        baseline = None
        labels, seconds_by_draft, tokens_per_pass = [], [], []
        for label, config in drafts:
            llm = engine(mlx_model, tokenizer, args.metal, num_blocks=256, **config)
            if llm.spec is not None:
                llm.spec.stats.__init__()  # forget the warm-up
            start = time.perf_counter()
            outputs = [llm.generate(prompt, max_tokens=args.max_tokens)[0].token_ids for prompt in PROMPTS]
            seconds = time.perf_counter() - start
            baseline = baseline or outputs
            same = sum(a == b for a, b in zip(outputs, baseline, strict=True))
            labels.append(label)
            seconds_by_draft.append(seconds)
            tokens_per_pass.append(1.0 if llm.spec is None else llm.spec.stats.tokens_per_step)
            if llm.spec is None:
                print(f"| {label} | — | 1.00 | {seconds:.2f} | — |")
            else:
                stats = llm.spec.stats
                print(f"| {label} | {stats.acceptance_rate:.3f} | {stats.tokens_per_step:.2f} | "
                      f"{seconds:.2f} | {same} of {len(PROMPTS)} |")
        print("\nOutput that differs from no draft differs at a bf16 near-tie: the verify pass computes"
              " k + 1 rows where a plain decode computes one, so its matmuls round differently.")

        figure = Figure(figsize=(11, 4))
        figure.suptitle(title)
        seconds_axes, passes_axes = figure.subplots(1, 2, sharey=True)
        seconds_axes.barh(labels, seconds_by_draft)
        seconds_axes.set_xlabel("seconds")
        seconds_axes.invert_yaxis()
        passes_axes.barh(labels, tokens_per_pass)
        passes_axes.set_xlabel("tokens per target pass")
        save_plot(figure, "bench_speculative")


if __name__ == "__main__":
    main()
