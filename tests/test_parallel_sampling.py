"""Parallel sampling, and the copy-on-write it finally puts on the live path.

`n > 1` prefills a prompt once and forks the result into `n` branches. The block manager
already supported sharing a prefix and copying the one page two branches would both write
into; parallel sampling is the first caller on the serving path, so these tests exercise
fork-and-copy-on-write the way a running engine does rather than by driving the manager
directly.

Two levels:

* Block accounting, on the manager, with no model: eight branches of a 4000-token prompt
  hold 250 blocks between them rather than 2000, and the first write on a branch copies
  exactly the one shared partial page.
* Output, on the engine with real weights (``@pytest.mark.oracle``): a prompt asked for
  `n` times returns `n` completions, and greedy branches each reproduce the single-sample
  run token for token.
"""

from __future__ import annotations

import pytest
import torch

from conftest import real_engine

from mini_vllm.block.block_manager import BlockManager
from mini_vllm.sampler import SamplingParams
from mini_vllm.serve.sequence import Sequence, SequenceStatus

GREEDY = SamplingParams(temperature=0.0)


# ------------------------------------------------------------------- sequence


def test_n_must_be_positive():
    with pytest.raises(ValueError, match="n must be >= 1"):
        SamplingParams(n=0)


def test_fork_copies_the_prompt_and_takes_its_own_first_token():
    parent = Sequence(prompt_token_ids=[1, 2, 3], sampling_params=SamplingParams(n=4))
    parent.num_computed_tokens = 3

    child = parent.fork(first_output_token=42)

    assert child.prompt_token_ids == [1, 2, 3]
    assert child.output_token_ids == [42], "the branch continues on its own token"
    assert child.num_computed_tokens == 3, "the shared prefix is already computed"
    assert child.parent_id == parent.seq_id
    assert child.status is SequenceStatus.RUNNING
    assert child.block_table is None, "the block table belongs to the manager's fork"


def test_forks_of_a_fork_group_under_the_original():
    parent = Sequence(prompt_token_ids=[1, 2], sampling_params=SamplingParams(n=3))
    first = parent.fork(first_output_token=5)
    second = first.fork(first_output_token=6)

    # A branch's branch still points at the group leader, not at the intermediate.
    assert first.parent_id == parent.seq_id
    assert second.parent_id == parent.seq_id


# --------------------------------------------------------------- block sharing


def test_eight_branches_share_one_prompt():
    """The headline: forking a 4000-token prompt eight ways costs 250 blocks, not 2000."""
    manager = BlockManager(num_blocks=300, block_size=16)
    prompt = list(range(1, 3991))  # 3990 tokens -> 250 blocks, last one partial
    parent = Sequence(prompt_token_ids=prompt, sampling_params=SamplingParams(n=8))
    manager.allocate(parent)

    held = manager.num_blocks - manager.num_free_blocks
    assert held == 250, "3990 tokens in blocks of 16"

    children = []
    for _ in range(7):
        child = parent.fork(first_output_token=999)
        manager.fork(parent, child)
        children.append(child)

    assert manager.num_free_blocks == manager.num_blocks - 250, "the fork copied nothing"
    assert all(
        manager.pool.ref_count(block) == 8 for block in manager.table(parent).block_ids
    ), "all eight branches share every prompt page"

    for child in children:
        manager.free(child)
    manager.free(parent)
    manager.check_no_leaks()


def test_the_first_write_on_a_branch_copies_one_page():
    """Copy-on-write, reached the way the engine reaches it: fork, then decode."""
    manager = BlockManager(num_blocks=64, block_size=16, num_layers=1, num_kv_heads=1, head_dim=8)
    prompt = list(range(1, 40))  # 39 tokens -> 3 blocks, the last partial
    parent = Sequence(prompt_token_ids=prompt, sampling_params=SamplingParams(n=2))
    manager.allocate(parent)
    shared = manager.table(parent).block_ids

    child = parent.fork(first_output_token=7)
    manager.fork(parent, child)

    free_before = manager.num_free_blocks
    manager.append_slot(child)  # the branch's first decode writes past the shared prefix

    child_ids = manager.table(child).block_ids
    assert child_ids[:2] == shared[:2], "the full prefix blocks stay shared"
    assert child_ids[2] != shared[2], "the shared partial page was copied for the branch"
    assert manager.num_free_blocks == free_before - 1, "exactly one page was copied"

    manager.free(child)
    manager.free(parent)
    manager.check_no_leaks()


# -------------------------------------------------------------- engine output


@pytest.mark.oracle
def test_parallel_sampling_returns_n_completions_that_match_a_solo_run():
    """Greedy `n = 3` yields three completions, each identical to the single-sample run.

    Greedy on purpose: parallel sampling's value is diverse continuations, but its
    *correctness* is that each branch sees exactly the logits it would alone, and greedy
    turns that into an exact token check instead of a distributional one.

    fp32 rather than the bf16 the engine serves in, for the reason `test_engine.py`
    documents: in bf16 the top two logits of a Qwen3 step are often one rounding apart,
    so a batch of one and a batch of three break those ties differently and the test
    would fail for arithmetic rather than for a bug. Both runs share one engine, because
    an fp32 Qwen3-0.6B is 2.4 GB and there is 8 GB on the card.
    """
    with real_engine(dtype=torch.float32) as llm:
        prompt = "The capital of France is"

        solo = llm.generate(prompt, sampling_params=GREEDY, max_tokens=16)
        assert len(solo) == 1

        triple = llm.generate(
            prompt, sampling_params=SamplingParams(temperature=0.0, n=3), max_tokens=16
        )

    assert len(triple) == 3, "three samples for one prompt"
    assert [completion.sample_index for completion in triple] == [0, 1, 2]
    for completion in triple:
        assert completion.token_ids == solo[0].token_ids, "a branch drifted from the solo run"


@pytest.mark.oracle
def test_parallel_sampling_leaves_no_blocks_behind():
    """Every branch's pages return to the pool once the group finishes."""
    with real_engine() as llm:
        free_before = llm.manager.num_free_blocks

        llm.generate(
            ["The capital of France is", "Once upon a time"],
            sampling_params=SamplingParams(temperature=0.0, n=4),
            max_tokens=12,
        )

        assert llm.manager.num_free_blocks == free_before, "a parallel-sampling run leaked blocks"
