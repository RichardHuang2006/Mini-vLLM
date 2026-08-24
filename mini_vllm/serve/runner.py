"""Executing a scheduler decision against the paged cache.

The replacement for `DenseModelRunner`. The dense runner issues one forward pass per
scheduled sequence, since a `B x H x S x D` cache cannot hold two sequences of
different lengths. This one builds a single ragged `ForwardBatch` and runs one pass for
the whole iteration: a 512-token prefill chunk and eleven decode steps together.

Ordering inside `execute` matters. Blocks are reserved before the batch is built,
because `slot_mapping` holds physical addresses and there is nothing to address until
the pages exist. The scheduler has already checked capacity, so the reservation is
expected to succeed; a failure means the scheduler and the pool have disagreed, and it
propagates rather than being absorbed.
"""

from __future__ import annotations

import torch

from mini_vllm.block.block_manager import BlockManager
from mini_vllm.sampler import sample
from mini_vllm.serve.batch import ForwardBatch
from mini_vllm.serve.scheduler import SchedulerOutput
from mini_vllm.serve.sequence import Sequence

__all__ = ["PagedModelRunner"]


class PagedModelRunner:
    """Runs one iteration: reserve pages, build the ragged batch, forward, sample."""

    def __init__(self, model, manager: BlockManager, device: torch.device | str | None = None):
        self.model = model
        self.manager = manager
        self.device = torch.device(device) if device else manager.kv.device

    def execute(self, output: SchedulerOutput, all_rows: bool = False) -> torch.Tensor:
        """One forward pass over the whole scheduled batch.

        ::

            returns: num_scheduled x V   (each sequence's last computed position)
                     total_tokens  x V   when `all_rows`

        `all_rows` is for speculative verification, which needs the target's
        distribution at every proposed position rather than only the last. Off
        otherwise: the LM head is a `V`-wide matmul and a prefill chunk's interior rows
        have no use for it.
        """
        for sequence, count in output.scheduled:
            self.manager.allocate(sequence, count)

        batch = self.build(output)
        return self.model(batch, all_rows=all_rows)

    def build(self, output: SchedulerOutput) -> ForwardBatch:
        """The batch for an already-reserved iteration. Split out for the tests."""
        return ForwardBatch.from_scheduled(output.scheduled, self.device, manager=self.manager)

    def sample_tokens(self, output: SchedulerOutput, logits: torch.Tensor) -> list[int]:
        """One token per scheduled sequence, honouring per-row sampling parameters.

        Rows belonging to a chunk that has not reached the end of its prompt are
        sampled and then discarded by `Scheduler.commit`: one wasted row of an
        already-batched sample instead of a branch in the hot path. Slicing the logits
        down to the finishing sequences first would cost a device synchronization to
        determine which those are.
        """
        params = [sequence.sampling_params for sequence, _ in output.scheduled]
        if all(parameter.is_greedy for parameter in params):
            return logits.argmax(dim=-1).tolist()
        return sample(logits, params).tolist()

    def free(self, sequence: Sequence) -> None:
        """Return a finished sequence's pages to the pool."""
        self.manager.free(sequence)
