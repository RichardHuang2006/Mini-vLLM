"""The engine's public API, and speculative decoding end to end.

Four sections:

* Rejection sampling as mathematics: the residual identity, a 120k-draw chi-square
  against a deliberately wrong draft (the distributional claim no example can make),
  and the greedy degenerate case. Pure CPU.
* Speculative bookkeeping: proposals live outside the output, acceptance commits
  exactly the surviving run, rollback returns exactly the rejected slots, and a full
  propose/reject/trim loop leaks nothing. Pure CPU.
* The `LLM` API on real weights: token-identical to `transformers.generate` for
  sixteen varied prompts, the stream and the batch API agreeing by construction,
  stop tokens, parallel sampling's CoW branches, prefix-cache identity, and resource
  release when a caller walks away.
* Speculation on real weights: greedy speculative output identical to greedy
  non-speculative output (including with a deliberately bad draft — rejection
  sampling makes draft quality irrelevant to correctness), self-draft acceptance
  1.0, whole accepted runs streamed in order, and both pools balanced after every
  test.

The engine tests build one engine at a time through `conftest.real_engine`: the card
of record has 8 GB, and an fp32 Qwen3-0.6B plus two KV pools is most of it.
"""

from __future__ import annotations

import pytest
import torch
from conftest import assert_tokens_equal, free_cuda_memory, real_engine

from mini_vllm.cache import BlockManager, BlockTable, PagedKvPool
from mini_vllm.config import SamplingParams
from mini_vllm.scheduler import Sequence, SequenceStatus
from mini_vllm.speculative import (
    ResidualError,
    accept_proposals,
    residual_distribution,
)

GREEDY = SamplingParams(temperature=0.0)


def distribution(*weights: float) -> torch.Tensor:
    tensor = torch.tensor(weights, dtype=torch.float64)
    return tensor / tensor.sum()


def rows(*distributions: torch.Tensor) -> torch.Tensor:
    return torch.stack(list(distributions))


# ========================================================== rejection sampling


def test_the_residual_is_the_mass_acceptance_leaves_behind():
    """``relu(p - q)`` normalized, and its raw mass is exactly ``1 - sum(min(p, q))``.

    That identity is the whole reason the correction is this expression and not
    another: the acceptance step emits token t with probability min(p(t), q(t)), so
    the shortfall it leaves is precisely the mass of relu(p - q).
    """
    target = distribution(0.6, 0.3, 0.1)
    draft = distribution(0.1, 0.7, 0.2)

    residual = residual_distribution(target, draft)

    raw = (target - draft).clamp_min(0.0)
    shortfall = 1.0 - torch.minimum(target, draft).sum()
    assert torch.isclose(raw.sum(), shortfall)
    assert torch.isclose(residual.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert residual[0] > 0 and residual[1] == 0, "mass only where the target wants more"


def test_an_empty_residual_is_an_error_not_a_silent_zero():
    """p == q cannot be rejected, so being asked for its residual is a caller bug."""
    same = distribution(0.5, 0.5)
    with pytest.raises(ResidualError, match="no mass"):
        residual_distribution(same, same)


@pytest.mark.parametrize("num_proposals", [1, 3])
def test_the_emitted_token_matches_the_target_distribution(num_proposals: int):
    """The headline claim, as a chi-square test over 120k sampled first tokens.

    The draft is deliberately wrong — its mass is the target's reversed — so a
    procedure that simply trusted the draft, or corrected rejections with the wrong
    distribution, would produce a visibly different histogram. Only the accept/reject
    rule with the `relu(p - q)` residual reproduces the target.

    The *first* emitted token is the one measured: it is the position every trial has
    in common, and where the correction acts.
    """
    vocabulary = 5
    target = distribution(0.40, 0.25, 0.20, 0.10, 0.05)
    draft = distribution(0.05, 0.10, 0.20, 0.25, 0.40)

    generator = torch.Generator().manual_seed(1234)
    trials = 120_000
    counts = torch.zeros(vocabulary, dtype=torch.float64)

    target_rows = rows(*[target] * (num_proposals + 1))
    draft_rows = rows(*[draft] * num_proposals)

    for _ in range(trials):
        # The draft proposes from its own distribution, as it does in the engine;
        # sampling the proposals otherwise would test a procedure never run.
        proposals = torch.multinomial(draft, num_proposals, replacement=True, generator=generator)
        emitted, _accepted = accept_proposals(target_rows, draft_rows, proposals,
                                              generator=generator)
        counts[emitted[0]] += 1

    expected = target * trials
    chi_square = float(((counts - expected) ** 2 / expected).sum())
    # 4 degrees of freedom: the 99.9th percentile of chi-square(4) is 18.47. A correct
    # implementation sits near 4; one that trusts the draft lands in the hundreds of
    # thousands.
    assert chi_square < 18.47, (
        f"chi-square {chi_square:.2f} rejects the target distribution; "
        f"got {(counts / trials).tolist()}, wanted {target.tolist()}"
    )


def test_a_matching_draft_is_always_accepted():
    """q == p makes the ratio 1: the self-draft case, and why acceptance rate is a
    meaningful diagnostic."""
    target = distribution(0.5, 0.3, 0.2)
    generator = torch.Generator().manual_seed(7)

    for _ in range(200):
        proposals = torch.multinomial(target, 3, replacement=True, generator=generator)
        emitted, accepted = accept_proposals(
            rows(target, target, target, target), rows(target, target, target), proposals,
            generator=generator,
        )
        assert accepted == 3, "an identical draft was rejected"
        assert emitted[:3] == [int(t) for t in proposals]
        assert len(emitted) == 4, "three accepted proposals plus the bonus"


def test_a_step_always_emits_at_least_one_token():
    """Even total rejection yields the residual draw, so speculation cannot stall."""
    target = distribution(1.0, 0.0, 0.0)
    draft = distribution(0.0, 1.0, 0.0)  # certain about a token the target forbids

    emitted, accepted = accept_proposals(rows(target, target), rows(draft), [1])

    assert accepted == 0 and emitted == [0], "the residual can only be token 0"


def test_rejection_truncates_rather_than_skipping():
    """Proposals after a rejection were conditioned on a token that no longer exists;
    keeping them would be the subtlest possible way to corrupt the output."""
    certain = distribution(1.0, 0.0)
    target_rows = rows(certain, certain, certain)
    draft_rows = rows(distribution(0.0, 1.0), certain)

    emitted, accepted = accept_proposals(target_rows, draft_rows, [1, 0])

    assert accepted == 0
    assert len(emitted) == 1, "a rejected position ends the step"


def test_greedy_keeps_exactly_the_prefix_where_the_models_agree():
    """Temperature zero: accept while the argmaxes match, then take the target's token."""
    target_rows = rows(*[distribution(0.9, 0.1)] * 4)
    draft_rows = rows(*[distribution(0.9, 0.1)] * 3)

    emitted, accepted = accept_proposals(target_rows, draft_rows, [0, 0, 1], greedy=True)

    assert accepted == 2, "the first two proposals matched the target's argmax"
    assert emitted == [0, 0, 0], "the third is replaced by the target's own choice"


def test_greedy_full_acceptance_takes_the_bonus_argmax():
    target_rows = rows(distribution(0.1, 0.9), distribution(0.8, 0.2))
    draft_rows = rows(distribution(0.1, 0.9))

    emitted, accepted = accept_proposals(target_rows, draft_rows, [1], greedy=True)

    assert accepted == 1
    assert emitted == [1, 0], "the bonus is the target's argmax at the next position"


def test_the_shape_contracts_are_enforced():
    probability = distribution(0.5, 0.5)
    with pytest.raises(ValueError, match="plus the bonus"):
        accept_proposals(rows(probability), rows(probability), [0])
    with pytest.raises(ValueError, match="vocabulary"):
        accept_proposals(rows(probability, probability), rows(distribution(0.3, 0.3, 0.4)), [0])


# ===================================================== speculative bookkeeping


def decoding_sequence(prompt: int = 4, outputs: int = 1, **kwargs) -> Sequence:
    """A sequence past prefill with one uncommitted token — the state a decode step
    sees: the last sampled token has been chosen but not yet run through the model."""
    sequence = Sequence(prompt_token_ids=list(range(1, prompt + 1)), max_tokens=64, **kwargs)
    sequence.set_status(SequenceStatus.RUNNING)
    sequence.output_token_ids = [100 + index for index in range(outputs)]
    sequence.num_computed_tokens = len(sequence) - 1
    return sequence


def test_proposals_lengthen_the_sequence_without_joining_the_output():
    sequence = decoding_sequence()
    length_before, output_before = len(sequence), list(sequence.output_token_ids)

    sequence.propose([201, 202, 203])

    assert len(sequence) == length_before + 3, "proposals must be forwarded, so they count"
    assert sequence.output_token_ids == output_before, "a proposal is not output"
    assert sequence.num_uncomputed_tokens == 4, "the pending token plus three proposals"


def test_a_proposed_stop_token_does_not_finish_the_sequence():
    """If a speculated end-of-text could set `is_done`, a request would be returned on
    the strength of a guess the target model was about to reject."""
    sequence = decoding_sequence(eos_token_id=999)
    sequence.propose([999])
    assert not sequence.is_done()
    assert sequence.finish_reason is None


def test_accepting_everything_commits_the_run_and_the_bonus():
    """`k` accepted proposals plus the bonus: `k + 1` tokens out of one forward pass."""
    sequence = decoding_sequence()
    computed_before = sequence.num_computed_tokens
    sequence.propose([201, 202, 203])

    to_trim = sequence.accept([201, 202, 203, 204], num_accepted=3)

    assert to_trim == 0, "nothing was rejected, so nothing is given back"
    assert sequence.output_token_ids[-4:] == [201, 202, 203, 204]
    # The forward computed the pending token and all three proposals; the bonus is not
    # computed, which restores the usual one-uncommitted-token invariant.
    assert sequence.num_computed_tokens == computed_before + 4
    assert sequence.num_uncomputed_tokens == 1


def test_rejecting_the_tail_gives_back_exactly_the_rejected_slots():
    sequence = decoding_sequence()
    computed_before = sequence.num_computed_tokens
    sequence.propose([201, 202, 203])

    to_trim = sequence.accept([201, 777], num_accepted=1)

    assert to_trim == 2, "two proposals were rejected, so two slots go back"
    assert sequence.output_token_ids[-2:] == [201, 777]
    assert sequence.num_computed_tokens == computed_before + 2
    assert sequence.num_uncomputed_tokens == 1


def test_a_stop_token_mid_run_truncates_the_rest():
    """Tokens after the stop were computed, but the sequence ended before them."""
    sequence = decoding_sequence(eos_token_id=999)
    sequence.propose([201, 999, 203])

    to_trim = sequence.accept([201, 999, 203, 204], num_accepted=3)

    assert sequence.output_token_ids[-2:] == [201, 999], "output ran past the stop token"
    assert sequence.is_done() and sequence.finish_reason == "stop"
    assert to_trim == 1, "one proposal survived; the other two go back"


def test_accept_rejects_tokens_that_were_never_proposed():
    sequence = decoding_sequence()
    sequence.propose([201, 202])
    with pytest.raises(ValueError, match="not the proposals"):
        sequence.accept([888, 999], num_accepted=1)


def test_preemption_drops_unverified_proposals():
    """They never belonged to the request, so a recomputed prefill must not include
    them."""
    sequence = decoding_sequence()
    sequence.propose([201, 202])
    length_with_proposals = len(sequence)

    sequence.reset_for_recompute()

    assert sequence.proposed_token_ids == []
    assert len(sequence) < length_with_proposals


def test_a_full_speculative_round_trip_leaks_no_blocks():
    """Propose, reject most of it, roll back, repeat: a speculative step that goes
    badly costs nothing permanent. Twelve rounds, because a leak of one page per step
    is invisible once and fatal a thousand times."""
    manager = BlockManager(num_blocks=32, block_size=4)
    sequence = Sequence(prompt_token_ids=[1, 2, 3, 4], max_tokens=256)
    sequence.set_status(SequenceStatus.RUNNING)
    manager.allocate(sequence, 4)
    sequence.num_computed_tokens = 4
    sequence.append_token(100)

    free_at_start = manager.num_free_blocks
    blocks_at_start = manager.table(sequence).num_blocks

    for round_index in range(12):
        proposals = [200 + round_index, 201 + round_index, 202 + round_index]
        sequence.propose(proposals)
        manager.allocate(sequence, sequence.num_uncomputed_tokens)

        to_trim = sequence.accept([proposals[0], 900 + round_index], num_accepted=1)
        manager.trim(sequence, to_trim)

        # The manager's reservation and the sequence's own count must not drift apart.
        assert manager.table(sequence).num_tokens == sequence.num_computed_tokens
        assert sequence.num_uncomputed_tokens == 1

    grown = manager.table(sequence).num_blocks - blocks_at_start
    assert manager.num_free_blocks == free_at_start - grown, "a rejected page never came back"

    manager.free(sequence)
    assert manager.num_free_blocks == free_at_start + blocks_at_start


def test_trim_tokens_reports_but_does_not_release():
    """Occupancy moves on the table; only the manager hands pages back."""
    table = BlockTable(4, block_ids=[10, 11, 12], num_tokens=9)
    assert table.trim_tokens(2) == 1
    assert table.num_blocks == 3


# ================================================================ the LLM API


@pytest.fixture(scope="module")
def llm():
    """One fp32 engine for the whole module: the weights are 2.4 GB in fp32.

    fp32 rather than the bf16 the engine serves in, for one reason: in bf16 the top two
    logits of a Qwen3 step are frequently one rounding apart, so greedy decoding has
    genuine ties and two correct implementations pick differently. fp32 removes the
    ambiguity, so a disagreement here means a bug rather than a rounding.
    """
    from mini_vllm import LLM
    from mini_vllm.model import resolve_model_path

    path = resolve_model_path()
    if not (path / "model.safetensors").is_file():
        pytest.skip("Qwen3-0.6B weights are not downloaded")
    if not torch.cuda.is_available():
        pytest.skip("the engine's own tests want the kernels")

    engine = LLM(dtype=torch.float32, num_blocks=192, max_sequences=16)
    yield engine
    del engine
    free_cuda_memory()


REAL_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time",
    "The three primary colors are",
    "Water boils at",
    "2 + 2 =",
    "The largest planet in the solar system is",
    "Shakespeare wrote",
    "In Python, a list comprehension",
    "The speed of light is approximately",
    "A neural network learns by",
    "The Pacific Ocean is",
    "To sort a list in Python you can",
    "The first president of the United States was",
    "Photosynthesis converts",
    "A paged attention kernel exists because",
]


@pytest.mark.oracle
def test_greedy_output_matches_transformers_for_sixteen_prompts(llm):
    """Sixteen varied prompts, batched through the engine, against
    `transformers.generate` one prompt at a time, token for token.

    Continuous batching, chunked prefill, a paged cache and hand-written kernels on
    one side; the reference running each request alone on the other. Greedy decoding
    makes the comparison exact: either the engine is right or it is not. The reference
    runs unbatched deliberately — left-padding changes its own answer, which would
    make a disagreement ambiguous.
    """
    from transformers import AutoModelForCausalLM

    from mini_vllm.model import resolve_model_path

    theirs = AutoModelForCausalLM.from_pretrained(resolve_model_path(), dtype=llm.config.dtype)
    theirs = theirs.to(llm.device).eval()
    try:
        completions = llm.generate(REAL_PROMPTS, max_tokens=24)

        for prompt, completion in zip(REAL_PROMPTS, completions, strict=True):
            ids = llm.tokenizer(prompt, return_tensors="pt").input_ids.to(llm.device)
            reference = theirs.generate(ids, max_new_tokens=24, do_sample=False,
                                        pad_token_id=llm.tokenizer.eos_token_id)
            assert_tokens_equal(
                completion.token_ids, reference[0, ids.shape[1] :],
                msg=f"prompt: {prompt!r}\nours:   {completion.text!r}",
            )
    finally:
        # Both models fp32 on one card is 5 GB, so the reference does not outlive the
        # test that needed it.
        del theirs
        torch.cuda.empty_cache()


@pytest.mark.oracle
def test_the_batch_api_is_the_streaming_api_drained(llm):
    """Two entry points, one loop: the deltas must reassemble into the completion."""
    prompts = ["The capital of France is", "def fibonacci(n):"]
    batched = llm.generate(prompts, max_tokens=16)

    streamed: dict[int, str] = {0: "", 1: ""}
    stream_tokens: dict[int, list[int]] = {0: [], 1: []}
    for update in llm.generate_stream(prompts, max_tokens=16):
        streamed[update.index] += update.text
        stream_tokens[update.index].append(update.token_id)

    for index, completion in enumerate(batched):
        assert streamed[index] == completion.text
        assert_tokens_equal(stream_tokens[index], completion.token_ids)


@pytest.mark.oracle
def test_a_stream_is_interleaved_across_prompts(llm):
    """A stream that delivered one request at a time would be a batch API with extra
    steps, and would mean the engine was not batching."""
    updates = list(llm.generate_stream(
        ["The capital of France is", "def fibonacci(n):"], max_tokens=8))
    indices = [update.index for update in updates]

    assert set(indices) == {0, 1}
    assert indices != sorted(indices), "the two requests must be interleaved"


@pytest.mark.oracle
def test_a_completion_says_why_it_stopped(llm):
    capped = llm.generate("Once upon a time", max_tokens=4)[0]
    assert capped.finish_reason == "length" and capped.num_tokens == 4

    prompt = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say the single word: hello"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ended = llm.generate(prompt, max_tokens=64)[0]

    assert ended.finish_reason == "stop", f"got {ended.text!r}"
    assert ended.num_tokens < 64
    assert llm.tokenizer.eos_token not in ended.text, "special tokens must not reach the text"


@pytest.mark.oracle
def test_an_abandoned_stream_returns_its_blocks(llm):
    """A caller that stops reading must not cost the engine a request's worth of pages
    — the difference between a web server surviving disconnects and one that stops
    admitting after a few hundred of them."""
    before = llm.manager.num_free_blocks

    for _update in llm.generate_stream(["Once upon a time"] * 4, max_tokens=64):
        break  # the disconnect

    assert llm.manager.num_free_blocks == before
    assert llm.scheduler.num_unfinished == 0


@pytest.mark.oracle
def test_sampling_parameters_are_per_request(llm):
    prompts = ["The capital of France is"] * 2
    params = [GREEDY, SamplingParams(temperature=1.5, top_p=0.9)]

    completions = llm.generate(prompts, sampling_params=params, max_tokens=12)

    assert completions[0].text == llm.generate(prompts[:1], max_tokens=12)[0].text
    assert completions[1].num_tokens == 12


@pytest.mark.oracle
def test_parallel_sampling_returns_n_greedy_branches_identical_to_solo(llm):
    """`n=4` greedy branches of one prompt: one prefill, four forks sharing its pages
    through copy-on-write, each branch identical to a solo run."""
    prompt = "The capital of France is"
    solo = llm.generate(prompt, max_tokens=12)[0]

    branches = llm.generate(prompt, sampling_params=SamplingParams(temperature=0.0, n=4),
                            max_tokens=12)

    assert len(branches) == 4
    for branch in branches:
        assert_tokens_equal(branch.token_ids, solo.token_ids)
    assert {branch.sample_index for branch in branches} == {0, 1, 2, 3}
    assert llm.scheduler.num_unfinished == 0


@pytest.mark.oracle
def test_the_engine_reports_what_it_did(llm):
    before = llm.stats.iterations
    llm.generate(["Once upon a time"] * 4, max_tokens=8)

    assert llm.stats.iterations - before <= 9, "4 prompts should share their iterations"
    assert llm.stats.tokens_per_second > 0
    assert llm.kv_cache_bytes > 0


# ------------------------------------------------- prefix caching, engine level


@pytest.mark.oracle
def test_a_prefix_cache_hit_changes_nothing_it_produces():
    """The whole invariant of a cache: it changes how much is computed, never what
    comes out. The second run of a shared preamble hits, and must reproduce the first
    token for token."""
    with real_engine(dtype=torch.float32, enable_prefix_caching=True) as llm:
        preamble = "You are a careful assistant. Answer concisely and correctly. " * 6
        prompt = preamble + "The capital of France is"

        first = llm.generate(prompt, sampling_params=GREEDY, max_tokens=16)[0]
        hits_after_first = llm.stats.cached_tokens

        second = llm.generate(prompt, sampling_params=GREEDY, max_tokens=16)[0]

        assert second.token_ids == first.token_ids, "the cache changed the output"
        assert llm.stats.cached_tokens > hits_after_first, "the second run did not hit"


@pytest.mark.oracle
def test_prefix_caching_matches_the_uncached_engine():
    """Caching on must not perturb even a cold run against caching off. Two engines,
    strictly one at a time: both do not fit on 8 GB."""
    prompt = "The capital of France is"
    with real_engine(dtype=torch.float32, enable_prefix_caching=True) as llm:
        cached = llm.generate(prompt, sampling_params=GREEDY, max_tokens=16)[0]
    with real_engine(dtype=torch.float32, enable_prefix_caching=False) as llm:
        plain = llm.generate(prompt, sampling_params=GREEDY, max_tokens=16)[0]

    assert cached.token_ids == plain.token_ids


# ==================================================== speculation, end to end


def assert_no_spec_leaks(llm) -> None:
    """Both pools back to full: the target's and the draft's."""
    assert llm.manager.num_free_blocks == llm.manager.num_blocks, "the target pool leaked"
    draft = llm.spec.proposer.manager
    assert draft.num_free_blocks == draft.num_blocks, "the draft pool leaked"
    assert llm.spec.proposer.num_tracked == 0, "a shadow sequence outlived its request"


@pytest.mark.oracle
def test_greedy_speculative_output_is_identical_to_non_speculative():
    """The whole claim, as an exact token comparison.

    fp32 rather than bf16: a speculative run reduces in a different order from a plain
    one, so in bf16 the two could differ by a rounding tie without either being wrong.
    fp32 leaves the comparison meaning what it says.
    """
    prompt = "The capital of France is"

    with real_engine(dtype=torch.float32, num_speculative_tokens=4) as llm:
        speculative = llm.generate(prompt, sampling_params=GREEDY, max_tokens=24)[0]
        stats = llm.spec.stats
        assert_no_spec_leaks(llm)

    with real_engine(dtype=torch.float32) as llm:
        plain = llm.generate(prompt, sampling_params=GREEDY, max_tokens=24)[0]

    assert speculative.token_ids == plain.token_ids, (
        "speculation changed the output, which it is not allowed to do"
    )
    assert stats.steps < len(plain.token_ids), "or it was not speculating at all"


@pytest.mark.oracle
def test_a_self_draft_is_accepted_every_time():
    """A draft that *is* the target must agree with it on every proposal.

    Acceptance below 1.0 here means a misaligned verification row or a `q` that is not
    the distribution the proposal was drawn from — the sharpest available check on the
    plumbing, separate from whether the output is right.
    """
    with real_engine(dtype=torch.float32, num_speculative_tokens=4) as llm:
        llm.generate("The capital of France is", sampling_params=GREEDY, max_tokens=24)
        stats = llm.spec.stats

        assert stats.acceptance_rate == 1.0, (
            f"a self-draft was rejected: {stats.accepted}/{stats.proposed} accepted"
        )
        assert stats.tokens_per_step == pytest.approx(5.0, abs=0.5), "four proposals + bonus"
        assert stats.rolled_back_blocks == 0, "nothing was rejected, so nothing rolled back"
        assert_no_spec_leaks(llm)


@pytest.mark.oracle
def test_speculation_streams_the_whole_accepted_run_in_order():
    """A step yields several tokens and the stream must show all of them: a step that
    emitted only its last token would finish with the right output but stream a
    truncated version of it."""
    with real_engine(dtype=torch.float32, num_speculative_tokens=4) as llm:
        updates = list(llm.generate_stream("The capital of France is",
                                           sampling_params=GREEDY, max_tokens=20))
        streamed = [update.token_id for update in updates]
        completion = llm.generate("The capital of France is",
                                  sampling_params=GREEDY, max_tokens=20)[0]

        assert streamed == list(completion.token_ids), "the stream and completion disagree"
        assert sum(update.finished for update in updates) == 1, "finished reported twice"
        assert_no_spec_leaks(llm)


@pytest.mark.oracle
def test_a_weak_draft_is_rejected_and_leaves_nothing_behind():
    """The case rollback exists for, driven hard: a 4-layer prefix of Qwen3-0.6B is a
    genuinely bad draft, rejected almost always. Every rejected proposal wrote KV in
    both pools and must give it back — and the output must still be correct, because
    rejection sampling does not care how bad the draft is."""
    prompt = "The capital of France is"

    with real_engine(dtype=torch.float32, num_speculative_tokens=4, num_draft_layers=4) as llm:
        weak = llm.generate(prompt, sampling_params=GREEDY, max_tokens=24)[0]
        stats = llm.spec.stats
        assert stats.acceptance_rate < 0.5, "a 4-layer draft should mostly be rejected"
        assert_no_spec_leaks(llm)

    with real_engine(dtype=torch.float32) as llm:
        plain = llm.generate(prompt, sampling_params=GREEDY, max_tokens=24)[0]

    assert weak.token_ids == plain.token_ids, (
        "a bad draft changed the output; rejection sampling must make draft quality "
        "irrelevant to correctness"
    )


@pytest.mark.oracle
def test_stochastic_speculation_stays_in_the_vocabulary_and_leaks_nothing():
    """The residual path end to end. Temperature sampling has no fixed answer, so what
    is checked is real tokens, the requested count, and balanced pools; the
    *distribution* is proven far more sharply by the chi-square test above."""
    warm = SamplingParams(temperature=0.9, top_p=0.95)
    with real_engine(num_speculative_tokens=3) as llm:
        completions = llm.generate(
            ["The capital of France is", "Once upon a time"],
            sampling_params=warm, max_tokens=20,
        )
        for completion in completions:
            assert len(completion.token_ids) <= 20
            assert all(0 <= token < llm.config.vocab_size for token in completion.token_ids)
        assert llm.spec.stats.proposed > 0, "nothing was speculated"
        assert_no_spec_leaks(llm)


@pytest.mark.oracle
def test_speculation_survives_chunked_prefill_beside_it():
    """A long prompt chunking through the batch while other sequences speculate: the
    two features touch the same budget from opposite ends (a chunk wants every token
    it can get; a speculative group is atomic) and must share an iteration without
    corrupting either."""
    long_prompt = "In a distant kingdom by the sea, the old chronicles record that " * 12
    with real_engine(num_speculative_tokens=4, chunk_size=64, max_batched_tokens=128) as llm:
        completions = llm.generate(
            [long_prompt, "The capital of France is", "Once upon a time"],
            sampling_params=GREEDY, max_tokens=16,
        )

        assert len(completions) == 3
        for completion in completions:
            assert completion.token_ids, "a sequence produced nothing"
        assert_no_spec_leaks(llm)


@pytest.mark.oracle
def test_speculation_survives_preemption():
    """A pool the requests almost exactly fill, so decodes crossing page boundaries
    force preemption. A preempted sequence discards its unverified proposals and
    recomputes only what it really committed; keeping the proposals would put tokens
    in the recomputed prefill the caller never received."""
    prompts = [f"Tell me about the number {index} and why it matters. " * 6
               for index in range(6)]
    with real_engine(num_blocks=20, num_speculative_tokens=3, max_sequences=6) as llm:
        completions = llm.generate(prompts, sampling_params=GREEDY, max_tokens=24)

        assert len(completions) == len(prompts)
        for completion in completions:
            assert completion.token_ids
        assert llm.stats.preemptions > 0, "the pool was not tight enough to preempt"
        assert_no_spec_leaks(llm)


@pytest.mark.oracle
def test_the_draft_pool_is_shallower_than_the_target_and_costs_less():
    """A self-draft shares weights, so the only memory it adds is its own KV — and its
    pool has the draft's layer count, which is why the arrangement fits at all."""
    with real_engine(num_speculative_tokens=2, num_draft_layers=4) as llm:
        target_pool, draft_pool = llm.manager.kv, llm.spec.proposer.manager.kv

        assert draft_pool.num_layers == 4
        assert target_pool.num_layers == llm.config.num_hidden_layers

        def per_page(pool) -> int:
            return PagedKvPool.bytes_for(pool.num_layers, 1, pool.block_size,
                                         pool.num_kv_heads, pool.head_dim, pool.kv_dtype)

        assert per_page(draft_pool) * 7 == per_page(target_pool), "4 layers vs 28"
        assert draft_pool.num_blocks > target_pool.num_blocks, (
            "the draft caches the same tokens plus proposals in flight"
        )
        assert llm.spec.proposer.model.blocks[0] is llm.model.blocks[0], (
            "the weights must be shared, not copied"
        )
