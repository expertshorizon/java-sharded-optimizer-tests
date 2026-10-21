"""Checkpoint/resume equivalence, swept across structurally meaningful split
points rather than a handful of arbitrarily chosen indices.

For every scenario used here, tests/oracle.interesting_split_points replays
the log once and records exactly which arrival indices sit on a real
boundary (right after a rejection, right after a poisoned commit, right
after a bucket reaches quorum while still blocked behind an earlier step,
plus the two trivial ends of the log). Splitting at any of those points,
running the head with --checkpoint-out and the tail with --checkpoint-in,
must reproduce exactly what one uninterrupted run computes, including the
checkpoint file's own JSON shape at that point.

Each sweep runs inside one test function rather than being parametrized
into one test per (scenario, split) pair: resume correctness is a single
requirement, and a repair that survives three splits out of forty has not
met it.
"""

import oracle
import scenarios
import support


def _sample(points, cap):
    if len(points) <= cap:
        return points
    step = (len(points) - 1) / (cap - 1)
    chosen = sorted({points[round(i * step)] for i in range(cap)})
    return chosen


_CHECKPOINT_SCENARIOS = [
    s for s in scenarios.SCENARIOS
    if len(oracle.interesting_split_points(s.config, s.events)["all"]) > 2
]

_BROAD_SPLIT_CASES = []
for _scenario in _CHECKPOINT_SCENARIOS:
    points = oracle.interesting_split_points(_scenario.config, _scenario.events)["all"]
    for _split in _sample(points, cap=3):
        _BROAD_SPLIT_CASES.append((_scenario, _split))


# A smaller, curated subset for the more expensive trace/rejection
# concatenation check below — every theme is still represented, just not the
# whole scenario library.
_DETAIL_TAGS = ("kitchen-sink", "cascade", "ordering", "poison", "quorum")
_DETAIL_SCENARIOS = [s for s in _CHECKPOINT_SCENARIOS if any(t in s.tags for t in _DETAIL_TAGS)][:8]

_DETAIL_SPLIT_CASES = []
for _scenario in _DETAIL_SCENARIOS:
    points = oracle.interesting_split_points(_scenario.config, _scenario.events)["all"]
    # cap=2 would pick only points[0] and points[-1], i.e. an empty head and an
    # empty tail, so neither test below would ever resume mid-log. cap=3 keeps
    # both ends and adds a genuine middle boundary.
    for _split in _sample(points, cap=3):
        _DETAIL_SPLIT_CASES.append((_scenario, _split))


def check_checkpoint_resume_equals_uninterrupted_replay():
    """Splitting at a structurally meaningful boundary and resuming from a
    checkpoint must reach exactly the same final_state as one uninterrupted
    run over the whole log, for every sampled boundary of every scenario
    that has one."""
    assert _BROAD_SPLIT_CASES, "no scenario offers a non-trivial split point"
    for scenario, split in _BROAD_SPLIT_CASES:
        resumed_out, _, _, _ = support.run_checkpoint_split(scenario.config, scenario.events, split)
        full_out = support.run_cli(scenario.config, scenario.events)
        assert resumed_out["final_state"] == full_out["final_state"], (
            f"{scenario.name}@{split}: resumed final_state diverged from the uninterrupted run"
        )


def check_checkpoint_resume_trace_and_rejections_reconstruct_exactly():
    """`trace` is not part of the checkpoint schema and is fresh per
    invocation, so the head run's own trace concatenated with the resumed
    run's own trace, in order, must equal the uninterrupted run's trace.
    `all_rejections` is the opposite — it is loaded from the checkpoint and
    continues to grow, so the resumed run's own `rejections` field is
    already the complete history and must equal the uninterrupted run's
    `rejections` field on its own, with no concatenation. Getting this
    asymmetry backwards (concatenating rejections, or forgetting to carry
    rejections across the checkpoint at all) is exactly the kind of
    plausible mistake this test is for."""
    for scenario, split in _DETAIL_SPLIT_CASES:
        resumed_out, head_out, _, _ = support.run_checkpoint_split(scenario.config, scenario.events, split)
        full_out = support.run_cli(scenario.config, scenario.events)
        combined_trace = head_out["trace"] + resumed_out["trace"]
        assert combined_trace == full_out["trace"], f"{scenario.name}@{split}: trace does not concatenate"
        assert resumed_out["rejections"] == full_out["rejections"], (
            f"{scenario.name}@{split}: rejection history did not survive the resume"
        )


def check_checkpoint_emitted_after_resume_matches_independent_oracle():
    """A checkpoint written by an invocation that itself loaded a checkpoint
    must be complete and canonical too. This specifically guards against
    implementations whose first-generation checkpoint is correct but whose
    save path drops pending/rejection state after CheckpointManager.load().
    """
    for scenario, split in _DETAIL_SPLIT_CASES:
        head = scenario.events[:split]
        tail = scenario.events[split:]

        # Derive both generations independently. Do not use the candidate's head
        # checkpoint as the oracle input, so a compensating pair of checkpoint
        # bugs cannot make the chained check self-consistent.
        _, expected_head_checkpoint = oracle.run_with_checkpoint(scenario.config, head)
        _, expected_resumed_checkpoint = oracle.run_with_checkpoint(
            scenario.config, tail, expected_head_checkpoint
        )

        _, _, _, resumed_checkpoint = support.run_checkpoint_split(
            scenario.config, scenario.events, split
        )
        assert resumed_checkpoint == expected_resumed_checkpoint, (
            f"{scenario.name}@{split}: second-generation checkpoint diverged from the oracle"
        )


def check_raw_checkpoint_file_shape_matches_independent_oracle_at_every_split():
    """The raw --checkpoint-out JSON itself (committed_step, params,
    momentum, the full pending buffer, rejections-by-step, and
    all_rejections) must match what an independent replay computes at that
    exact point — not only the CLI's final --out after a later resume.

    This sweeps every arrival index of every scenario, not a sampled subset,
    because the checkpoint's pending buffer is the one place where a value's
    textual form is decided separately from its arithmetic value. An
    implementation that keeps each gradient's original input lexeme for
    persistence while computing on parsed rationals produces exactly correct
    output, traces and resumed state, and only reveals itself in a pending
    entry — for instance writing a `1/-2` or `2/10` input lexeme straight
    through where the contract requires a reduced, positive-denominator
    rational. Those buckets are open at ordinary mid-log indices that no
    structural-boundary sampler proposes, so anything less than a full sweep
    leaves the requirement unchecked for most of the library."""
    for scenario in scenarios.SCENARIOS:
        for split in range(len(scenario.events) + 1):
            _, _, head_checkpoint, _ = support.run_checkpoint_split(
                scenario.config, scenario.events, split
            )
            _, expected_checkpoint = oracle.run_with_checkpoint(
                scenario.config, scenario.events[:split]
            )
            assert head_checkpoint == expected_checkpoint, (
                f"{scenario.name}@{split}: checkpoint file shape diverged from the oracle"
            )


def check_checkpoint_preserves_stray_rejection_for_already_committed_step():
    """A late rejection against a step number that has already committed
    has no effect on any future output, but the checkpoint contract requires
    the checkpoint's rejections-by-step map to still contain it byte for
    byte, since nothing is filtered out of that map on save. This is the
    scenario validated directly against the real reference implementation
    during task construction."""
    scenario = scenarios.by_name("stray_late_rejection_after_commit")
    split = 3  # right after the second late rejection, before step1 commits
    _, _, head_checkpoint, _ = support.run_checkpoint_split(scenario.config, scenario.events, split)
    assert head_checkpoint["rejections"]["0"] == [
        {"worker_id": 1, "step": 0, "reason": "late"},
        {"worker_id": 2, "step": 0, "reason": "late"},
    ]


def check_checkpoint_preserves_rejections_for_blocked_future_step():
    """Rejections against a step whose bucket is already full but which is
    still blocked behind an earlier, uncommitted frontier step must appear
    in the checkpoint's rejections-by-step map exactly like rejections
    against any other step. Serializing only the current frontier's
    per-step rejections (and quietly dropping entries for blocked-but-full
    future buckets) would still reproduce the correct final state and flat
    all_rejections log, so this has to be checked against the raw
    checkpoint shape directly rather than end-to-end equivalence."""
    scenario = scenarios.by_name("duplicate_outranks_superfluous_on_full_bucket")
    split = 4  # right after the superfluous rejection, before step0 commits
    _, _, head_checkpoint, _ = support.run_checkpoint_split(scenario.config, scenario.events, split)
    assert head_checkpoint["committed_step"] == -1
    assert head_checkpoint["rejections"]["3"] == [
        {"worker_id": 0, "step": 3, "reason": "duplicate"},
        {"worker_id": 2, "step": 3, "reason": "superfluous"},
    ]


def check_checkpoint_captures_pending_raw_nonfinite_poison_before_quorum():
    """A bucket's first accepted contribution can be poisoned by a raw
    non-finite gradient token even though the event's own `poisoned` JSON
    field said false; the checkpoint must record that as an ordinary
    poisoned pending entry (`poisoned: true`, `gradient: null`), the same
    shape an explicitly-poisoned contribution gets. An implementation that
    derives the poison decision on demand from the raw tokens at commit
    time, but checkpoints the untouched, originally-false `poisoned` field
    for a contribution still sitting in a pending bucket, would pass every
    other poisoning test while silently losing the poison across a
    resume."""
    scenario = scenarios.by_name("raw_plus_inf_token_poisons_isolated_step")
    split = 1  # right after the sole, not-yet-quorate raw-token contribution
    resumed_out, _, head_checkpoint, _ = support.run_checkpoint_split(
        scenario.config, scenario.events, split
    )
    full_out = support.run_cli(scenario.config, scenario.events)

    assert head_checkpoint["committed_step"] == -1
    assert head_checkpoint["pending"]["0"] == [
        {"worker_id": 0, "step": 0, "poisoned": True, "gradient": None}
    ]
    assert resumed_out["final_state"] == full_out["final_state"]
    assert resumed_out["trace"] == [
        {
            "step": 0,
            "status": "poisoned_skip",
            "workers": [0, 1],
            "l1_norm": None,
            "clip_scale": None,
            "rejections": [],
        }
    ]


def check_kitchen_sink_scenarios_survive_every_meaningful_checkpoint_split():
    """Each multi-responsibility scenario (out-of-order arrival + staleness
    + cross-shard clipping + poisoning + rational arithmetic all in one
    trajectory) must still resume correctly no matter where it is split —
    a bug that only shows up when several of these interact should be
    caught here even if it slips past the single-responsibility scenarios."""
    kitchen_sink = scenarios.with_tag("kitchen-sink")
    assert kitchen_sink, "kitchen-sink scenarios disappeared from the library"
    for scenario in kitchen_sink:
        points = oracle.interesting_split_points(scenario.config, scenario.events)["all"]
        full_out = support.run_cli(scenario.config, scenario.events)
        for split in points:
            resumed_out, head_out, _, _ = support.run_checkpoint_split(scenario.config, scenario.events, split)
            assert resumed_out["final_state"] == full_out["final_state"], f"{scenario.name}@{split}"
            assert head_out["trace"] + resumed_out["trace"] == full_out["trace"], f"{scenario.name}@{split}"
            assert resumed_out["rejections"] == full_out["rejections"], f"{scenario.name}@{split}"


def _drain_config():
    return scenarios.cfg(
        num_workers=4,
        param_size=3,
        num_shards=2,
        quorum_size=2,
        max_staleness=1,
        clip_threshold="+10",
        learning_rate="+1/10",
        momentum_beta="1/+2",
    )


def check_checkpoint_with_quorate_frontier_drains_before_tail_events():
    """A valid inbound checkpoint may be captured after quorum admission but
    before the resolver cascade. On resume, the already-quorate frontier must
    drain immediately, including poisoned skips, before any tail event is
    classified against the new frontier."""
    config = _drain_config()
    checkpoint = {
        "committed_step": -1,
        "params": ["0/1", "0/1", "0/1"],
        "momentum": ["0/1", "0/1", "0/1"],
        "pending": {
            "0": [
                {"worker_id": 0, "step": 0, "poisoned": False,
                 "gradient": ["1/1", "2/1", "3/1"]},
                {"worker_id": 1, "step": 0, "poisoned": True,
                 "gradient": None},
            ],
            "1": [
                {"worker_id": 2, "step": 1, "poisoned": False,
                 "gradient": ["2/1", "2/1", "2/1"]},
                {"worker_id": 3, "step": 1, "poisoned": False,
                 "gradient": ["4/1", "4/1", "4/1"]},
            ],
        },
        "rejections": {
            "0": [{"worker_id": 3, "step": 0, "reason": "duplicate"}],
        },
        "all_rejections": [
            {"worker_id": 3, "step": 0, "reason": "duplicate"},
        ],
    }

    expected = oracle.run(config, [], checkpoint)
    actual = support.run_cli(config, [], checkpoint_in=checkpoint)
    assert actual == expected
    assert [entry["step"] for entry in actual["trace"]] == [0, 1]
    assert [entry["status"] for entry in actual["trace"]] == ["poisoned_skip", "committed"]
    assert actual["rejections"] == checkpoint["all_rejections"]


def check_checkpoint_drain_happens_before_first_tail_event_is_classified():
    """Checkpoint draining is an ordering rule, not merely an eventual-work
    requirement. A quorate frontier in the inbound checkpoint must commit
    before the first tail event is classified. If an implementation postpones
    drain until after the event loop, this event is incorrectly classified as
    superfluous instead of late.
    """
    config = _drain_config()
    checkpoint = {
        "committed_step": -1,
        "params": ["0/1", "0/1", "0/1"],
        "momentum": ["0/1", "0/1", "0/1"],
        "pending": {
            "0": [
                {"worker_id": 0, "step": 0, "poisoned": False,
                 "gradient": ["1/1", "1/1", "1/1"]},
                {"worker_id": 1, "step": 0, "poisoned": False,
                 "gradient": ["1/1", "1/1", "1/1"]},
            ],
        },
        "rejections": {},
        "all_rejections": [],
    }
    tail = [
        {"worker_id": 2, "step": 0, "gradient": ["9/1", "9/1", "9/1"]},
    ]

    expected = oracle.run(config, tail, checkpoint)
    actual = support.run_cli(config, tail, checkpoint_in=checkpoint)

    assert actual == expected
    assert actual["trace"][0]["step"] == 0
    assert actual["rejections"][-1] == {
        "worker_id": 2, "step": 0, "reason": "late"
    }


def check_checkpoint_input_accepts_every_documented_numeric_string_form():
    """The accepted numeric-string syntax (decimal, scientific notation, or
    `p/q` with a sign on either component) is the same wherever a number is
    read, including the params and momentum of a checkpoint handed to
    `--checkpoint-in`. Nothing in the contract restricts an inbound
    checkpoint to output-canonical spellings, and nothing restricts where it
    came from, so a hand-written or externally generated checkpoint carrying
    `0.5`, `2.5e-1`, or `1/-4` must load to exactly those rational values
    and be re-emitted canonically. Parsing checkpoint numbers through a
    fraction-only path instead of the general numeric parser fails here."""
    config = scenarios.cfg(
        num_workers=2,
        param_size=4,
        num_shards=2,
        quorum_size=2,
        max_staleness=1,
        clip_threshold="10",
        learning_rate="1/10",
        momentum_beta="1/2",
    )
    checkpoint = {
        "committed_step": -1,
        "params": ["+0.5", "2.5e-1", "+1/-4", "-3"],
        "momentum": ["-0.125", "3/+6", "1e+2", "0/1"],
        "pending": {},
        "rejections": {},
        "all_rejections": [],
    }

    expected_out, expected_checkpoint = oracle.run_with_checkpoint(config, [], checkpoint)
    actual_out, actual_checkpoint = support.run_cli_with_checkpoint(config, [], checkpoint_in=checkpoint)

    assert actual_out == expected_out
    assert actual_checkpoint == expected_checkpoint
    flattened = [
        value
        for shard in actual_out["final_state"]["shards"]
        for value in shard["params"]
    ]
    assert flattened == ["1/2", "1/4", "-1/4", "-3/1"]


def check_checkpoint_input_numeric_forms_inside_pending_gradients_replay_exactly():
    """The same accepted syntax applies to a pending contribution's gradient
    components inside an inbound checkpoint. Here the buffered contribution
    for the frontier step is spelled with a decimal, a scientific-notation
    value, and a signed denominator; the tail event completes the quorum, so
    those values have to be averaged, clipped and applied at their exact
    rational values, and every rational the run writes back out has to be
    canonical."""
    config = scenarios.cfg(
        num_workers=2,
        param_size=3,
        num_shards=1,
        quorum_size=2,
        max_staleness=1,
        clip_threshold="+7/2",
        learning_rate="1/+5",
        momentum_beta="+1/3",
    )
    checkpoint = {
        "committed_step": -1,
        "params": ["0/1", "0/1", "0/1"],
        "momentum": ["0/1", "0/1", "0/1"],
        "pending": {
            "0": [
                {"worker_id": 0, "step": 0, "poisoned": False,
                 "gradient": ["+1.5", "2.5e-1", "+3/-4"]},
            ],
        },
        "rejections": {},
        "all_rejections": [],
    }
    tail = [
        {"worker_id": 1, "step": 0, "poisoned": False, "gradient": ["+1/2", "-1/+4", "5/4"]},
    ]

    expected_out, expected_checkpoint = oracle.run_with_checkpoint(config, tail, checkpoint)
    actual_out, actual_checkpoint = support.run_cli_with_checkpoint(config, tail, checkpoint_in=checkpoint)

    assert actual_out == expected_out
    assert actual_checkpoint == expected_checkpoint
    assert [entry["status"] for entry in actual_out["trace"]] == ["committed"]
    for value in support.all_numeric_values(actual_out):
        support.assert_canonical_fraction(value)


def check_checkpoint_pending_poison_is_normalized_on_load():
    """Inbound pending entries obey the same poison normalization as log events.
    Raw non-finite tokens poison the contribution, and an explicit poisoned flag
    always forces gradient=null even if a non-normalized checkpoint supplied values.
    """
    config = scenarios.cfg(
        num_workers=2, param_size=2, num_shards=1, quorum_size=2, max_staleness=1,
        clip_threshold="10", learning_rate="1/10", momentum_beta="1/2",
    )

    cases = [
        {"worker_id": 0, "step": 0, "poisoned": False, "gradient": ["+1/2", "Inf"]},
        {"worker_id": 0, "step": 0, "poisoned": True, "gradient": ["+1/2", "3/+4"]},
    ]
    tail = [{"worker_id": 1, "step": 0, "poisoned": False, "gradient": ["1/2", "1/4"]}]

    for pending_event in cases:
        checkpoint = {
            "committed_step": -1,
            "params": ["0/1", "0/1"],
            "momentum": ["0/1", "0/1"],
            "pending": {"0": [pending_event]},
            "rejections": {},
            "all_rejections": [],
        }
        expected_out, expected_checkpoint = oracle.run_with_checkpoint(config, tail, checkpoint)
        actual_out, actual_checkpoint = support.run_cli_with_checkpoint(
            config, tail, checkpoint_in=checkpoint
        )

        assert actual_out == expected_out
        assert actual_checkpoint == expected_checkpoint
        assert actual_out["trace"][0]["status"] == "poisoned_skip"
        assert actual_checkpoint["pending"] == {}
