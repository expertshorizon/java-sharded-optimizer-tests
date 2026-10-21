"""Regression layer: the original six hand-captured golden fixtures.

These are the earliest, most literal checks in the suite — a full run over
each bundled scenario must reproduce a byte-for-byte golden output captured
once from the reference implementation. They are kept verbatim (rather than
routed through tests/oracle.py) so this file remains a golden-file anchor
that is independent of the oracle's own correctness. checks_scenarios.py
and friends are where almost all of the new, oracle-driven coverage lives.

Nothing in this file above the five grouped tests at the bottom is collected
by pytest: each `check_*` function is a helper those five call. Every check
walks its whole fixture set in-function rather than being parametrized per
fixture, so a repair that satisfies `basic` but breaks `stress` fails the
same test rather than collecting partial credit.
"""

import json
from fractions import Fraction
from pathlib import Path

import oracle
import scenarios
import support

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_NAMES = ["basic", "cascade", "staleness", "clip", "poison", "stress"]


def check_project_compiles():
    """javac must build the copied /app/src tree into /work/out with no errors."""
    assert support.COMPILE_RESULT.returncode == 0, support.COMPILE_RESULT.stderr


def check_agent_code_runs_without_access_to_the_verifier():
    """The verifier really does execute agent-produced code: it compiles /app/src
    and runs the resulting CLI as a subprocess. That process therefore has to run
    as a uid that owns neither the reward channel nor the expectations, or a
    malicious build could read the goldens or overwrite /logs/verifier/reward.txt,
    including from something double-forked to outlive the pytest session. This
    probes what that process can actually reach, from a process launched exactly
    the way the compiled CLI is, rather than trusting the image build to have got
    the permissions right.
    """
    assert support.privilege_dropping_available(), (
        "the verifier session cannot switch users, so agent code would run with the "
        "same privileges as the verifier itself"
    )

    whoami = support.run_probe("id -u")
    assert whoami.returncode == 0
    assert whoami.stdout.strip() == str(support.GRADED_UID), (
        f"agent code runs as uid {whoami.stdout.strip()}, not the unprivileged {support.GRADED_UID}"
    )

    for command in (
        "cat /logs/verifier/reward.txt",
        "cat /logs/verifier/ctrf.json",
        "ls -a /logs/verifier",
        "echo 1 > /logs/verifier/reward.txt",
        "echo 1 > /logs/verifier/planted.txt",
    ):
        probe = support.run_probe(command)
        assert probe.returncode != 0, f"agent code could reach the reward channel: {command}"
    assert not Path("/logs/verifier/planted.txt").exists()

    for command in (
        "cat /tests/support.py",
        "cat /tests/oracle.py",
        "ls -a /tests",
        "cat /tests/fixtures/golden_basic.json",
    ):
        probe = support.run_probe(command)
        assert probe.returncode != 0, f"agent code could read the expectations: {command}"

    # Relative paths too, in case the working directory was inherited.
    for command in ("cat ./support.py", "cat ./fixtures/golden_basic.json", "ls -a ."):
        probe = support.run_probe(command)
        assert probe.returncode != 0 or "golden" not in probe.stdout, (
            f"agent code reached the expectations through its working directory: {command}"
        )


def golden(name):
    return support.load_json(FIXTURES / f"golden_{name}.json")


def config_of(name):
    return support.load_json(FIXTURES / f"config_{name}.json")


def events_of(name):
    return support.load_json(FIXTURES / f"log_{name}.json")


def run_full(name):
    return support.run_cli(config_of(name), events_of(name))


def check_final_output_matches_golden_exactly():
    """A full run over every bundled scenario must exactly reproduce the
    reference final state, trace, and rejection log."""
    for name in FIXTURE_NAMES:
        assert run_full(name) == golden(name), f"golden mismatch for {name}"


def check_emitted_rationals_and_shard_partitions_are_well_formed():
    """Two structural invariants over the same six runs: every emitted
    rational is an integer over a strictly positive, lowest-terms
    denominator rather than a malformed or float-derived string, and each
    fixture's shards partition its configured parameter vector with equal
    params/momentum lengths, no gap, and no overlap."""
    for name in FIXTURE_NAMES:
        actual = run_full(name)
        values = support.all_numeric_values(actual)
        assert values, f"expected at least one numeric value to check in {name}"
        for value in values:
            support.assert_canonical_fraction(value)
            numerator, denominator = value.split("/")
            assert Fraction(int(numerator), int(denominator)) == Fraction(numerator + "/" + denominator), (
                f"{name}: {value} does not round-trip as an exact rational"
            )

        config = config_of(name)
        shards = actual["final_state"]["shards"]
        assert len(shards) == config["num_shards"], f"{name}: wrong shard count"
        total = 0
        for shard in shards:
            assert len(shard["params"]) == len(shard["momentum"]), f"{name}: shard length mismatch"
            total += len(shard["params"])
        assert total == config["param_size"], f"{name}: shards do not cover param_size"


def check_clip_scale_matches_threshold_over_norm_formula():
    """clip_scale on a committed step must equal min(1, clip_threshold /
    l1_norm), recomputed independently from the config, for every committed
    step of every fixture."""
    for name in FIXTURE_NAMES:
        threshold = Fraction(config_of(name)["clip_threshold"])
        committed = [entry for entry in run_full(name)["trace"] if entry["status"] == "committed"]
        for entry in committed:
            l1_norm = Fraction(entry["l1_norm"])
            scale = Fraction(entry["clip_scale"])
            expected = Fraction(1) if l1_norm == 0 else min(Fraction(1), threshold / l1_norm)
            assert scale == expected, f"{name}: step {entry['step']} clip_scale {scale} != {expected}"


def check_cross_shard_clip_uses_global_norm_not_per_shard():
    """Clipping must scale by the L1 norm of the full gradient vector, not by any single shard's own local norm."""
    actual = run_full("clip")
    entry = actual["trace"][0]
    assert entry["l1_norm"] == "12/1"
    assert entry["clip_scale"] == "2/3"
    for shard in actual["final_state"]["shards"]:
        assert shard["params"] == ["-1/5", "-1/5"]
        assert shard["momentum"] == ["2/1", "2/1"]


def check_staleness_boundary_accepts_exactly_at_window_edge():
    """A submission whose lookahead equals max_staleness must be admitted; one step further must be rejected as stale."""
    actual = run_full("staleness")
    assert actual["uncommitted_pending"] == {"3": [0]}
    assert actual["rejections"] == [{"worker_id": 1, "step": 4, "reason": "stale"}]


def check_rejection_reasons_and_poison_are_attributed_correctly():
    """All four rejection reasons must be attributed to the correct worker
    and step in the cascade scenario, and a poisoned gradient must void its
    whole step even when it is not the first of the quorum to arrive,
    leaving optimizer state untouched."""
    reasons = {(r["worker_id"], r["step"]): r["reason"] for r in run_full("cascade")["rejections"]}
    assert reasons[(0, 3)] == "duplicate"
    assert reasons[(2, 4)] == "superfluous"
    assert reasons[(2, 6)] == "stale"
    assert reasons[(2, 1)] == "late"

    poisoned = run_full("poison")
    assert poisoned["trace"][0]["status"] == "poisoned_skip"
    for shard in poisoned["final_state"]["shards"]:
        assert all(value == "0/1" for value in shard["params"])
        assert all(value == "0/1" for value in shard["momentum"])


def check_commit_order_is_gap_free_and_trailing_pending_is_reported():
    """Trace entries must appear in strictly increasing, gap-free step order
    regardless of arrival order in the log, and steps that never reach
    quorum before the log ends must be reported as pending rather than
    silently dropped or force-committed."""
    for name in ("cascade", "stress"):
        steps = [entry["step"] for entry in run_full(name)["trace"]]
        assert steps == sorted(steps), f"{name}: trace is out of order"
        assert steps == list(range(steps[0], steps[0] + len(steps))), f"{name}: trace has a gap"

    stress = run_full("stress")
    assert stress["uncommitted_pending"] == {"7": [2], "8": [0]}
    assert len(stress["trace"]) == 6


def check_checkpoint_resume_over_golden_logs_reaches_the_same_final_state():
    """Splitting either bundled multi-step log at any of several points
    through a checkpoint and resuming must reach the same final parameter
    and momentum state as one uninterrupted run. The stress split at 12 is
    additionally checked for the pending buffer: two of the three
    submissions already buffered for step 3 when the checkpoint is taken
    are never resubmitted in the tail, so dropping that buffer leaves step
    3 — and everything cascading from it — permanently uncommittable."""
    for name, splits in (("cascade", [1, 3, 6, 9, 11, 13]), ("stress", [2, 5, 8, 9, 12, 16, 20])):
        config = config_of(name)
        events = events_of(name)
        for split in splits:
            resumed, _, _, _ = support.run_checkpoint_split(config, events, split)
            assert resumed["final_state"] == golden(name)["final_state"], (
                f"{name}: resume from split {split} diverged from the uninterrupted run"
            )

    resumed, _, _, _ = support.run_checkpoint_split(config_of("stress"), events_of("stress"), 12)
    committed_steps = {entry["step"] for entry in resumed["trace"]}
    assert {3, 4, 5}.issubset(committed_steps), "pending buffer was dropped across the checkpoint"


# ---------------------------------------------------------------------------
# Clause-by-clause checks
#
# Each test below names the instruction.md clause it pins and asserts the one
# value that clause decides, alongside full equality against tests/oracle.py.
# The oracle equality catches any drift in the rest of the artifact; the
# literal assertion is what makes the test fail for the specific reason it
# exists, rather than because something unrelated moved.
# ---------------------------------------------------------------------------

ZERO_STATE_2 = {
    "committed_step": -1,
    "params": ["0/1", "0/1"],
    "momentum": ["0/1", "0/1"],
    "rejections": {},
    "all_rejections": [],
}

POISON_TOKENS = ["NaN", "Inf", "+Inf", "-Inf"]


def small_cfg(**overrides):
    base = dict(
        num_workers=2, param_size=2, num_shards=1, quorum_size=2,
        max_staleness=1, clip_threshold="10", learning_rate="1/10",
        momentum_beta="1/2",
    )
    base.update(overrides)
    return scenarios.cfg(**base)


def checkpoint_with_pending(pending_event, params=None, momentum=None, committed_step=-1):
    checkpoint = dict(ZERO_STATE_2)
    checkpoint["committed_step"] = committed_step
    if params is not None:
        checkpoint["params"] = params
    if momentum is not None:
        checkpoint["momentum"] = momentum
    checkpoint["pending"] = {str(pending_event["step"]): [pending_event]}
    return checkpoint


def check_every_poison_token_is_detected_at_every_position_on_both_input_paths():
    """instruction.md: raw `NaN`, `Inf`, `+Inf` or `-Inf` gradient tokens
    poison a submission, and on `--checkpoint-in` such a token in a pending
    entry yields `poisoned: true` with `gradient: null`.

    The contract says nothing about where in the gradient array the token sits
    or which input carried it, so this sweeps all four spellings against all
    three positions of a three-component gradient, on both paths that parse a
    gradient: an ordinary `--log` event, and a pending entry loaded from
    `--checkpoint-in`. Both matter separately, because the two are distinct
    deserializers in the shipped code -- a submission that reuses only part of
    its token set in one of them, inspects only `gradient[0]`, or stops at the
    first component that parses as finite, passes on one path and one position
    and fails here.

    Either way the step must resolve as `poisoned_skip` with no norm or scale
    reported, its bucket must be gone rather than carried forward as
    unfinished work, and optimizer state must be untouched.
    """
    config = small_cfg(param_size=3, num_shards=2)
    finite = ["1/2", "-3/4", "5/8"]
    zeros = ["0/1", "0/1", "0/1"]
    partner = {"worker_id": 1, "step": 0, "poisoned": False, "gradient": ["1/2", "1/4", "1/8"]}

    for position in range(3):
        for token in POISON_TOKENS:
            gradient = list(finite)
            gradient[position] = token
            poisoned_event = {
                "worker_id": 0, "step": 0, "poisoned": False, "gradient": gradient,
            }
            label = f"{token} at gradient index {position}"

            # Path 1: the token arrives in an ordinary --log event.
            log_events = [poisoned_event, partner]
            expected_out, expected_checkpoint = oracle.run_with_checkpoint(config, log_events)
            actual_out, actual_checkpoint = support.run_cli_with_checkpoint(config, log_events)
            assert actual_out == expected_out, f"log {label}: output diverged from the oracle"
            assert actual_checkpoint == expected_checkpoint, f"log {label}: checkpoint diverged"
            assert_step_zero_was_poisoned(actual_out, actual_checkpoint, zeros, f"log {label}")

            # Path 2: the same token arrives as a checkpoint pending entry,
            # completed by a tail event.
            checkpoint = dict(ZERO_STATE_2)
            checkpoint["params"] = zeros
            checkpoint["momentum"] = zeros
            checkpoint["pending"] = {"0": [poisoned_event]}
            expected_out, expected_checkpoint = oracle.run_with_checkpoint(
                config, [partner], checkpoint
            )
            actual_out, actual_checkpoint = support.run_cli_with_checkpoint(
                config, [partner], checkpoint_in=checkpoint
            )
            assert actual_out == expected_out, f"checkpoint {label}: output diverged"
            assert actual_checkpoint == expected_checkpoint, f"checkpoint {label}: diverged"
            assert_step_zero_was_poisoned(
                actual_out, actual_checkpoint, zeros, f"checkpoint {label}"
            )


def assert_step_zero_was_poisoned(output, checkpoint, zeros, label):
    entry = output["trace"][0]
    assert entry["step"] == 0
    assert entry["status"] == "poisoned_skip", f"{label} was not detected as poison"
    assert entry["l1_norm"] is None, f"{label}: a skipped step reported an l1_norm"
    assert entry["clip_scale"] is None, f"{label}: a skipped step reported a clip_scale"
    assert output["uncommitted_pending"] == {}, f"{label}: resolved step left a pending bucket"
    assert checkpoint["pending"] == {}, f"{label}: resolved step left a checkpoint bucket"
    assert checkpoint["committed_step"] == 0, f"{label}: the frontier did not advance"
    assert checkpoint["params"] == zeros and checkpoint["momentum"] == zeros, (
        f"{label}: a skipped step still moved optimizer state"
    )
    for shard in output["final_state"]["shards"]:
        assert all(value == "0/1" for value in shard["params"]), f"{label}: params moved"
        assert all(value == "0/1" for value in shard["momentum"]), f"{label}: momentum moved"


def check_checkpoint_explicit_poison_flag_discards_any_supplied_gradient():
    """instruction.md: an explicit `poisoned: true` yields `gradient: null`.
    A non-normalized checkpoint may still carry finite values alongside the
    flag; loading must drop them rather than average them into the step."""
    config = small_cfg()
    tail = [{"worker_id": 1, "step": 0, "poisoned": False, "gradient": ["1/2", "1/4"]}]

    for supplied in (["+1/2", "3/+4"], ["0/1", "0/1"], None):
        checkpoint = checkpoint_with_pending(
            {"worker_id": 0, "step": 0, "poisoned": True, "gradient": supplied}
        )
        expected_out, expected_checkpoint = oracle.run_with_checkpoint(config, tail, checkpoint)
        actual_out, actual_checkpoint = support.run_cli_with_checkpoint(
            config, tail, checkpoint_in=checkpoint
        )

        assert actual_out == expected_out, f"gradient={supplied}: output diverged"
        assert actual_checkpoint == expected_checkpoint, f"gradient={supplied}: checkpoint diverged"
        assert actual_out["trace"][0]["status"] == "poisoned_skip"
        for shard in actual_out["final_state"]["shards"]:
            assert shard["params"] == ["0/1", "0/1"]

    # The cases above all reach quorum, so the bucket is gone by the time
    # --checkpoint-out is written and the normalized entry is never visible in
    # either artifact. Re-run one of them with nothing to complete the quorum:
    # the entry is still pending at the end, so the checkpoint has to show what
    # loading actually made of it, and a loader that kept the supplied values
    # alongside the flag writes them straight back out here.
    checkpoint = checkpoint_with_pending(
        {"worker_id": 0, "step": 0, "poisoned": True, "gradient": ["+1/2", "3/+4"]}
    )
    expected_out, expected_checkpoint = oracle.run_with_checkpoint(config, [], checkpoint)
    actual_out, actual_checkpoint = support.run_cli_with_checkpoint(
        config, [], checkpoint_in=checkpoint
    )
    assert actual_out == expected_out
    assert actual_checkpoint == expected_checkpoint
    still_pending = actual_checkpoint["pending"]["0"]
    assert len(still_pending) == 1
    assert still_pending[0]["poisoned"] is True
    assert still_pending[0]["gradient"] is None, (
        "an explicit poisoned:true entry must load as gradient:null, not carry "
        "the values a non-normalized checkpoint supplied"
    )


def check_poison_token_recognition_does_not_move_with_the_jvm_default_locale():
    """instruction.md: results must be deterministic. Poison detection
    therefore cannot depend on ambient JVM state. Case-folding a token with
    the default locale makes `Inf` fold to a dotless `ınf` under a Turkish
    locale, where it matches nothing and reaches the numeric parser instead;
    this runs the same log under three default locales and requires
    byte-identical output from all of them."""
    config = small_cfg(num_workers=3, quorum_size=2, max_staleness=0)
    events = [
        scenarios.sub(0, 0, "1/2", "1/4"),
        {"worker_id": 1, "step": 0, "poisoned": False, "gradient": ["Inf", "3"]},
        {"worker_id": 0, "step": 1, "poisoned": False, "gradient": ["NaN", "1"]},
        {"worker_id": 1, "step": 1, "poisoned": False, "gradient": ["1/2", "1/2"]},
        {"worker_id": 0, "step": 2, "poisoned": False, "gradient": ["+Inf", "1"]},
        {"worker_id": 1, "step": 2, "poisoned": False, "gradient": ["-Inf", "1"]},
    ]

    baseline = support.run_cli_raw_text_uncached(config, events)
    for locale_args in (
        ("-Duser.language=tr", "-Duser.country=TR"),
        ("-Duser.language=lt", "-Duser.country=LT"),
    ):
        other = support.run_cli_raw_text_uncached(config, events, jvm_args=locale_args)
        assert other == baseline, (
            f"output changed under {' '.join(locale_args)} -- poison detection or numeric "
            "parsing depends on the ambient default locale"
        )

    parsed = json.loads(baseline)
    assert [entry["status"] for entry in parsed["trace"]] == [
        "poisoned_skip", "poisoned_skip", "poisoned_skip"
    ]


def check_staleness_window_admits_exactly_up_to_max_staleness():
    """instruction.md: the frontier is `committed_step + 1`; lookahead equal
    to `max_staleness` is admissible and only larger lookahead is stale, while
    a target at or below `committed_step` is late. Swept over several window
    sizes, including the degenerate `max_staleness = 0` where only the
    frontier itself is admissible."""
    for max_staleness in (0, 1, 2, 3):
        config = small_cfg(num_workers=2, max_staleness=max_staleness)
        # One lone submission per candidate target step: nothing reaches
        # quorum, so every accepted target shows up in uncommitted_pending
        # and every rejected one in rejections.
        targets = list(range(0, max_staleness + 3))
        events = [scenarios.sub(0, step, "1/2", "1/4") for step in targets]

        expected = oracle.run(config, events)
        actual = support.run_cli(config, events)
        assert actual == expected, f"max_staleness={max_staleness}: output diverged from the oracle"

        admitted = sorted(int(step) for step in actual["uncommitted_pending"])
        stale = sorted(r["step"] for r in actual["rejections"] if r["reason"] == "stale")
        assert admitted == [s for s in targets if s <= max_staleness], (
            f"max_staleness={max_staleness}: admitted {admitted}"
        )
        assert stale == [s for s in targets if s > max_staleness], (
            f"max_staleness={max_staleness}: stale {stale}"
        )


def check_first_accepted_contribution_per_worker_is_the_one_that_counts():
    """instruction.md: within a step, retain the first accepted contribution
    per worker; repeats are `duplicate`. The repeat here carries a different
    gradient, so a submission that overwrites the bucket entry (rather than
    rejecting the repeat) still reaches quorum but commits the wrong numbers."""
    config = small_cfg(num_workers=2, quorum_size=2, momentum_beta="0", clip_threshold="1000")
    events = [
        scenarios.sub(0, 0, "1", "0"),
        scenarios.sub(0, 0, "9", "9"),          # duplicate: must not replace ["1", "0"]
        scenarios.sub(1, 0, "3", "0"),
    ]

    expected = oracle.run(config, events)
    actual = support.run_cli(config, events)
    assert actual == expected

    assert actual["rejections"] == [{"worker_id": 0, "step": 0, "reason": "duplicate"}]
    entry = actual["trace"][0]
    assert entry["status"] == "committed"
    assert entry["workers"] == [0, 1]
    # average of [1,0] and [3,0] is [2,0]; below the threshold, so scale 1.
    assert entry["l1_norm"] == "2/1"
    assert entry["clip_scale"] == "1/1"
    shard = actual["final_state"]["shards"][0]
    assert shard["momentum"] == ["2/1", "0/1"]
    assert shard["params"] == ["-1/5", "0/1"]


def check_clip_is_applied_once_to_the_averaged_gradient():
    """instruction.md: average the quorum, then clip that averaged gradient
    once using the L1 norm of the averaged vector. Clipping each contribution
    before averaging would scale worker 0's [4, 0] by 1/4 and leave the
    average at [1/2, 0]; clipping the average scales [2, 0] by 1/2 and lands
    on [1, 0]."""
    config = small_cfg(
        num_workers=2, quorum_size=2, max_staleness=0,
        clip_threshold="1", learning_rate="1", momentum_beta="0",
    )
    events = [scenarios.sub(0, 0, "4", "0"), scenarios.sub(1, 0, "0", "0")]

    expected = oracle.run(config, events)
    actual = support.run_cli(config, events)
    assert actual == expected

    entry = actual["trace"][0]
    assert entry["l1_norm"] == "2/1", "l1_norm must be the norm of the averaged vector"
    assert entry["clip_scale"] == "1/2"
    shard = actual["final_state"]["shards"][0]
    assert shard["momentum"] == ["1/1", "0/1"]
    assert shard["params"] == ["-1/1", "0/1"]


def check_poison_at_any_quorum_position_voids_the_step_and_advances_the_frontier():
    """instruction.md: if any accepted contribution in a quorum is poisoned,
    skip the optimizer update but still advance the frontier. Swept over every
    arrival position in a three-worker quorum, since an implementation that
    inspects only the first (or last) chosen contribution passes one position
    and fails the others; the following step must then commit, proving the
    frontier moved rather than stalling."""
    config = small_cfg(num_workers=3, quorum_size=3, max_staleness=1, momentum_beta="0")

    for position in range(3):
        events = []
        for worker in range(3):
            if worker == position:
                events.append(
                    {"worker_id": worker, "step": 0, "poisoned": False, "gradient": ["NaN", "1/2"]}
                )
            else:
                events.append(scenarios.sub(worker, 0, "1/2", "1/2"))
        events += [scenarios.sub(worker, 1, "1", "0") for worker in range(3)]

        expected = oracle.run(config, events)
        actual = support.run_cli(config, events)
        assert actual == expected, f"poison at arrival position {position}: output diverged"

        statuses = [(entry["step"], entry["status"]) for entry in actual["trace"]]
        assert statuses == [(0, "poisoned_skip"), (1, "committed")], (
            f"poison at arrival position {position}: {statuses}"
        )
        shard = actual["final_state"]["shards"][0]
        # Step 0 contributed nothing; only step 1's [1, 0] moved the state.
        assert shard["momentum"] == ["1/1", "0/1"]
        assert shard["params"] == ["-1/10", "0/1"]


def check_non_canonical_numeric_inputs_are_rendered_canonically_everywhere():
    """instruction.md: every rational written to `--out` or `--checkpoint-out`
    is `numerator/denominator` in lowest terms with a strictly positive
    denominator, and the accepted input syntax includes decimals, scientific
    notation and `p/q` with a sign on either component. Non-canonical input
    lexemes must not survive into either artifact -- including inside a
    checkpoint pending bucket, which is the one place an input string can be
    echoed back out untouched."""
    config = small_cfg(
        num_workers=3, quorum_size=2, max_staleness=2,
        clip_threshold="1e3", learning_rate="0.1", momentum_beta="2/-4",
    )
    events = [
        scenarios.sub(0, 0, "2/10", "-0/5"),
        scenarios.sub(1, 0, "+3/-6", "1.50E+1"),
        # Left pending at the end of the log, so its gradient is serialized
        # into the checkpoint's pending bucket.
        scenarios.sub(2, 1, "4/-8", "0.000"),
    ]

    expected_out, expected_checkpoint = oracle.run_with_checkpoint(config, events)
    actual_out, actual_checkpoint = support.run_cli_with_checkpoint(config, events)
    assert actual_out == expected_out
    assert actual_checkpoint == expected_checkpoint

    pending = actual_checkpoint["pending"]["1"]
    assert len(pending) == 1
    assert pending[0]["poisoned"] is False
    assert pending[0]["gradient"] == ["-1/2", "0/1"], (
        "a pending gradient echoed its input lexemes instead of canonical rationals"
    )

    for value in support.all_numeric_values(actual_out):
        support.assert_canonical_fraction(value)
    for value in actual_checkpoint["params"] + actual_checkpoint["momentum"]:
        support.assert_canonical_fraction(value)


def check_more_shards_than_parameters_still_emits_a_present_but_empty_tail_shard():
    """instruction.md: output shards cover the vector contiguously and as
    evenly as possible, with earlier shards receiving remainder elements. When
    `num_shards` exceeds `param_size` that rule assigns nothing to the trailing
    shards, which must still be emitted (empty) rather than dropped, and the
    shard ids must stay contiguous from zero."""
    for param_size, num_shards, expected_lengths in (
        (2, 3, [1, 1, 0]),
        (1, 4, [1, 0, 0, 0]),
        (5, 3, [2, 2, 1]),
        (10, 4, [3, 3, 2, 2]),
    ):
        config = small_cfg(
            param_size=param_size, num_shards=num_shards,
            num_workers=2, quorum_size=2, max_staleness=0,
        )
        gradient = [str(index + 1) for index in range(param_size)]
        events = [scenarios.sub(0, 0, *gradient), scenarios.sub(1, 0, *gradient)]

        expected = oracle.run(config, events)
        actual = support.run_cli(config, events)
        assert actual == expected, f"param_size={param_size}/num_shards={num_shards}: diverged"

        shards = actual["final_state"]["shards"]
        assert [shard["shard_id"] for shard in shards] == list(range(num_shards))
        assert [len(shard["params"]) for shard in shards] == expected_lengths, (
            f"param_size={param_size}/num_shards={num_shards}: front-loaded partition expected"
        )
        assert [len(shard["momentum"]) for shard in shards] == expected_lengths


def check_zero_clip_threshold_scales_the_whole_gradient_away():
    """instruction.md: the averaged gradient is clipped once using its own L1
    norm, and nothing in the contract excludes a threshold of zero. At
    clip_threshold=0 any nonzero averaged gradient is above the threshold, so
    the scale is 0/1 and the update contributes nothing -- momentum and params
    stay at 0/1. An implementation that guards clipping with a positivity test
    on the threshold (reading a non-positive threshold as "clipping disabled")
    instead applies the raw gradient and moves the state."""
    config = scenarios.cfg(
        num_workers=1, param_size=2, num_shards=1, quorum_size=1, max_staleness=0,
        clip_threshold="0", learning_rate="1", momentum_beta="0",
    )
    events = [scenarios.sub(0, 0, "1", "-3")]

    expected = oracle.run(config, events)
    actual = support.run_cli(config, events)
    assert actual == expected

    entry = actual["trace"][0]
    assert entry["status"] == "committed"
    assert entry["l1_norm"] == "4/1"
    assert entry["clip_scale"] == "0/1", (
        "a zero clip threshold must scale the gradient away, not disable clipping"
    )
    shard = actual["final_state"]["shards"][0]
    assert shard["momentum"] == ["0/1", "0/1"]
    assert shard["params"] == ["0/1", "0/1"]


def check_only_accepted_contributions_can_poison_a_step():
    """instruction.md: "if any *accepted* contribution in a quorum is
    poisoned". A poisoned submission that was rejected on arrival never joined
    the bucket, so it cannot void the step.

    Worker 0 submits a clean gradient, then resubmits a poisoned one -- a
    repeat, so `duplicate`, and never admitted. Worker 1 completes the quorum
    with a clean gradient, and the step must commit on the two accepted
    contributions. An implementation that flags the step at ingest time,
    before or independently of admission, emits poisoned_skip instead.
    """
    config = scenarios.cfg(
        num_workers=2, param_size=1, num_shards=1, quorum_size=2, max_staleness=0,
        clip_threshold="1000", learning_rate="1", momentum_beta="0",
    )
    events = [
        scenarios.sub(0, 0, "2"),
        {"worker_id": 0, "step": 0, "poisoned": True, "gradient": None},
        scenarios.sub(1, 0, "4"),
    ]

    expected = oracle.run(config, events)
    actual = support.run_cli(config, events)
    assert actual == expected

    assert actual["rejections"] == [{"worker_id": 0, "step": 0, "reason": "duplicate"}]
    entry = actual["trace"][0]
    assert entry["status"] == "committed", (
        "a rejected poisoned submission must not void a step whose accepted "
        "contributions are all clean"
    )
    assert entry["workers"] == [0, 1]
    # average of 2 and 4 is 3, below the threshold, so the state moves by 3.
    assert entry["l1_norm"] == "3/1" and entry["clip_scale"] == "1/1"
    assert actual["final_state"]["shards"][0]["params"] == ["-3/1"]


def check_a_poisoned_log_event_is_checkpointed_without_its_gradient():
    """instruction.md: pending contributions use the normalized checkpoint
    shape, where a poisoned entry carries `gradient: null`. That applies to a
    contribution that arrived as an ordinary `--log` event, not only to one
    loaded from `--checkpoint-in`.

    The event below is explicitly poisoned yet supplies finite values, and no
    partner arrives to complete the quorum, so it is still pending when
    `--checkpoint-out` is written -- which is the only place the normalization
    is observable. An implementation that keeps the supplied values for
    diagnostics while treating the contribution as poisoned for the optimizer
    writes them straight back out here.
    """
    config = scenarios.cfg(
        num_workers=2, param_size=2, num_shards=1, quorum_size=2, max_staleness=1,
        clip_threshold="10", learning_rate="1/10", momentum_beta="1/2",
    )
    events = [{"worker_id": 0, "step": 0, "poisoned": True, "gradient": ["1/2", "3/4"]}]

    expected_out, expected_checkpoint = oracle.run_with_checkpoint(config, events)
    actual_out, actual_checkpoint = support.run_cli_with_checkpoint(config, events)
    assert actual_out == expected_out
    assert actual_checkpoint == expected_checkpoint

    assert actual_out["trace"] == []
    assert actual_out["uncommitted_pending"] == {"0": [0]}
    pending = actual_checkpoint["pending"]["0"]
    assert len(pending) == 1
    assert pending[0]["poisoned"] is True
    assert pending[0]["gradient"] is None, (
        "a poisoned pending contribution must be checkpointed with gradient null, "
        "not with the values the log event supplied"
    )


# ---------------------------------------------------------------------------
# The five graded tests.
#
# Everything above, and every check in checks_scenarios / checks_checkpoint /
# checks_properties, is a plain helper rather than a collected test. The suite
# grades five groups instead of fifty individual functions for one reason: the
# reward is a single binary outcome, and a fan-out of small tests turns a
# partial repair into a mostly-green report. One wrong event, in any scenario,
# at any checkpoint split, fails the group it belongs to.
#
# The groups are ordered cheapest-first, so a submission that does not build,
# or that reaches for the goldens, fails before the long sweeps run.
# ---------------------------------------------------------------------------

import checks_checkpoint
import checks_properties
import checks_scenarios


def _run_all(*checks) -> None:
    for check in checks:
        check()


def test_build_isolation_and_verifier_self_checks():
    """The submission compiles from the documented source tree; the account
    that runs it can reach neither the goldens, the oracle, nor the reward
    channel; and the schema validators every graded run passes through
    actually reject what they claim to (native-type aliases, non-canonical
    rationals, a poisoned pending entry that kept its gradient). The oracle
    itself is anchored against the six hand-captured goldens here, so no
    expectation in the rest of the suite rests on one implementation alone."""
    _run_all(
        check_project_compiles,
        check_agent_code_runs_without_access_to_the_verifier,
        checks_properties.check_schema_validation_rejects_type_aliases_and_non_canonical_rationals,
        checks_properties.check_oracle_and_reference_regression_fixtures_agree,
    )


def test_full_replay_matches_the_goldens_and_the_independent_oracle():
    """Whole-artifact equality, over the six golden fixtures and over every
    scenario in the library, plus the invariances that pin them down: output
    unchanged by shard count or worker arrival order, generalization variants
    recomputed from scratch, byte-identical repeated runs, and identical
    output under three JVM default locales."""
    _run_all(
        check_final_output_matches_golden_exactly,
        check_emitted_rationals_and_shard_partitions_are_well_formed,
        checks_scenarios.check_full_output_matches_independent_oracle_for_every_scenario,
        checks_scenarios.check_generalization_variants_recompute_independently,
        checks_scenarios.check_shard_count_never_changes_the_numeric_outcome,
        checks_scenarios.check_worker_arrival_order_permutation_does_not_change_final_state,
        checks_properties.check_repeated_runs_are_byte_identical,
        check_poison_token_recognition_does_not_move_with_the_jvm_default_locale,
    )


def test_admission_resolution_and_poison_rules_at_their_boundaries():
    """The replay rules where an off-by-one or a mis-ordered check changes the
    answer: the staleness window swept across its edge, late classification,
    frontier-only resolution, duplicate outranking superfluous on a full
    bucket, first-accepted-wins, gap-free commit order, and poisoning --- all
    four token spellings at every gradient position on both input paths, at
    every arrival position in a quorum, keyed on accepted contributions only,
    skipping the update while the frontier still advances."""
    _run_all(
        check_staleness_boundary_accepts_exactly_at_window_edge,
        check_staleness_window_admits_exactly_up_to_max_staleness,
        checks_scenarios.check_only_the_frontier_resolves_and_a_stale_worker_may_retry_later,
        checks_scenarios.check_duplicate_check_outranks_superfluous_check,
        check_first_accepted_contribution_per_worker_is_the_one_that_counts,
        check_rejection_reasons_and_poison_are_attributed_correctly,
        check_commit_order_is_gap_free_and_trailing_pending_is_reported,
        check_every_poison_token_is_detected_at_every_position_on_both_input_paths,
        checks_scenarios.check_every_raw_nonfinite_token_spelling_poisons_its_step,
        check_poison_at_any_quorum_position_voids_the_step_and_advances_the_frontier,
        check_only_accepted_contributions_can_poison_a_step,
        checks_scenarios.check_poisoned_step_leaves_momentum_chain_untouched,
        checks_scenarios.check_all_rejections_log_never_mutates_optimizer_state,
        checks_properties.check_deleting_rejected_events_from_the_log_does_not_change_final_state,
        checks_properties.check_scenarios_with_no_commits_leave_all_zero_state,
    )


def test_clipping_sharding_and_exact_rational_arithmetic():
    """The numeric half of the contract: one clip of the averaged vector using
    that vector's own global L1 norm, including a threshold of zero and
    thresholds that sit razor-thin either side of the norm; a front-loaded
    contiguous shard partition, including remainders and more shards than
    parameters; and exact rational arithmetic end to end, with every accepted
    input syntax rendered back in canonical lowest terms."""
    _run_all(
        check_clip_scale_matches_threshold_over_norm_formula,
        check_cross_shard_clip_uses_global_norm_not_per_shard,
        check_clip_is_applied_once_to_the_averaged_gradient,
        check_zero_clip_threshold_scales_the_whole_gradient_away,
        checks_scenarios.check_exact_rational_clip_decision_survives_float_adjacent_values,
        checks_scenarios.check_shard_partitioning_is_contiguous_exhaustive_and_front_loaded,
        check_more_shards_than_parameters_still_emits_a_present_but_empty_tail_shard,
        checks_scenarios.check_all_accepted_numeric_string_forms_stay_exact_and_render_canonically,
        check_non_canonical_numeric_inputs_are_rendered_canonically_everywhere,
    )


def test_checkpoint_persistence_and_resume_semantics():
    """Everything about `--checkpoint-out` and `--checkpoint-in`: the raw
    checkpoint compared against the oracle at every split of every scenario,
    resume reaching the same final state, a fresh trace alongside a cumulative
    rejection history, the per-step rejection entry cleared when its step
    resolves yet reopened by a later stray, pending buckets normalized on both
    the write and the read side, every accepted numeric form surviving a
    round trip, and already-quorate frontier buckets drained before the first
    tail event is classified."""
    _run_all(
        checks_checkpoint.check_raw_checkpoint_file_shape_matches_independent_oracle_at_every_split,
        checks_checkpoint.check_checkpoint_resume_equals_uninterrupted_replay,
        check_checkpoint_resume_over_golden_logs_reaches_the_same_final_state,
        checks_checkpoint.check_checkpoint_resume_trace_and_rejections_reconstruct_exactly,
        checks_checkpoint.check_checkpoint_emitted_after_resume_matches_independent_oracle,
        checks_checkpoint.check_kitchen_sink_scenarios_survive_every_meaningful_checkpoint_split,
        checks_checkpoint.check_checkpoint_preserves_stray_rejection_for_already_committed_step,
        checks_checkpoint.check_checkpoint_preserves_rejections_for_blocked_future_step,
        checks_checkpoint.check_checkpoint_captures_pending_raw_nonfinite_poison_before_quorum,
        checks_checkpoint.check_checkpoint_pending_poison_is_normalized_on_load,
        check_checkpoint_explicit_poison_flag_discards_any_supplied_gradient,
        check_a_poisoned_log_event_is_checkpointed_without_its_gradient,
        checks_checkpoint.check_checkpoint_input_accepts_every_documented_numeric_string_form,
        checks_checkpoint.check_checkpoint_input_numeric_forms_inside_pending_gradients_replay_exactly,
        checks_checkpoint.check_checkpoint_with_quorate_frontier_drains_before_tail_events,
        checks_checkpoint.check_checkpoint_drain_happens_before_first_tail_event_is_classified,
    )
