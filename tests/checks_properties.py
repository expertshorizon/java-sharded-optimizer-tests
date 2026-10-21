"""Cross-cutting invariants that hold across the whole scenario library,
independent of any one scenario's specific numbers. Each of these is
checked against every scenario for which it is meaningful, rather than
pinned to one hand-picked fixture, so a fix that only special-cases the
original six fixtures still gets caught here.
"""

import copy
from pathlib import Path

import pytest

import oracle
import scenarios
import support

DETERMINISM_SCENARIOS = [
    scenarios.by_name("kitchen_sink_a_ordering_staleness_clip_rational"),
    scenarios.by_name("kitchen_sink_c_uneven_config_duplicate_clip_momentum"),
    scenarios.by_name("deep_out_of_order_cascade"),
    scenarios.by_name("burst_cascade_multiple_ready_before_frontier"),
]

_EMPTY_TRACE_SCENARIOS = [
    s for s in scenarios.SCENARIOS
    if oracle.run(s.config, s.events)["trace"] == []
]

_REMOVAL_CANDIDATES = [
    s for s in scenarios.SCENARIOS
    if oracle.run(s.config, s.events)["rejections"]
][:14]

GOLDEN_NAMES = ["basic", "cascade", "staleness", "clip", "poison", "stress"]


def check_repeated_runs_are_byte_identical():
    """Running the exact same log twice, in two entirely separate process
    invocations, must produce byte-identical --out text — not merely
    structurally-equal JSON. This catches HashMap/HashSet iteration-order
    dependence, unintended shared mutable state between runs, or any other
    source of nondeterminism, none of which the parsed-JSON comparisons
    elsewhere in this suite would notice if both runs happened to scramble
    key order the same way by coincidence."""
    for scenario in DETERMINISM_SCENARIOS:
        first = support.run_cli_raw_text_uncached(scenario.config, scenario.events)
        second = support.run_cli_raw_text_uncached(scenario.config, scenario.events)
        assert first == second, f"{scenario.name}: two runs of the same log differ"


def check_scenarios_with_no_commits_leave_all_zero_state():
    """Wherever the independent oracle predicts that nothing ever reaches
    quorum (every submission is rejected or forever blocked), the actual
    program's params and momentum must still be exactly all-zero — a
    rejected or perpetually-pending submission must never partially write
    through to optimizer state."""
    assert _EMPTY_TRACE_SCENARIOS, "no commit-free scenario left in the library"
    for scenario in _EMPTY_TRACE_SCENARIOS:
        actual = support.run_cli(scenario.config, scenario.events)
        assert actual["trace"] == [], scenario.name
        for shard in actual["final_state"]["shards"]:
            assert all(v == "0/1" for v in shard["params"]), scenario.name
            assert all(v == "0/1" for v in shard["momentum"]), scenario.name


def check_deleting_rejected_events_from_the_log_does_not_change_final_state():
    """A rejected submission is classified purely from state that existed
    before it arrived; it never joins a bucket, so removing it from the log
    entirely must be invisible to every later admission decision and to the
    final params/momentum. This is a strictly stronger check than 'an
    all-rejected log leaves zero state': it holds even when real commits
    are mixed in among the rejections."""
    assert _REMOVAL_CANDIDATES, "no scenario with rejections left in the library"
    for scenario in _REMOVAL_CANDIDATES:
        decisions = oracle.classify_each_event(scenario.config, scenario.events)
        trimmed_events = [e for e, d in zip(scenario.events, decisions) if d == "accept"]
        assert len(trimmed_events) < len(scenario.events), (
            f"{scenario.name} has no rejections to remove"
        )

        full = support.run_cli(scenario.config, scenario.events)
        trimmed = support.run_cli(scenario.config, trimmed_events)
        assert full["final_state"] == trimmed["final_state"], (
            f"{scenario.name}: removing rejected events changed the final state"
        )


def check_oracle_and_reference_regression_fixtures_agree():
    """Sanity anchor: the independent oracle must reproduce every one of
    the six original hand-captured golden fixtures exactly. If this ever
    fails, the oracle itself (not the agent's code) has drifted from
    the replay contract, and every oracle-driven test in this suite is suspect
    until it is fixed."""
    fixtures = Path(__file__).parent / "fixtures"
    for name in GOLDEN_NAMES:
        config = support.load_json(fixtures / f"config_{name}.json")
        events = support.load_json(fixtures / f"log_{name}.json")
        golden = support.load_json(fixtures / f"golden_{name}.json")
        assert oracle.run(config, events) == golden, f"oracle diverged from golden_{name}.json"


def check_schema_validation_rejects_type_aliases_and_non_canonical_rationals():
    """Self-check on the validator every graded run passes through, in two
    halves.

    Types: it must reject booleans/floats in integer-valued JSON fields even
    though Python compares False == 0, True == 1 and 0.0 == 0, so whole-object
    equality cannot accept an output that changes the native JSON schema while
    preserving Python equality.

    Canonical form: every rational either artifact carries has to be a reduced
    `p/q` with a strictly positive denominator. A pending checkpoint gradient
    is the field most likely to carry a raw input lexeme instead ('1/-2',
    '2/10', a bare '1'), and equality against the oracle only catches that on
    the splits some test happens to sample, so the validator has to reject it
    on its own, on every run.
    """
    scenario = scenarios.by_name("deep_out_of_order_cascade")
    output = oracle.run(scenario.config, scenario.events)
    _, checkpoint = oracle.run_with_checkpoint(scenario.config, scenario.events[:1])

    bad_shard = copy.deepcopy(output)
    bad_shard["final_state"]["shards"][0]["shard_id"] = False
    with pytest.raises(AssertionError):
        support.validate_output_schema(bad_shard)

    bad_worker = copy.deepcopy(output)
    bad_worker["trace"][0]["workers"][0] = 0.0
    with pytest.raises(AssertionError):
        support.validate_output_schema(bad_worker)

    bad_checkpoint_step = copy.deepcopy(checkpoint)
    bad_checkpoint_step["committed_step"] = -1.0
    with pytest.raises(AssertionError):
        support.validate_checkpoint_schema(bad_checkpoint_step)

    bad_checkpoint_worker = copy.deepcopy(checkpoint)
    first_bucket = next(iter(bad_checkpoint_worker["pending"].values()))
    first_bucket[0]["worker_id"] = False
    with pytest.raises(AssertionError):
        support.validate_checkpoint_schema(bad_checkpoint_worker)

    assert next(iter(checkpoint["pending"].values()))[0]["gradient"], (
        "fixture no longer has a pending gradient to mangle"
    )

    support.validate_output_schema(output)
    support.validate_checkpoint_schema(checkpoint)

    for bad_value in ("1/-2", "2/10", "1", "0.5", "-0/1", "1/0"):
        bad_output = copy.deepcopy(output)
        bad_output["final_state"]["shards"][0]["params"][0] = bad_value
        with pytest.raises(AssertionError):
            support.validate_output_schema(bad_output)

        bad_checkpoint = copy.deepcopy(checkpoint)
        next(iter(bad_checkpoint["pending"].values()))[0]["gradient"][0] = bad_value
        with pytest.raises(AssertionError):
            support.validate_checkpoint_schema(bad_checkpoint)

        bad_momentum = copy.deepcopy(checkpoint)
        bad_momentum["momentum"][0] = bad_value
        with pytest.raises(AssertionError):
            support.validate_checkpoint_schema(bad_momentum)

    bad_norm = copy.deepcopy(output)
    committed = [entry for entry in bad_norm["trace"] if entry["l1_norm"] is not None]
    assert committed, "fixture no longer has a committed step to mangle"
    committed[0]["l1_norm"] = "12"
    with pytest.raises(AssertionError):
        support.validate_output_schema(bad_norm)
