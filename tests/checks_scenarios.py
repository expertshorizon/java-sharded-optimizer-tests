"""Scenario-driven coverage: every scenario in tests/scenarios.py checked
against tests/oracle.py, an independent from-scratch reimplementation of
the candidate-visible contract that never imports or calls anything under
/app/src. No expected value here is a hand-authored golden file — it is
computed fresh, every run, from the same (config, events) the agent's
binary sees.

The scenario library is swept inside single test functions rather than
parametrized one-test-per-scenario: the library exists to make a repair
generalize, so passing 50 of 51 scenarios is a failure, not 98% credit.
Failure messages carry the scenario name, so a red run still says which
trajectory diverged.
"""

import oracle
import scenarios
import support

ALL = scenarios.SCENARIOS


def run(scenario: scenarios.Scenario) -> dict:
    return support.run_cli(scenario.config, scenario.events)


def assert_matches_oracle(scenario: scenarios.Scenario) -> dict:
    actual = run(scenario)
    expected = oracle.run(scenario.config, scenario.events)
    assert actual == expected, f"{scenario.name}: output diverged from the independent oracle"
    return actual


def check_full_output_matches_independent_oracle_for_every_scenario():
    """The whole --out structure (final_state, trace, uncommitted_pending,
    rejections) must exactly equal what an independent Python replay of the
    same protocol computes, for every scenario in the library. This single
    check already implies canonical fraction form, the clip_scale formula,
    exact shard partitioning, and gap-free commit ordering everywhere —
    since the oracle's own output always has those properties by
    construction, so checking any of them again separately here would be
    the repetitive-assertion pattern the task brief explicitly warns
    against. Those properties are instead exercised where they carry
    independent signal: test_outputs.py's golden-file layer (which predates
    and does not depend on the oracle), and the cross-run / oracle-free
    checks in test_properties.py and test_checkpoint.py."""
    assert len(ALL) >= 50, "scenario library shrank unexpectedly"
    for scenario in ALL:
        assert_matches_oracle(scenario)


# --------------------------------------------------------------------------
# Bespoke checks tied to a specific scenario's documented interaction.
# --------------------------------------------------------------------------

def check_only_the_frontier_resolves_and_a_stale_worker_may_retry_later():
    """Two halves of the same admission rule. A quorate future step cannot
    resolve while the frontier is still short of quorum, and anything past the
    staleness window is rejected; but a stale rejection is not a ban -- once
    the frontier catches up, the same worker's resubmission for that step is
    admitted and commits."""
    blocked = run(scenarios.by_name("staleness_blocked_by_stuck_frontier"))
    assert blocked["trace"] == []
    assert blocked["uncommitted_pending"] == {"0": [0], "2": [1, 2]}
    assert blocked["rejections"] == [{"worker_id": 3, "step": 3, "reason": "stale"}]

    retried = run(scenarios.by_name("staleness_retry_succeeds_after_frontier_catches_up"))
    stale = [r for r in retried["rejections"] if r["reason"] == "stale"]
    assert stale == [{"worker_id": 0, "step": 3, "reason": "stale"}]
    committed_steps = {e["step"] for e in retried["trace"] if e["status"] == "committed"}
    assert committed_steps == {0, 1, 2, 3}


def check_duplicate_check_outranks_superfluous_check():
    scenario = scenarios.by_name("duplicate_outranks_superfluous_on_full_bucket")
    actual = run(scenario)
    reasons = {(r["worker_id"], r["step"]): r["reason"] for r in actual["rejections"]}
    assert reasons[(0, 3)] == "duplicate"
    assert reasons[(2, 3)] == "superfluous"


def check_poisoned_step_leaves_momentum_chain_untouched():
    scenario = scenarios.by_name("poison_then_next_step_preserves_prior_state")
    actual = assert_matches_oracle(scenario)
    statuses = [e["status"] for e in actual["trace"]]
    assert statuses == ["committed", "poisoned_skip", "committed"]
    # step 0 alone would leave momentum at 1.5 (beta=0.5, clipped=3);
    # step 2 must resume the EMA from that value, not from zero.


def check_all_rejections_log_never_mutates_optimizer_state():
    scenario = scenarios.by_name("rejected_only_log_never_mutates_state")
    actual = run(scenario)
    assert actual["trace"] == []
    for shard in actual["final_state"]["shards"]:
        assert all(v == "0/1" for v in shard["params"])
        assert all(v == "0/1" for v in shard["momentum"])


def check_exact_rational_clip_decision_survives_float_adjacent_values():
    """These scenarios are specifically constructed so that IEEE-double
    summation of the gradient components lands on the wrong side of the
    clip_threshold comparison (see scenarios.py docstrings for the exact
    ULP-level reasoning); only exact rational arithmetic gets the clip
    decision right."""
    for name in (
        "adversarial_decimal_sum_boundary",
        "clip_ratio_razor_thin_over_threshold",
        "clip_ratio_razor_thin_under_threshold",
    ):
        assert_matches_oracle(scenarios.by_name(name))


def check_worker_arrival_order_permutation_does_not_change_final_state():
    """Two scenarios submit the exact same two quorums with a different
    relative arrival order among interchangeable workers; the aggregated,
    clipped, momentum-updated final_state must be identical even though
    each trace entry's own 'workers' arrival list may legitimately differ."""
    a = run(scenarios.by_name("worker_permutation_variant_a"))
    b = run(scenarios.by_name("worker_permutation_variant_b"))
    assert a["final_state"] == b["final_state"]
    workers_a = [e["workers"] for e in a["trace"]]
    workers_b = [e["workers"] for e in b["trace"]]
    assert workers_a != workers_b, "test fixture no longer exercises a real arrival-order difference"


def check_shard_count_never_changes_the_numeric_outcome():
    """The same aggregated gradient reported through 1, 2, and 4 shards
    must produce identical params/momentum once the shard partitioning is
    flattened back out — only the reporting partition may differ."""
    variants = [
        "dominant_shard_norm_cross_shard_clip",
        "dominant_shard_norm_cross_shard_clip__shards1",
        "dominant_shard_norm_cross_shard_clip__shards4",
    ]
    flattened = []
    for name in variants:
        out = run(scenarios.by_name(name))
        params = [v for shard in out["final_state"]["shards"] for v in shard["params"]]
        momentum = [v for shard in out["final_state"]["shards"] for v in shard["momentum"]]
        flattened.append((params, momentum))
    assert flattened[0] == flattened[1] == flattened[2]


def check_generalization_variants_recompute_independently():
    """Worker-id-offset and shard-count variants of richer base scenarios
    must still match a freshly recomputed oracle result — nothing about
    correctness here can depend on the specific worker ids or shard count
    used in the base scenario."""
    variants = scenarios.with_tag("generalization")
    assert variants, "generalization variants disappeared from the library"
    for scenario in variants:
        assert_matches_oracle(scenario)


def check_every_raw_nonfinite_token_spelling_poisons_its_step():
    """instruction.md names four distinct raw non-finite spellings ('NaN',
    'Inf', '+Inf', '-Inf') that must poison a submission even when the
    event's own poisoned flag is false. The four isolated fixtures are each
    a single step, single shard, two-worker quorum with no staleness,
    sharding, or multi-step cascade to confound the result: the step must
    resolve as poisoned_skip, both workers must still occupy the quorum
    slots, and no optimizer update may be applied, so a solution that only
    recognizes 'NaN' (or only an unsigned 'Inf') fails here even if it
    passes the combined cascading scenario. That cascading scenario is
    checked too: a raw token must void its step and still advance the
    frontier so the following step commits."""
    for name in (
        "raw_nan_token_poisons_isolated_step",
        "raw_inf_token_poisons_isolated_step",
        "raw_plus_inf_token_poisons_isolated_step",
        "raw_minus_inf_token_poisons_isolated_step",
    ):
        scenario = scenarios.by_name(name)
        actual = assert_matches_oracle(scenario)
        assert [entry["status"] for entry in actual["trace"]] == ["poisoned_skip"], name
        assert actual["trace"][0]["workers"] == [0, 1], name
        for shard in actual["final_state"]["shards"]:
            assert all(v == "0/1" for v in shard["params"]), name
            assert all(v == "0/1" for v in shard["momentum"]), name

    cascading = scenarios.by_name("raw_nonfinite_token_poison_cascades")
    actual = assert_matches_oracle(cascading)
    assert [entry["status"] for entry in actual["trace"]] == ["poisoned_skip", "committed"]
    assert actual["trace"][0]["workers"] == [0, 1]


def check_all_accepted_numeric_string_forms_stay_exact_and_render_canonically():
    """Fractions, scientific notation, decimals, and p/q strings carrying a
    sign on the denominator are all valid input syntax, in both the config
    and a gradient. Every one of them must be parsed to its exact rational
    value, and no emitted rational may preserve a negative denominator or an
    unreduced form."""
    for name in (
        "exact_fraction_scientific_inputs",
        "negative_denominator_fraction_inputs_normalize",
    ):
        actual = assert_matches_oracle(scenarios.by_name(name))
        values = support.all_numeric_values(actual)
        assert values, f"{name}: expected at least one numeric value to check"
        for value in values:
            support.assert_canonical_fraction(value)
            assert "/-" not in value, f"{name}: emitted a negative denominator in {value}"


def check_shard_partitioning_is_contiguous_exhaustive_and_front_loaded():
    """Three partitioning cases with different remainder structure.

    5 over 3 has remainder == num_shards - 1, where the correct
    front-loaded partition (2,2,1) coincides with the wrong strategy of
    giving every shard but the last a ceil-sized share and dumping the
    leftover into the last shard alone. 10 over 4 breaks that coincidence:
    the correct partition is 3,3,2,2, while ceil-except-last produces
    3,3,3,1. And when num_shards exceeds param_size, every configured
    shard must still be emitted — 2 over 3 is 1,1,0, with empty arrays on
    the trailing shard rather than dropping it or inventing an element.
    Total coverage alone would let all three wrong answers through, so the
    exact per-shard sizes are asserted."""
    expected_sizes = {
        "uneven_shards_cover_entire_vector": [2, 2, 1],
        "uneven_shards_remainder_not_confined_to_final_shard": [3, 3, 2, 2],
        "more_shards_than_parameters_preserves_empty_tail_shard": [1, 1, 0],
    }
    for name, sizes in expected_sizes.items():
        scenario = scenarios.by_name(name)
        actual = assert_matches_oracle(scenario)
        shards = actual["final_state"]["shards"]
        assert [shard["shard_id"] for shard in shards] == list(range(len(sizes))), name
        assert [len(shard["params"]) for shard in shards] == sizes, name
        assert [len(shard["momentum"]) for shard in shards] == sizes, name
        assert sum(len(shard["params"]) for shard in shards) == scenario.config["param_size"], name
