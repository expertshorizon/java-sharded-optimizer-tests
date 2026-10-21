"""Deterministic scenario library for the hardened test suite.

Every scenario is plain (config, events) data — no scenario carries a
hand-written expected result. Expected values are always computed at test
time by `oracle.py`, independently of both the fixture and the submitted
implementation, so nothing here can accidentally become a hardcoded
trajectory. Scenario names double as pytest parametrize ids; keep them
unique and descriptive of what interaction the scenario is stressing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Scenario:
    name: str
    config: dict
    events: list
    doc: str
    tags: Tuple[str, ...] = ()


def cfg(*, num_workers, param_size, num_shards, quorum_size, max_staleness,
        clip_threshold, learning_rate, momentum_beta):
    return {
        "num_workers": num_workers,
        "param_size": param_size,
        "num_shards": num_shards,
        "quorum_size": quorum_size,
        "max_staleness": max_staleness,
        "clip_threshold": str(clip_threshold),
        "learning_rate": str(learning_rate),
        "momentum_beta": str(momentum_beta),
    }


def sub(worker_id, step, *gradient):
    return {
        "type": "submit",
        "worker_id": worker_id,
        "step": step,
        "poisoned": False,
        "gradient": [str(g) for g in gradient],
    }


def poison(worker_id, step):
    return {"type": "submit", "worker_id": worker_id, "step": step, "poisoned": True}


_BASE: List[Scenario] = []


def _add(name, config, events, doc, tags=()):
    _BASE.append(Scenario(name, config, events, doc, tuple(tags)))


# --------------------------------------------------------------------------
# A. Global cascade / commit-order complexity
# --------------------------------------------------------------------------

_add(
    "deep_out_of_order_cascade",
    cfg(num_workers=4, param_size=2, num_shards=1, quorum_size=2, max_staleness=5,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0.3"),
    [
        sub(2, 3, 1, 1), sub(0, 4, 1, 1), sub(1, 2, 1, 1), sub(3, 0, 1, 1),
        sub(0, 1, 1, 1), sub(2, 0, 1, 1),   # completes step0 (3,2)
        sub(1, 1, 1, 1),                    # completes step1 (0,1)
        sub(3, 2, 1, 1),                    # completes step2 (1,3)
        sub(1, 3, 1, 1),                    # completes step3 (2,1)
        sub(3, 4, 1, 1),                    # completes step4 (0,3)
    ],
    "Five logical steps arrive fully scrambled (step numbers appear in the "
    "order 3,4,2,0,1,0,1,2,3,4); a greedy processor that commits whichever "
    "bucket reaches quorum first would commit in arrival order, not "
    "frontier order, and — since momentum_beta != 0 — a wrong commit order "
    "changes every downstream momentum value even though every step uses "
    "the same gradient shape.",
    tags=("cascade", "ordering"),
)

_add(
    "burst_cascade_multiple_ready_before_frontier",
    cfg(num_workers=6, param_size=2, num_shards=1, quorum_size=2, max_staleness=10,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0.4"),
    [
        sub(1, 1, 4, 0), sub(2, 1, 0, 4),   # step1 fully quorate, blocked behind step0
        sub(3, 2, 1, 1), sub(4, 2, 3, 3),   # step2 fully quorate, blocked
        sub(5, 3, 5, 1), sub(0, 3, 1, 5),   # step3 fully quorate, blocked
        sub(0, 0, 2, 0), sub(1, 0, 0, 2),   # step0 completes -> cascades through 1,2,3 in one call
    ],
    "Steps 1, 2 and 3 each reach full quorum while step 0 is still empty. "
    "The single event that finally completes step 0 must trigger a single "
    "cascade that commits four steps back-to-back in strict order — "
    "committing 1, 2 or 3 the moment their own bucket filled (ignoring the "
    "frontier) would both reorder the trace and, via momentum_beta, change "
    "the final numeric state.",
    tags=("cascade", "ordering", "quorum"),
)

_add(
    "stray_late_rejection_after_commit",
    cfg(num_workers=3, param_size=1, num_shards=1, quorum_size=1, max_staleness=2,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0"),
    [
        sub(0, 0, 1),      # commits step0 immediately (quorum_size=1)
        sub(1, 0, 2),      # late: step0 already committed
        sub(2, 0, 3),      # late again, from a third, different worker
        sub(0, 1, 1),      # commits step1
    ],
    "Two further submissions arrive for a step number that has already "
    "committed. Both must be recorded as 'late' in the flat rejection log "
    "even though that step's own trace entry (frozen at commit time) will "
    "never show them — and (see checks_checkpoint.py) a checkpoint taken "
    "after this point must still carry the stray per-step rejection entry.",
    tags=("ordering", "late"),
)

_add(
    "duplicate_outranks_superfluous_on_full_bucket",
    cfg(num_workers=4, param_size=1, num_shards=1, quorum_size=2, max_staleness=5,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0"),
    [
        sub(0, 3, 1), sub(1, 3, 1),   # step3 bucket reaches quorum_size=2, blocked behind step0/1/2
        sub(0, 3, 9),                  # same worker resubmits to an already-full bucket -> duplicate, not superfluous
        sub(2, 3, 9),                  # a genuinely new worker on a full bucket -> superfluous
        sub(0, 0, 1), sub(1, 0, 1),    # commit step0
        sub(0, 1, 1), sub(1, 1, 1),    # commit step1
        sub(0, 2, 1), sub(1, 2, 1),    # commit step2 -> cascades step3 too
    ],
    "A worker that already has a slot in a step's bucket resubmits after "
    "that bucket has already reached quorum_size. The duplicate check must "
    "still fire before the capacity check — classifying this as "
    "'superfluous' instead of 'duplicate' would be a plausible but wrong "
    "reordering of the two checks.",
    tags=("ordering", "duplicate", "superfluous"),
)

# --------------------------------------------------------------------------
# B. Bounded staleness x quorum interaction
# --------------------------------------------------------------------------

_add(
    "staleness_blocked_by_stuck_frontier",
    cfg(num_workers=4, param_size=1, num_shards=1, quorum_size=2, max_staleness=2,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0"),
    [
        sub(0, 0, 1),                 # step0 permanently stuck at 1/2 workers: frontier never leaves 0
        sub(1, 2, 1), sub(2, 2, 1),   # step2 (lookahead=2, exactly at the boundary): both accepted, fully
                                        # quorate, yet can never commit because it is not the frontier
        sub(3, 3, 1),                  # step3 (lookahead=3 > max_staleness=2): rejected stale
    ],
    "One gradient is individually fresh enough AND its step reaches full "
    "quorum, yet it can never commit purely because an earlier step is "
    "stuck; a step one hop further out is rejected outright as stale. Both "
    "facts depend on the *same* frozen frontier, so staleness cannot be "
    "checked in isolation from quorum/commit-order state.",
    tags=("staleness", "quorum"),
)

_add(
    "staleness_retry_succeeds_after_frontier_catches_up",
    cfg(num_workers=4, param_size=1, num_shards=1, quorum_size=2, max_staleness=1,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0.25"),
    [
        sub(0, 3, 9),                     # frontier=0, lookahead=3 > 1 -> rejected stale, NOT buffered
        sub(1, 0, 1), sub(2, 0, 1),        # commit step0, frontier=1
        sub(1, 1, 1), sub(2, 1, 1),        # commit step1, frontier=2
        sub(1, 2, 1), sub(2, 2, 1),        # commit step2, frontier=3
        sub(0, 3, 9),                      # retry: frontier=3, lookahead=0 -> accepted this time
        sub(3, 3, 3),                       # completes step3's quorum -> commits
    ],
    "Worker 0's first attempt at step 3 is rejected stale while the "
    "frontier is still far behind. The exact same worker resubmits to the "
    "exact same step later, once three intervening steps have advanced the "
    "frontier, and this time is accepted — proving a stale rejection does "
    "not permanently reserve or poison that worker's slot in the bucket.",
    tags=("staleness", "retry"),
)

_add(
    "staleness_boundary_chain_shifting_frontier",
    cfg(num_workers=6, param_size=1, num_shards=1, quorum_size=1, max_staleness=3,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0"),
    [
        sub(0, 3, 1),   # frontier=0: lookahead=3 == max_staleness -> accepted, buffered (blocked, not frontier)
        sub(1, 4, 1),   # frontier=0: lookahead=4 > max_staleness -> rejected stale
        sub(2, 0, 1),   # commit step0 (quorum_size=1) -> frontier=1 -> cascades: step3's bucket isn't frontier yet, stays
        sub(3, 4, 1),   # frontier=1: lookahead=3 == max_staleness -> accepted (would have been stale at frontier=0)
        sub(4, 1, 1),   # commit step1 -> frontier=2
        sub(5, 2, 1),   # commit step2 -> frontier=3 -> cascades step3 (already buffered) -> commits -> cascades step4
    ],
    "The same lookahead value (3) is accepted when the frontier is 0 and "
    "again when the frontier is 1, but a lookahead of 4 is rejected at "
    "frontier 0 — the boundary is re-evaluated against the *current* "
    "frontier at each arrival, not a fixed reference version, and one "
    "step's commit is what shifts that reference for the next arrival.",
    tags=("staleness", "boundary", "ordering"),
)

_add(
    "max_staleness_zero_strict",
    cfg(num_workers=3, param_size=1, num_shards=1, quorum_size=1, max_staleness=0,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0"),
    [
        sub(0, 1, 1),   # frontier=0, lookahead=1 > 0 -> stale even though it is only one step ahead
        sub(1, 0, 1),   # frontier=0, lookahead=0 -> accepted, commits immediately
        sub(2, 1, 1),   # frontier=1, lookahead=0 -> accepted now
    ],
    "With max_staleness=0 only the exact current frontier is ever "
    "admissible; a lookahead of exactly 1 is already too stale. Exercises "
    "the degenerate edge of the staleness window.",
    tags=("staleness", "boundary"),
)

# --------------------------------------------------------------------------
# C. Sharding and cross-shard clipping
# --------------------------------------------------------------------------

_add(
    "single_element_shards_full_partition",
    cfg(num_workers=1, param_size=4, num_shards=4, quorum_size=1, max_staleness=1,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0"),
    [sub(0, 0, 3, -5, 0, 12)],
    "num_shards == param_size: every shard holds exactly one component, "
    "including a negative and a zero value, so any off-by-one in shard "
    "boundary math is immediately visible as a value landing in the wrong "
    "shard.",
    tags=("sharding",),
)

_add(
    "shard_boundary_crossing_distinct_values",
    cfg(num_workers=1, param_size=4, num_shards=2, quorum_size=1, max_staleness=1,
        clip_threshold="100000", learning_rate="0.1", momentum_beta="0"),
    [sub(0, 0, 10, 20, 30, 40)],
    "Four distinct, easily-misplaced values (10,20,30,40) split into two "
    "shards of two; clip_threshold is large enough that clipping never "
    "activates, isolating pure shard-partitioning correctness from "
    "clipping correctness.",
    tags=("sharding",),
)

_add(
    "dominant_shard_norm_cross_shard_clip",
    cfg(num_workers=1, param_size=4, num_shards=2, quorum_size=1, max_staleness=1,
        clip_threshold="10", learning_rate="0.1", momentum_beta="0"),
    [sub(0, 0, 1, 1, 100, 100)],
    "Shard 0 (indices 0,1) is individually well under the clip threshold "
    "(l1=2); shard 1 alone would also look enormous either way. The "
    "correct global norm is 202, and the resulting scale must be applied "
    "uniformly to *every* component including the small shard-0 values — "
    "a per-shard clipper would leave shard 0 completely unclipped.",
    tags=("clipping", "sharding"),
)

_add(
    "clip_ratio_exactly_one_boundary",
    cfg(num_workers=1, param_size=1, num_shards=1, quorum_size=1, max_staleness=1,
        clip_threshold="5", learning_rate="0.1", momentum_beta="0"),
    [sub(0, 0, 5)],
    "l1_norm equals clip_threshold exactly: the spec requires scale = 1/1 "
    "via the <= branch, not the division branch, but the numeric result "
    "coincides either way — this is really a canonical-fraction-reduction "
    "check (does the division branch, if taken anyway, still reduce "
    "threshold/l1_norm to a clean 1/1?).",
    tags=("clipping", "boundary", "rational"),
)

_add(
    "clip_ratio_razor_thin_over_threshold",
    cfg(num_workers=3, param_size=1, num_shards=1, quorum_size=3, max_staleness=1,
        clip_threshold="7", learning_rate="0.1", momentum_beta="0"),
    [sub(0, 0, 7), sub(1, 0, 7), sub(2, 0, 8)],
    "Aggregate = (7+7+8)/3 = 22/3, a non-terminating decimal strictly "
    "greater than clip_threshold=7 by exactly 1/3. clip_scale must come "
    "out to the exact reduced fraction 21/22, not a rounded approximation "
    "— a float implementation would blur this razor-thin excess.",
    tags=("clipping", "boundary", "rational"),
)

_add(
    "clip_ratio_razor_thin_under_threshold",
    cfg(num_workers=3, param_size=1, num_shards=1, quorum_size=3, max_staleness=1,
        clip_threshold="7", learning_rate="0.1", momentum_beta="0"),
    [sub(0, 0, 7), sub(1, 0, 7), sub(2, 0, 6)],
    "Aggregate = (7+7+6)/3 = 20/3 = 6.6666..., strictly under "
    "clip_threshold=7 by exactly 1/3 — the mirror image of the "
    "razor-thin-over case. Must NOT clip (scale = 1/1) even though the "
    "decimal expansion is non-terminating and looks close to the "
    "threshold.",
    tags=("clipping", "boundary", "rational"),
)

_add(
    "negative_and_zero_cancellation_exact",
    cfg(num_workers=2, param_size=3, num_shards=1, quorum_size=2, max_staleness=1,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0"),
    [sub(0, 0, 7, -3, 0), sub(1, 0, -7, 3, 0)],
    "Two workers' contributions cancel exactly to zero at every index "
    "after aggregation (7-7=0, -3+3=0, 0+0=0). The L1 norm must be taken "
    "of the *aggregated* (already-cancelled) vector, giving l1_norm=0 and "
    "clip_scale=1/1 — not the sum of each worker's own L1 norm, which "
    "would wrongly be nonzero. Also exercises canonical 0/1 output.",
    tags=("rational", "clipping"),
)

_add(
    "adversarial_decimal_sum_boundary",
    cfg(num_workers=1, param_size=2, num_shards=1, quorum_size=1, max_staleness=1,
        clip_threshold="0.30000000000000001", learning_rate="0.1", momentum_beta="0"),
    [sub(0, 0, "0.1", "0.2")],
    "l1_norm is exactly 3/10 (0.1+0.2 exactly, as rationals); "
    "clip_threshold is exactly 3/10 plus one part in 10^17 — strictly "
    "greater than l1_norm, so the correct clip_scale is exactly 1/1. But "
    "IEEE double addition of 0.1+0.2 famously rounds *up* to "
    "0.30000000000000004, while this specific threshold string's nearest "
    "double rounds *down* to the ordinary double for 0.3 — so a "
    "double-accumulated clipper sees l1_norm > threshold and wrongly "
    "clips, producing some scale other than 1/1. Exact rational "
    "arithmetic must not.",
    tags=("rational", "clipping", "float-trap"),
)

_add(
    "large_precision_decimal_stress",
    cfg(num_workers=2, param_size=1, num_shards=1, quorum_size=2, max_staleness=1,
        clip_threshold="1000", learning_rate="0.123456789012345", momentum_beta="0.111111111111111"),
    [sub(0, 0, "1.234567890123456"), sub(1, 0, "9.876543210987654")],
    "Fifteen-digit decimal inputs for gradient, learning_rate and "
    "momentum_beta force numerators/denominators well beyond what a "
    "double or a 64-bit long can represent exactly without drift.",
    tags=("rational", "precision"),
)

_add(
    "common_factor_reduction_forces_gcd",
    cfg(num_workers=3, param_size=1, num_shards=1, quorum_size=3, max_staleness=1,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0"),
    [sub(0, 0, 4), sub(1, 0, 8), sub(2, 0, 12)],
    "Aggregate = (4+8+12)/3 = 8/1 exactly (a large common factor of 4 "
    "cancels against the quorum of 3... actually reduces cleanly to an "
    "integer), forcing the division step to reduce numerator/denominator "
    "by their gcd rather than emit an unreduced fraction like 24/3.",
    tags=("rational", "reduction"),
)

# --------------------------------------------------------------------------
# D. Whole-step poisoning at different quorum positions
# --------------------------------------------------------------------------

_add(
    "poison_last_of_quorum",
    cfg(num_workers=3, param_size=2, num_shards=1, quorum_size=3, max_staleness=1,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0"),
    [sub(0, 0, 1, 1), sub(1, 0, 2, 2), poison(2, 0)],
    "The poisoned submission is the third and final one to complete the "
    "step's quorum, arriving after two perfectly valid submissions — "
    "poisoning must still void the whole step even though it was the very "
    "last thing anyone was waiting on.",
    tags=("poison",),
)

_add(
    "poison_middle_of_larger_quorum",
    cfg(num_workers=4, param_size=2, num_shards=1, quorum_size=4, max_staleness=1,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0"),
    [sub(0, 0, 1, 1), poison(1, 0), sub(2, 0, 2, 2), sub(3, 0, 3, 3)],
    "The poisoned submission is neither first nor last of a 4-worker "
    "quorum; two valid submissions arrive after it. An any-match check "
    "over the whole chosen list is required — a check that only looks at "
    "the first or last arrival would miss this.",
    tags=("poison",),
)

_add(
    "poison_with_duplicate_from_same_worker",
    cfg(num_workers=3, param_size=1, num_shards=1, quorum_size=2, max_staleness=1,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0"),
    [poison(0, 0), sub(0, 0, 9), sub(1, 0, 1)],
    "The worker whose accepted submission is poisoned also tries to "
    "resubmit non-poisoned data to the same step; the resubmission is "
    "rejected as a duplicate (it never replaces the original slot), so "
    "the step still resolves as poisoned_skip, not as a clean commit.",
    tags=("poison", "duplicate"),
)

_add(
    "poison_after_stale_rejection_of_other_worker",
    cfg(num_workers=4, param_size=1, num_shards=1, quorum_size=2, max_staleness=1,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0"),
    [sub(3, 2, 9), sub(0, 0, 1), poison(1, 0)],
    "Worker 3's attempt at a future step is rejected stale before step 0's "
    "quorum is even formed; step 0 then resolves as poisoned via a "
    "different worker entirely. The two rejections/poisoning must be "
    "attributed independently and not interfere with each other.",
    tags=("poison", "staleness"),
)

_add(
    "poison_then_next_step_preserves_prior_state",
    cfg(num_workers=2, param_size=1, num_shards=1, quorum_size=2, max_staleness=1,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0.5"),
    [
        sub(0, 0, 4), sub(1, 0, 2),     # commits: momentum=(1-0.5)*3=1.5, params=-0.15
        poison(0, 1), sub(1, 1, 999),    # poisoned_skip: momentum/params must stay exactly as above
        sub(0, 2, 4), sub(1, 2, 2),      # next real commit must build on the PRE-poison momentum (1.5), not zero
    ],
    "Step 1 is poisoned between two real commits. Step 2's momentum EMA "
    "must be computed from step 0's momentum (1.5), completely unaffected "
    "by the poisoned step in between — proving a poisoned step neither "
    "resets nor partially blends into the optimizer state chain.",
    tags=("poison", "chain", "momentum"),
)

_add(
    "rejected_only_log_never_mutates_state",
    cfg(num_workers=2, param_size=2, num_shards=1, quorum_size=5, max_staleness=0,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0.5"),
    [sub(0, 0, 9, 9), sub(0, 0, 9, 9), sub(1, 5, 1, 1), sub(0, 1, 1, 1)],
    "Every single event in this log is rejected (duplicate, stale, stale) "
    "and quorum_size=5 is never reachable with 2 workers, so nothing ever "
    "commits. params and momentum must remain exactly all-zero.",
    tags=("chain", "rejection"),
)

# --------------------------------------------------------------------------
# E. Optimizer-state chains across many steps
# --------------------------------------------------------------------------

_add(
    "momentum_ema_chain_multistep",
    cfg(num_workers=2, param_size=1, num_shards=1, quorum_size=2, max_staleness=1,
        clip_threshold="1000", learning_rate="0.2", momentum_beta="0.6"),
    [
        sub(0, 0, 10), sub(1, 0, 0),
        sub(0, 1, 0), sub(1, 1, 10),
        sub(0, 2, 5), sub(1, 2, 5),
        sub(0, 3, 8), sub(1, 3, 2),
    ],
    "Four sequential commits with a nonzero momentum_beta and varying "
    "gradient values, so the EMA recurrence "
    "momentum[i] = beta*momentum[i] + (1-beta)*clipped[i] must compound "
    "correctly across the whole chain, not just for a single step.",
    tags=("momentum", "chain"),
)

_add(
    "quorum_size_one_immediate_commit_chain",
    cfg(num_workers=3, param_size=1, num_shards=1, quorum_size=1, max_staleness=0,
        clip_threshold="4", learning_rate="0.1", momentum_beta="0.2"),
    [sub(0, 0, 9), poison(1, 1), sub(2, 2, 3)],
    "quorum_size=1 means every step commits (or poisons) on its very "
    "first submission with no buffering at all; mixes a clip-triggering "
    "commit, an immediate poison, and a clean commit in one short chain.",
    tags=("chain", "quorum", "poison"),
)

_add(
    "quorum_size_equals_available_workers_exact",
    cfg(num_workers=5, param_size=1, num_shards=1, quorum_size=5, max_staleness=0,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0"),
    [sub(0, 0, 1), sub(1, 0, 2), sub(2, 0, 3), sub(3, 0, 4), sub(4, 0, 5)],
    "quorum_size exactly equals the number of distinct workers that ever "
    "appear in the log — there is no slack at all; losing or duplicating "
    "even one submission makes the step unreachable forever.",
    tags=("quorum",),
)

_add(
    "large_quorum_wide_fanout",
    cfg(num_workers=6, param_size=2, num_shards=2, quorum_size=6, max_staleness=0,
        clip_threshold="9", learning_rate="0.05", momentum_beta="0.1"),
    [
        sub(0, 0, 1, 6), sub(1, 0, 2, 5), sub(2, 0, 3, 4),
        sub(3, 0, 4, 3), sub(4, 0, 5, 2), sub(5, 0, 6, 1),
    ],
    "Six distinct workers all contribute to a single step's quorum; "
    "aggregation must correctly sum and average across all six before "
    "clipping and the momentum update.",
    tags=("quorum", "clipping"),
)

# --------------------------------------------------------------------------
# F. Worker-arrival permutation invariance (same math, different arrival order)
# --------------------------------------------------------------------------

_add(
    "worker_permutation_variant_a",
    cfg(num_workers=3, param_size=2, num_shards=1, quorum_size=3, max_staleness=1,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0.3"),
    [sub(0, 0, 3, 6), sub(1, 0, 9, 0), sub(2, 0, 0, 9),
     sub(0, 1, 1, 1), sub(1, 1, 2, 2), sub(2, 1, 3, 3)],
    "Baseline arrival order for the permutation-invariance property test "
    "in checks_properties.py: the same three workers contribute to steps 0 "
    "and 1 in worker-id order.",
    tags=("permutation", "property-pair"),
)

_add(
    "worker_permutation_variant_b",
    cfg(num_workers=3, param_size=2, num_shards=1, quorum_size=3, max_staleness=1,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0.3"),
    [sub(2, 0, 0, 9), sub(0, 0, 3, 6), sub(1, 0, 9, 0),
     sub(2, 1, 3, 3), sub(0, 1, 1, 1), sub(1, 1, 2, 2)],
    "Same two quorums as worker_permutation_variant_a, but each step's "
    "three submissions arrive in a different relative order (2,0,1 "
    "instead of 0,1,2) while every event's classification (accept/late/"
    "stale/duplicate/superfluous) is unchanged. final_state must be "
    "byte-identical to variant_a since aggregation is commutative — only "
    "each trace entry's 'workers' arrival-order list is allowed to differ.",
    tags=("permutation", "property-pair"),
)

# --------------------------------------------------------------------------
# G. Multi-responsibility integration ("kitchen sink") scenarios
# --------------------------------------------------------------------------

_add(
    "kitchen_sink_a_ordering_staleness_clip_rational",
    cfg(num_workers=4, param_size=4, num_shards=2, quorum_size=3, max_staleness=2,
        clip_threshold="9", learning_rate="0.1", momentum_beta="0.25"),
    [
        sub(2, 1, "0.1", "0.1", "0.1", "0.1"),        # arrives before step0 exists at all: buffered, blocked
        sub(0, 0, 1, 1, 1, 1),
        sub(3, 4, 1, 1, 1, 1),                          # frontier=0, lookahead=4 > max_staleness=2 -> stale
        sub(1, 0, 2, 2, 2, 2),
        sub(3, 0, 3, 3, 3, 3),                          # completes step0 (quorum=3) -> cascades: step1 not yet quorate
        sub(0, 1, "0.1", "0.1", "0.1", "0.1"),
        sub(1, 1, "0.1", "0.1", "0.1", "0.1"),          # completes step1: aggregate is exactly 0.1 per component,
                                                           # l1_norm exactly 0.4, well under clip_threshold=9
    ],
    "Out-of-order arrival (step1 buffered before step0 exists) + a "
    "staleness rejection that depends on the still-zero frontier + a "
    "cross-shard config (2 shards of 2) + adversarial 0.1-decimal "
    "components feeding the L1 norm, all in one trajectory whose final "
    "params/momentum depend on getting every one of those pieces right.",
    tags=("kitchen-sink",),
)

_add(
    "kitchen_sink_b_checkpoint_stale_poison_resume",
    cfg(num_workers=4, param_size=2, num_shards=1, quorum_size=2, max_staleness=1,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0.3"),
    [
        sub(3, 5, 9, 9),    # frontier=0, lookahead=5 > max_staleness=1 -> rejected stale
        sub(0, 0, 2, 2),
        poison(1, 0),        # completes step0's quorum -> poisoned_skip, no state change
        sub(2, 1, 4, 4),
        sub(3, 1, 6, 6),      # completes step1 with a clean, real commit
    ],
    "A stale rejection, a poisoned shard, and a clean commit that must "
    "resume correctly from a checkpoint taken at any point in between — "
    "this scenario is deliberately checkpoint-swept in checks_checkpoint.py "
    "(tag: kitchen-sink) to verify the resumed run's trace/rejections "
    "reconstruct byte-for-byte regardless of where the split lands.",
    tags=("kitchen-sink", "poison", "staleness"),
)

_add(
    "superfluous_worker_value_excluded_from_aggregate",
    cfg(num_workers=3, param_size=1, num_shards=1, quorum_size=2, max_staleness=1,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0"),
    [sub(0, 0, 1), sub(1, 0, 1), sub(2, 0, 1000)],
    "A third, distinct worker submits an extreme value (1000) to a step "
    "whose bucket has already reached quorum_size=2; it must be rejected "
    "as superfluous and have zero influence on the aggregate. If it leaked "
    "into the average even partially, the resulting momentum/params would "
    "be off by roughly three orders of magnitude — impossible to miss.",
    tags=("quorum", "superfluous"),
)

_add(
    "long_momentum_chain_wide_shards",
    cfg(num_workers=3, param_size=8, num_shards=4, quorum_size=3, max_staleness=1,
        clip_threshold="15", learning_rate="0.05", momentum_beta="0.35"),
    [
        sub(0, 0, 1, 2, 3, 4, 5, 6, 7, 8), sub(1, 0, 2, 1, 4, 3, 6, 5, 8, 7), sub(2, 0, 3, 3, 3, 3, 3, 3, 3, 3),
        sub(0, 1, 4, 4, 1, 1, 2, 2, 5, 5), sub(1, 1, 1, 1, 4, 4, 5, 5, 2, 2), sub(2, 1, 2, 2, 2, 2, 1, 1, 1, 1),
        sub(0, 2, 9, 0, 0, 9, 9, 0, 0, 9), sub(1, 2, 0, 9, 9, 0, 0, 9, 9, 0), sub(2, 2, 1, 1, 1, 1, 1, 1, 1, 1),
        sub(0, 3, 2, 2, 2, 2, 2, 2, 2, 2), sub(1, 3, 3, 3, 3, 3, 3, 3, 3, 3), sub(2, 3, 1, 1, 1, 1, 1, 1, 1, 1),
    ],
    "Four sequential 3-worker commits over an 8-wide, 4-shard parameter "
    "vector, with a clip threshold that bites on some steps and not "
    "others and a nonzero momentum_beta throughout — a longer, wider "
    "chain than the other momentum scenarios, so a subtly wrong EMA or "
    "shard-boundary computation has more steps to visibly drift over.",
    tags=("momentum", "chain", "sharding"),
)

_add(
    "kitchen_sink_c_uneven_config_duplicate_clip_momentum",
    cfg(num_workers=3, param_size=6, num_shards=3, quorum_size=2, max_staleness=1,
        clip_threshold="10", learning_rate="0.1", momentum_beta="0.4"),
    [
        sub(0, 0, 2, 2, 2, 2, 2, 2),
        sub(0, 0, 9, 9, 9, 9, 9, 9),   # duplicate from worker 0: rejected, must not affect aggregation
        sub(1, 0, 4, 4, 4, 4, 4, 4),   # completes step0's quorum with worker 0's ORIGINAL contribution only
        sub(2, 1, 5, 5, 5, 5, 5, 5),
        sub(0, 1, 5, 5, 5, 5, 5, 5),   # completes step1; l1_norm=30 > 10 -> global clip scale 1/3
    ],
    "Three equal shards of a 6-wide vector, a duplicate resubmission that "
    "must be dropped (not double-counted) before aggregation, and a "
    "second step whose global clip ratio (1/3) must apply uniformly "
    "across all three shards while carrying forward the first step's "
    "momentum.",
    tags=("kitchen-sink",),
)

_add(
    "kitchen_sink_d_overlapping_steps_delayed_worker",
    cfg(num_workers=4, param_size=2, num_shards=1, quorum_size=2, max_staleness=3,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0.2"),
    [
        sub(0, 2, 1, 1),                # step2 half-quorate, far ahead of frontier=0
        sub(1, 0, 2, 2),                # step0 half-quorate
        sub(2, 2, 3, 3),                # completes step2's quorum; still blocked behind step0/1
        sub(3, 0, 4, 4),                # completes step0 -> frontier=1; step2 still blocked behind step1
        sub(0, 1, 1, 1), sub(1, 1, 1, 1),  # completes step1 -> frontier=2 -> cascades step2 (already quorate)
    ],
    "Steps 0 and 2 are both partially filled before step 1 has a single "
    "submission; step 2 reaches full quorum while two full steps still "
    "sit ahead of it in the queue. Only once both earlier steps resolve "
    "does a single cascade release step 2 — with a delayed, out-of-order "
    "worker contributing to each.",
    tags=("kitchen-sink",),
)


# --------------------------------------------------------------------------
# H. Hardened input/remainder interactions
# --------------------------------------------------------------------------

_add(
    "raw_nonfinite_token_poison_cascades",
    cfg(num_workers=3, param_size=3, num_shards=2, quorum_size=2, max_staleness=2,
        clip_threshold="100", learning_rate="0.1", momentum_beta="0.5"),
    [
        sub(0, 0, "1", "2", "3"),
        sub(1, 0, "4", "NaN", "6"),  # non-finite token poisons the whole accepted contribution
        sub(0, 1, "2", "2", "2"),
        sub(2, 1, "2", "2", "2"),
    ],
    "A raw NaN token appears inside an otherwise ordinary gradient while the "
    "event's poisoned flag is false. It still occupies a quorum slot, makes "
    "step 0 a poisoned skip, advances the frontier, and allows step 1 to "
    "commit using the momentum state from before the skipped step.",
    tags=("poison", "raw-nonfinite", "sharding"),
)

_add(
    "raw_nan_token_poisons_isolated_step",
    cfg(num_workers=2, param_size=2, num_shards=1, quorum_size=2, max_staleness=0,
        clip_threshold="100", learning_rate="0.1", momentum_beta="0.5"),
    [
        sub(0, 0, "1", "2"),
        sub(1, 0, "4", "NaN"),  # bare "NaN" token, poisoned flag false: must poison alone
    ],
    "A single step, single shard, two-worker quorum with no staleness, "
    "cascading, or sharding in play — unlike "
    "raw_nonfinite_token_poison_cascades, which bundles the NaN rule with a "
    "second step, a second shard, and a frontier cascade. Inverting only "
    "the non-finite-token rule here would flip this step from "
    "poisoned_skip to either a crash or a clean commit, so this fixture "
    "pins the NaN spelling on its own.",
    tags=("poison", "raw-nonfinite"),
)

_add(
    "raw_inf_token_poisons_isolated_step",
    cfg(num_workers=2, param_size=2, num_shards=1, quorum_size=2, max_staleness=0,
        clip_threshold="100", learning_rate="0.1", momentum_beta="0.5"),
    [
        sub(0, 0, "1", "2"),
        sub(1, 0, "Inf", "3"),  # bare "Inf" token, poisoned flag false: must poison alone
    ],
    "A single step, single shard, two-worker quorum with no staleness, "
    "cascading, or sharding in play. Worker 1's gradient carries a bare "
    "'Inf' token with poisoned=false. Inverting only the non-finite-token "
    "rule (treating 'Inf' as an ordinary unparsable/parseable value) would "
    "flip this step from poisoned_skip to either a crash or a clean commit "
    "with nonzero params/momentum, so this fixture pins that rule alone.",
    tags=("poison", "raw-nonfinite"),
)

_add(
    "raw_plus_inf_token_poisons_isolated_step",
    cfg(num_workers=2, param_size=2, num_shards=1, quorum_size=2, max_staleness=0,
        clip_threshold="100", learning_rate="0.1", momentum_beta="0.5"),
    [
        sub(0, 0, "+Inf", "2"),  # explicit "+Inf" token as the first component
        sub(1, 0, "4", "3"),
    ],
    "Same isolated single-step, single-shard, no-cascade shape as the bare "
    "'Inf' case, but the poisoning token is the explicitly-signed '+Inf' "
    "spelling named in instruction.md. The token is the very first "
    "gradient component, from the worker that arrives first, so a solution "
    "that only checks non-finiteness on later components or later arrivals "
    "would still pass without this fixture.",
    tags=("poison", "raw-nonfinite"),
)

_add(
    "raw_minus_inf_token_poisons_isolated_step",
    cfg(num_workers=2, param_size=2, num_shards=1, quorum_size=2, max_staleness=0,
        clip_threshold="100", learning_rate="0.1", momentum_beta="0.5"),
    [
        sub(0, 0, "1", "2"),
        sub(1, 0, "3", "-Inf"),  # explicit "-Inf" token as the last component
    ],
    "Same isolated single-step, single-shard, no-cascade shape again, this "
    "time with the explicitly-signed '-Inf' spelling named in "
    "instruction.md, placed as the last gradient component of the second "
    "(quorum-completing) worker. Distinguishes '-Inf' handling from '+Inf' "
    "and bare 'Inf', and from a lucky first-component-only check.",
    tags=("poison", "raw-nonfinite"),
)

_add(
    "exact_fraction_scientific_inputs",
    cfg(num_workers=2, param_size=3, num_shards=1, quorum_size=2, max_staleness=1,
        clip_threshold="+5/2", learning_rate="1e-1", momentum_beta="1/+3"),
    [
        sub(0, 0, "+1/3", "2e-1", "-5E-2"),
        sub(1, 0, "2/+3", "+3e-1", "5E-2"),
        sub(0, 1, "1e+1", "-1/4", "+2.50e-1"),
        sub(1, 1, "-8", "+1/4", "-2.50e-1"),
    ],
    "Configuration and gradient numeric strings mix explicit fractions, "
    "scientific notation, and decimals. Every comparison and update must be "
    "performed on their exact rational values; parsing through double would "
    "change clipping and later momentum values.",
    tags=("rational", "input-format", "momentum"),
)

_add(
    "negative_denominator_fraction_inputs_normalize",
    cfg(num_workers=2, param_size=2, num_shards=1, quorum_size=2, max_staleness=0,
        clip_threshold="100", learning_rate="1/-10", momentum_beta="-1/-2"),
    [
        sub(0, 0, "1/-2", "-3/-4"),
        sub(1, 0, "5/-2", "1/-4"),
    ],
    "Valid p/q numeric strings may carry a sign on the denominator. This "
    "scenario puts negative denominators in both optimizer configuration and "
    "gradient values, forcing parsing, arithmetic, and every emitted rational "
    "to normalize the sign onto the numerator. A repair that reduces fractions "
    "but leaves denominators negative would emit forms such as '1/-2' and must "
    "fail even if all positive-denominator fraction tests pass.",
    tags=("rational", "input-format", "negative-denominator"),
)


_add(
    "uneven_shards_cover_entire_vector",
    cfg(num_workers=3, param_size=5, num_shards=3, quorum_size=2, max_staleness=2,
        clip_threshold="7/2", learning_rate="0.2", momentum_beta="0.25"),
    [
        sub(0, 1, "8", "1", "1", "1", "8"),
        sub(1, 1, "2", "1", "1", "1", "2"),  # future bucket becomes quorate first
        sub(1, 0, "1", "2", "3", "4", "5"),
        sub(2, 0, "5", "4", "3", "2", "1"),  # step0 -> then cascades quorate step1
    ],
    "A five-wide optimizer vector is reported through three shards. The "
    "partition must be contiguous and exhaustive with sizes 2,2,1, while "
    "clipping remains global across all five values and an out-of-order "
    "future quorum cascades only after step 0 commits.",
    tags=("sharding", "uneven-shards", "cascade", "clip"),
)

_add(
    "uneven_shards_remainder_not_confined_to_final_shard",
    cfg(num_workers=2, param_size=10, num_shards=4, quorum_size=2, max_staleness=0,
        clip_threshold="1000", learning_rate="1/10", momentum_beta="1/2"),
    [
        sub(0, 0, "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"),
        sub(1, 0, "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"),
    ],
    "A ten-wide vector over four shards has param_size % num_shards == 2, "
    "so exactly two (not num_shards - 1) leading shards must receive one "
    "extra element each: sizes 3,3,2,2. uneven_shards_cover_entire_vector "
    "above only exercises the remainder == num_shards - 1 case (5 over 3), "
    "where the correct front-loaded partition (2,2,1) happens to coincide "
    "with a plausible-but-wrong strategy of 'give every shard but the "
    "last a ceil-sized share, and let the last shard absorb whatever is "
    "left over' -- both produce 2,2,1 there. With remainder=2 here that "
    "wrong strategy instead produces 3,3,3,1 (three ceil-sized shards "
    "then a nearly-empty last one), which diverges from the correct "
    "3,3,2,2 and must be rejected.",
    tags=("sharding", "uneven-shards"),
)


_add(
    "more_shards_than_parameters_preserves_empty_tail_shard",
    cfg(num_workers=1, param_size=2, num_shards=3, quorum_size=1, max_staleness=1,
        clip_threshold="1000", learning_rate="0.1", momentum_beta="0"),
    [sub(0, 0, "7", "11")],
    "A two-element parameter vector is reported through three configured shards. "
    "The even contiguous partition is sizes 1,1,0: the first two shards receive "
    "the two remainder elements and the final configured shard remains present "
    "with empty params and momentum arrays. This isolates the valid "
    "num_shards > param_size case rather than assuming every shard is non-empty.",
    tags=("sharding", "uneven-shards", "empty-shard"),
)


SCENARIOS: List[Scenario] = list(_BASE)


# --------------------------------------------------------------------------
# Generalization variants: mechanical, independently-reverifiable transforms
# of a few scenarios above, so no test can pass by hardcoding a specific
# worker-id namespace or shard topology.
# --------------------------------------------------------------------------

def _remap_workers(scenario: Scenario, suffix: str, offset: int) -> Scenario:
    new_events = []
    for event in scenario.events:
        moved = dict(event)
        moved["worker_id"] = event["worker_id"] + offset
        new_events.append(moved)
    return Scenario(
        f"{scenario.name}__{suffix}",
        dict(scenario.config),
        new_events,
        scenario.doc + f" [generalization variant: worker ids offset by {offset}]",
        scenario.tags + ("generalization",),
    )


def _find(name: str) -> Scenario:
    for scenario in _BASE:
        if scenario.name == name:
            return scenario
    raise KeyError(name)


_GENERALIZATION_VARIANTS: List[Scenario] = [
    _remap_workers(_find("deep_out_of_order_cascade"), "worker_offset_100", 100),
    _remap_workers(_find("kitchen_sink_a_ordering_staleness_clip_rational"), "worker_offset_50", 50),
    _remap_workers(_find("staleness_boundary_chain_shifting_frontier"), "worker_offset_77", 77),
]

SCENARIOS.extend(_GENERALIZATION_VARIANTS)


def _with_shard_count(scenario: Scenario, suffix: str, num_shards: int) -> Scenario:
    assert scenario.config["param_size"] % num_shards == 0, (
        f"{scenario.name}: param_size not divisible by num_shards={num_shards}"
    )
    new_config = dict(scenario.config)
    new_config["num_shards"] = num_shards
    return Scenario(
        f"{scenario.name}__{suffix}",
        new_config,
        scenario.events,
        scenario.doc + f" [generalization variant: num_shards={num_shards}]",
        scenario.tags + ("generalization", "shard-count-variant"),
    )


# Same gradient/clip math as dominant_shard_norm_cross_shard_clip, reported
# through a single shard and through four single-element shards, to back a
# property test that shard count never changes the numeric outcome.
_SHARD_COUNT_VARIANTS: List[Scenario] = [
    _with_shard_count(_find("dominant_shard_norm_cross_shard_clip"), "shards1", 1),
    _with_shard_count(_find("dominant_shard_norm_cross_shard_clip"), "shards4", 4),
]

SCENARIOS.extend(_SHARD_COUNT_VARIANTS)


def by_name(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(name)


def with_tag(tag: str) -> List[Scenario]:
    return [s for s in SCENARIOS if tag in s.tags]
