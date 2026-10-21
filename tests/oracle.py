"""Independent reference implementation of the candidate-visible replay contract.

This module never imports, subprocesses, or otherwise depends on anything
under /app/src, solution/src, or environment/app/src. It is a from-scratch
re-derivation of the replay algorithm in Python, using fractions.Fraction
for all numeric state (never float), so that tests can compute the
*expected* output of an arbitrary, freshly generated (config, event log)
pair without needing a hand-captured golden file for it.

Every test in this suite that calls into this module is checking the actual
Java program's behavior against this independent computation, never against
itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional


def parse_rational(text: str) -> Fraction:
    """Parse exact decimal, scientific-notation, or explicit p/q text.

    Fraction's string parser rejects a signed denominator such as ``1/-2``
    even though the candidate contract allows p/q numeric strings. Parse the
    two integer components explicitly so the oracle covers that valid input
    and lets Fraction normalize the sign canonically.
    """
    text = str(text).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    return Fraction(text)


# The four raw non-finite gradient tokens the contract names, matched
# literally. Case folding here would define a wider poison rule than the
# contract states (and, on the Java side, would make detection depend on the
# JVM's default locale), so both implementations compare exact spellings.
NON_FINITE_TOKENS = frozenset({"NaN", "Inf", "+Inf", "-Inf"})


def is_non_finite(text: str) -> bool:
    return text.strip() in NON_FINITE_TOKENS


def rstr(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True)
class Config:
    num_workers: int
    param_size: int
    num_shards: int
    quorum_size: int
    max_staleness: int
    clip_threshold: Fraction
    learning_rate: Fraction
    momentum_beta: Fraction

    @property
    def shard_size(self) -> int:
        return self.param_size // self.num_shards

    def shard_bounds(self, shard_id: int):
        base, remainder = divmod(self.param_size, self.num_shards)
        start = shard_id * base + min(shard_id, remainder)
        end = start + base + (1 if shard_id < remainder else 0)
        return start, end

    @staticmethod
    def from_json(d: dict) -> "Config":
        return Config(
            num_workers=int(d.get("num_workers", 0)),
            param_size=int(d["param_size"]),
            num_shards=int(d["num_shards"]),
            quorum_size=int(d["quorum_size"]),
            max_staleness=int(d["max_staleness"]),
            clip_threshold=parse_rational(d["clip_threshold"]),
            learning_rate=parse_rational(d["learning_rate"]),
            momentum_beta=parse_rational(d["momentum_beta"]),
        )


@dataclass
class Event:
    worker_id: int
    step: int
    poisoned: bool
    gradient: Optional[List[Fraction]]

    @staticmethod
    def from_json(d: dict) -> "Event":
        poisoned = bool(d.get("poisoned", False))
        gradient = None
        if not poisoned:
            raw = [str(g) for g in d["gradient"]]
            poisoned = any(is_non_finite(g) for g in raw)
            if not poisoned:
                gradient = [parse_rational(g) for g in raw]
        return Event(int(d["worker_id"]), int(d["step"]), poisoned, gradient)

    def to_json(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "step": self.step,
            "poisoned": self.poisoned,
            "gradient": None if self.poisoned else [rstr(g) for g in self.gradient],
        }

    @staticmethod
    def from_checkpoint_json(d: dict) -> "Event":
        poisoned = bool(d.get("poisoned", False))
        gradient = None
        if not poisoned and d.get("gradient") is not None:
            raw = [str(g).strip() for g in d["gradient"]]
            poisoned = any(is_non_finite(g) for g in raw)
            if not poisoned:
                gradient = [parse_rational(g) for g in raw]
        return Event(int(d["worker_id"]), int(d["step"]), poisoned, gradient)


@dataclass
class Rejection:
    worker_id: int
    step: int
    reason: str

    def to_json(self) -> dict:
        return {"worker_id": self.worker_id, "step": self.step, "reason": self.reason}

    @staticmethod
    def from_json(d: dict) -> "Rejection":
        return Rejection(int(d["worker_id"]), int(d["step"]), d["reason"])


@dataclass
class TraceEntry:
    step: int
    status: str
    workers: List[int]
    l1_norm: Optional[Fraction]
    clip_scale: Optional[Fraction]
    rejections: List[Rejection]

    def to_json(self) -> dict:
        return {
            "step": self.step,
            "status": self.status,
            "workers": list(self.workers),
            "l1_norm": None if self.l1_norm is None else rstr(self.l1_norm),
            "clip_scale": None if self.clip_scale is None else rstr(self.clip_scale),
            "rejections": [r.to_json() for r in self.rejections],
        }


@dataclass
class State:
    params: List[Fraction]
    momentum: List[Fraction]
    committed_step: int = -1
    pending: Dict[int, List[Event]] = field(default_factory=dict)
    rejections_by_step: Dict[int, List[Rejection]] = field(default_factory=dict)
    all_rejections: List[Rejection] = field(default_factory=list)

    @staticmethod
    def fresh(param_size: int) -> "State":
        return State(
            params=[Fraction(0)] * param_size,
            momentum=[Fraction(0)] * param_size,
        )

    @property
    def frontier(self) -> int:
        return self.committed_step + 1

    def to_checkpoint_json(self) -> dict:
        return {
            "committed_step": self.committed_step,
            "params": [rstr(p) for p in self.params],
            "momentum": [rstr(m) for m in self.momentum],
            "pending": {
                str(step): [e.to_json() for e in bucket]
                for step, bucket in self.pending.items()
            },
            "rejections": {
                str(step): [r.to_json() for r in lst]
                for step, lst in self.rejections_by_step.items()
            },
            "all_rejections": [r.to_json() for r in self.all_rejections],
        }

    @staticmethod
    def from_checkpoint_json(d: dict, param_size: int) -> "State":
        state = State.fresh(param_size)
        state.committed_step = int(d["committed_step"])
        state.params = [parse_rational(p) for p in d["params"]]
        state.momentum = [parse_rational(m) for m in d["momentum"]]
        for step_str, bucket in d["pending"].items():
            state.pending[int(step_str)] = [Event.from_checkpoint_json(e) for e in bucket]
        for step_str, lst in d["rejections"].items():
            state.rejections_by_step[int(step_str)] = [Rejection.from_json(r) for r in lst]
        state.all_rejections = [Rejection.from_json(r) for r in d["all_rejections"]]
        return state


def classify(committed_step: int, target_step: int, max_staleness: int) -> str:
    if target_step <= committed_step:
        return "late"
    frontier = committed_step + 1
    lookahead = target_step - frontier
    if lookahead > max_staleness:
        return "stale"
    return "accept"


def _commit(step: int, chosen: List[Event], config: Config, state: State) -> TraceEntry:
    workers = [e.worker_id for e in chosen]
    poisoned = any(e.poisoned for e in chosen)
    rejections = list(state.rejections_by_step.get(step, []))

    if poisoned:
        return TraceEntry(step, "poisoned_skip", workers, None, None, rejections)

    aggregated = [Fraction(0)] * config.param_size
    for event in chosen:
        for i in range(config.param_size):
            aggregated[i] += event.gradient[i]
    quorum = Fraction(config.quorum_size)
    aggregated = [a / quorum for a in aggregated]

    l1_norm = Fraction(0)
    for a in aggregated:
        l1_norm += abs(a)
    if l1_norm <= config.clip_threshold:
        scale = Fraction(1)
    else:
        scale = config.clip_threshold / l1_norm
    clipped = [a * scale for a in aggregated]

    for i in range(config.param_size):
        retained = config.momentum_beta * state.momentum[i]
        incoming = (Fraction(1) - config.momentum_beta) * clipped[i]
        new_momentum = retained + incoming
        state.momentum[i] = new_momentum
        state.params[i] = state.params[i] - config.learning_rate * new_momentum

    return TraceEntry(step, "committed", workers, l1_norm, scale, rejections)


def _cascade(config: Config, state: State, trace_out: List[TraceEntry]) -> None:
    while True:
        frontier = state.frontier
        bucket = state.pending.get(frontier)
        if bucket is None or len(bucket) < config.quorum_size:
            break
        trace_out.append(_commit(frontier, bucket, config, state))
        del state.pending[frontier]
        state.rejections_by_step.pop(frontier, None)
        state.committed_step = frontier


def ingest(event: Event, config: Config, state: State, trace_out: List[TraceEntry]) -> None:
    decision = classify(state.committed_step, event.step, config.max_staleness)

    if decision == "accept":
        bucket = state.pending.setdefault(event.step, [])
        if any(existing.worker_id == event.worker_id for existing in bucket):
            decision = "duplicate"
        elif len(bucket) >= config.quorum_size:
            decision = "superfluous"
        else:
            bucket.append(event)

    if decision != "accept":
        rejection = Rejection(event.worker_id, event.step, decision)
        state.rejections_by_step.setdefault(event.step, []).append(rejection)
        state.all_rejections.append(rejection)

    _cascade(config, state, trace_out)


def replay(config: Config, events: List[Event], initial_state: Optional[State] = None):
    """Runs the whole log against a state, returning (final_state, trace)."""
    state = initial_state if initial_state is not None else State.fresh(config.param_size)
    trace: List[TraceEntry] = []
    if initial_state is not None:
        _cascade(config, state, trace)
    for event in events:
        ingest(event, config, state, trace)
    return state, trace


def to_output_json(config: Config, state: State, trace: List[TraceEntry]) -> dict:
    shards = []
    for shard_id in range(config.num_shards):
        start, end = config.shard_bounds(shard_id)
        shards.append({
            "shard_id": shard_id,
            "params": [rstr(p) for p in state.params[start:end]],
            "momentum": [rstr(m) for m in state.momentum[start:end]],
        })
    uncommitted = {
        str(step): [e.worker_id for e in bucket]
        for step, bucket in state.pending.items()
    }
    return {
        "final_state": {"shards": shards},
        "trace": [t.to_json() for t in trace],
        "uncommitted_pending": uncommitted,
        "rejections": [r.to_json() for r in state.all_rejections],
    }


def run(config_json: dict, events_json: List[dict], checkpoint_in_json: Optional[dict] = None) -> dict:
    """End-to-end equivalent of one `Main` invocation: parse config/events
    (and an optional inbound checkpoint), replay, and return the exact
    `--out` JSON structure."""
    config = Config.from_json(config_json)
    events = [Event.from_json(e) for e in events_json]
    initial_state = None
    if checkpoint_in_json is not None:
        initial_state = State.from_checkpoint_json(checkpoint_in_json, config.param_size)
    state, trace = replay(config, events, initial_state)
    return to_output_json(config, state, trace)


def classify_each_event(config_json: dict, events_json: List[dict]) -> List[str]:
    """Returns, for each event in arrival order, the admission decision it
    actually received ("accept", "late", "stale", "duplicate", or
    "superfluous") when replayed against every event before it. Used to
    build "what if the rejected events had simply never been sent" variants
    of a scenario without guessing which index was rejected by hand."""
    config = Config.from_json(config_json)
    events = [Event.from_json(e) for e in events_json]
    state = State.fresh(config.param_size)
    decisions = []
    trace: List[TraceEntry] = []
    for event in events:
        decision = classify(state.committed_step, event.step, config.max_staleness)
        if decision == "accept":
            bucket = state.pending.setdefault(event.step, [])
            if any(existing.worker_id == event.worker_id for existing in bucket):
                decision = "duplicate"
            elif len(bucket) >= config.quorum_size:
                decision = "superfluous"
            else:
                bucket.append(event)
        if decision != "accept":
            rejection = Rejection(event.worker_id, event.step, decision)
            state.rejections_by_step.setdefault(event.step, []).append(rejection)
            state.all_rejections.append(rejection)
        decisions.append(decision)
        _cascade(config, state, trace)
    return decisions


def interesting_split_points(config_json: dict, events_json: List[dict]) -> Dict[str, List[int]]:
    """Replays the log once, recording *why* each event's arrival index is a
    structurally meaningful place to checkpoint, so tests can pick split
    points that land on a real protocol boundary
    instead of guessing indices by hand for every new scenario.

    Returned indices are "events consumed so far" counts — i.e. a valid
    split index for `events[:k]` / `events[k:]` — not raw event positions.
    """
    config = Config.from_json(config_json)
    events = [Event.from_json(e) for e in events_json]
    state = State.fresh(config.param_size)

    after_rejection: List[int] = []
    after_poisoned_commit: List[int] = []
    after_any_commit: List[int] = []
    bucket_just_quorate_but_blocked: List[int] = []

    for idx, event in enumerate(events):
        trace_before = []
        before_rejections = len(state.all_rejections)
        decision = classify(state.committed_step, event.step, config.max_staleness)
        if decision == "accept":
            bucket = state.pending.setdefault(event.step, [])
            if any(existing.worker_id == event.worker_id for existing in bucket):
                decision = "duplicate"
            elif len(bucket) >= config.quorum_size:
                decision = "superfluous"
            else:
                bucket.append(event)
                if len(bucket) == config.quorum_size and event.step != state.frontier:
                    bucket_just_quorate_but_blocked.append(idx + 1)
        if decision != "accept":
            rejection = Rejection(event.worker_id, event.step, decision)
            state.rejections_by_step.setdefault(event.step, []).append(rejection)
            state.all_rejections.append(rejection)
        _cascade(config, state, trace_before)

        if len(state.all_rejections) > before_rejections:
            after_rejection.append(idx + 1)
        for trace_entry in trace_before:
            after_any_commit.append(idx + 1)
            if trace_entry.status == "poisoned_skip":
                after_poisoned_commit.append(idx + 1)

    all_points = sorted(set(
        [0, len(events)]
        + after_rejection
        + after_poisoned_commit
        + after_any_commit
        + bucket_just_quorate_but_blocked
    ))
    return {
        "after_rejection": after_rejection,
        "after_poisoned_commit": after_poisoned_commit,
        "after_any_commit": after_any_commit,
        "bucket_just_quorate_but_blocked": bucket_just_quorate_but_blocked,
        "all": all_points,
    }


def run_with_checkpoint(config_json: dict, events_json: List[dict], checkpoint_in_json: Optional[dict] = None):
    """Same as `run`, but also returns the checkpoint JSON a
    `--checkpoint-out` at exactly this point would contain."""
    config = Config.from_json(config_json)
    events = [Event.from_json(e) for e in events_json]
    initial_state = None
    if checkpoint_in_json is not None:
        initial_state = State.from_checkpoint_json(checkpoint_in_json, config.param_size)
    state, trace = replay(config, events, initial_state)
    return to_output_json(config, state, trace), state.to_checkpoint_json()
