"""Shared plumbing for the test suite: compiling the agent's copied
/app/src, invoking the compiled CLI, and memoizing subprocess runs so the
same (config, events, checkpoint-in) triple is never executed twice in one
test session. This module is imported by the test_*.py files; it is not
itself collected as a test.
"""

import hashlib
import json
import os
import re
import subprocess
from math import gcd
from pathlib import Path

APP_SRC = Path("/app/src/main/java")

# Everything the agent's code touches lives under WORK_ROOT: the classes javac
# produces, and the scratch files each CLI invocation reads and writes. /tests
# (the suite, the oracle, the golden fixtures) and /logs/verifier (the reward
# channel) are root-owned and unreachable from there.
WORK_ROOT = Path("/work")
APP_OUT = WORK_ROOT / "out"

# The pytest session runs as root and owns the expectations. javac and every
# java process it launches run as this unprivileged uid instead, so agent-
# produced code cannot read the goldens, cannot write the reward channel, and
# cannot leave behind a process that outlives the session with the privileges
# to do either. tests/Dockerfile creates the uid and gives it /work.
GRADED_UID = 10002
GRADED_GID = 10002

# Scratch files are written by root and read by the graded uid, so nothing here
# depends on the inherited umask.
os.umask(0)

FRACTION_RE = re.compile(r"^-?\d+/[1-9]\d*$")


def privilege_dropping_available() -> bool:
    """Whether this session can launch agent code as a different, unprivileged
    user. True in the verifier image, which runs the session as root."""
    return hasattr(os, "geteuid") and os.geteuid() == 0


def _setpriv_prefix():
    """Prefix used for every candidate-controlled process in the verifier.

    Numeric IDs avoid account-name ambiguity, clear supplementary groups,
    and no_new_privs prevents privilege regain through setuid/setgid files.
    """
    if not privilege_dropping_available():
        return []
    return [
        "setpriv",
        f"--reuid={GRADED_UID}",
        f"--regid={GRADED_GID}",
        "--clear-groups",
        "--no-new-privs",
        "--",
    ]


def _shared_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o777)
    except OSError:
        pass
    return path


def _make_app_readable():
    """Adds read (and, for directories, search) permission for everyone under
    /app, removing nothing. The graded uid has to read the sources it
    compiles. Skips anything that is itself a symlink: os.walk still lists a
    symlinked directory in dirnames even with followlinks=False, and
    os.chmod follows symlinks by default, so a candidate-planted symlink
    under /app (e.g. pointing at /tests) would otherwise widen permissions
    on its target instead of on /app."""
    if not privilege_dropping_available():
        return
    for current, dirnames, filenames in os.walk("/app", followlinks=False):
        for name in [current] + [os.path.join(current, d) for d in dirnames]:
            if os.path.islink(name):
                continue
            try:
                os.chmod(name, os.stat(name).st_mode | 0o555)
            except OSError:
                pass
        for name in filenames:
            target = os.path.join(current, name)
            if os.path.islink(target):
                continue
            try:
                os.chmod(target, os.stat(target).st_mode | 0o444)
            except OSError:
                pass


def _graded_env():
    """Environment for candidate-controlled processes, with the JVM's
    implicit option channels removed. JAVA_TOOL_OPTIONS, _JAVA_OPTIONS and
    JDK_JAVA_OPTIONS are picked up by every `java`/`javac` launch without
    appearing on the command line, so an inherited value could change
    locale, encoding, or arithmetic-adjacent flags between runs of an
    otherwise identical submission."""
    env = dict(os.environ)
    for name in ("JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS"):
        env.pop(name, None)
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    return env


def _run_as_graded(args, **kwargs):
    """Runs args as the unprivileged uid with no_new_privs, out of /work,
    never inheriting the session's working directory inside /tests."""
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("env", _graded_env())
    command = [*_setpriv_prefix(), *args]
    return subprocess.run(
        command,
        cwd=str(_shared_dir(WORK_ROOT)),
        **kwargs,
    )


def compile_project():
    sources = [str(p) for p in APP_SRC.rglob("*.java")]
    _shared_dir(WORK_ROOT)
    _shared_dir(APP_OUT)
    _make_app_readable()
    return _run_as_graded(["javac", "-d", str(APP_OUT), *sources])


COMPILE_RESULT = compile_project()

_CACHE_DIR = _shared_dir(WORK_ROOT / "replay-cache")
_RESULT_CACHE = {}
_UNCACHED_CALL_COUNTER = [0]


def run_main(*args, jvm_args=()):
    return _run_as_graded(
        ["java", *jvm_args, "-cp", str(APP_OUT), "com.trainingsim.Main", *args]
    )


def run_probe(command: str):
    """Runs a shell command as the graded uid, the same way the compiled agent
    code is run, so a test can check what that code can actually reach."""
    return _run_as_graded(["sh", "-c", command])


def _digest(*parts) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(json.dumps(part, sort_keys=True).encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:24]


def _require_exact_keys(value: dict, keys: set[str], path: str):
    assert type(value) is dict, f"{path} must be a JSON object"
    assert set(value) == keys, f"{path} keys must be {sorted(keys)}, got {sorted(value)}"


def _require_int(value, path: str):
    # `bool` is a subclass of int in Python, and 0.0 == 0.  The output
    # contract is native JSON integer, so use exact type identity here.
    assert type(value) is int, f"{path} must be a JSON integer, got {type(value).__name__}"


def _require_bool(value, path: str):
    assert type(value) is bool, f"{path} must be a JSON boolean, got {type(value).__name__}"


def _require_str(value, path: str):
    assert type(value) is str, f"{path} must be a JSON string, got {type(value).__name__}"


def _require_list(value, path: str):
    assert type(value) is list, f"{path} must be a JSON array, got {type(value).__name__}"


def _require_step_key(key: str, path: str):
    _require_str(key, path)
    assert re.fullmatch(r"0|[1-9]\d*", key), f"{path} must be a canonical non-negative decimal step key"


def _validate_rejection(value, path: str):
    _require_exact_keys(value, {"worker_id", "step", "reason"}, path)
    _require_int(value["worker_id"], f"{path}.worker_id")
    _require_int(value["step"], f"{path}.step")
    _require_str(value["reason"], f"{path}.reason")


def _validate_rejection_list(value, path: str):
    _require_list(value, path)
    for i, rejection in enumerate(value):
        _validate_rejection(rejection, f"{path}[{i}]")


def _require_canonical_rational(value, path: str):
    """Every rational the program emits, anywhere in --out or
    --checkpoint-out, has to be `numerator/denominator` in lowest terms with
    a strictly positive denominator. Checked here rather than only in the
    curated comparisons so that a checkpoint field whose value happens to be
    arithmetically right but textually non-canonical (a retained `1/-2`
    input lexeme in a pending bucket, an unreduced `2/10`, a bare `1`) fails
    on any run that produces it, not only on the splits some test happens to
    sample.
    """
    _require_str(value, path)
    assert FRACTION_RE.match(value), f"{path} must be a canonical p/q rational, got {value!r}"
    numerator, denominator = value.split("/")
    assert gcd(abs(int(numerator)), int(denominator)) == 1, (
        f"{path} must be reduced to lowest terms, got {value!r}"
    )
    assert int(numerator) != 0 or value == "0/1", (
        f"{path} must render zero as 0/1, got {value!r}"
    )


def _validate_rational_list(value, path: str):
    _require_list(value, path)
    for i, component in enumerate(value):
        _require_canonical_rational(component, f"{path}[{i}]")


def validate_output_schema(output: dict):
    """Rejects schema-invalid --out JSON before semantic comparisons.

    In particular, Python's equality intentionally aliases False with 0,
    True with 1, and integral floats with ints.  Grading must not inherit
    that coercion for fields whose Java/JSON schema is integer-valued.
    """
    _require_exact_keys(output, {"final_state", "trace", "uncommitted_pending", "rejections"}, "output")

    final_state = output["final_state"]
    _require_exact_keys(final_state, {"shards"}, "output.final_state")
    shards = final_state["shards"]
    _require_list(shards, "output.final_state.shards")
    for i, shard in enumerate(shards):
        path = f"output.final_state.shards[{i}]"
        _require_exact_keys(shard, {"shard_id", "params", "momentum"}, path)
        _require_int(shard["shard_id"], f"{path}.shard_id")
        _validate_rational_list(shard["params"], f"{path}.params")
        _validate_rational_list(shard["momentum"], f"{path}.momentum")

    trace = output["trace"]
    _require_list(trace, "output.trace")
    for i, entry in enumerate(trace):
        path = f"output.trace[{i}]"
        _require_exact_keys(
            entry,
            {"step", "status", "workers", "l1_norm", "clip_scale", "rejections"},
            path,
        )
        _require_int(entry["step"], f"{path}.step")
        _require_str(entry["status"], f"{path}.status")
        _require_list(entry["workers"], f"{path}.workers")
        for j, worker_id in enumerate(entry["workers"]):
            _require_int(worker_id, f"{path}.workers[{j}]")
        for field in ("l1_norm", "clip_scale"):
            if entry[field] is not None:
                _require_canonical_rational(entry[field], f"{path}.{field}")
        _validate_rejection_list(entry["rejections"], f"{path}.rejections")

    pending = output["uncommitted_pending"]
    assert type(pending) is dict, "output.uncommitted_pending must be a JSON object"
    for step, workers in pending.items():
        _require_step_key(step, "output.uncommitted_pending key")
        _require_list(workers, f"output.uncommitted_pending[{step!r}]")
        for i, worker_id in enumerate(workers):
            _require_int(worker_id, f"output.uncommitted_pending[{step!r}][{i}]")

    _validate_rejection_list(output["rejections"], "output.rejections")


def validate_checkpoint_schema(checkpoint: dict):
    """Rejects schema-invalid --checkpoint-out JSON before comparisons."""
    _require_exact_keys(
        checkpoint,
        {"committed_step", "params", "momentum", "pending", "rejections", "all_rejections"},
        "checkpoint",
    )
    _require_int(checkpoint["committed_step"], "checkpoint.committed_step")
    _validate_rational_list(checkpoint["params"], "checkpoint.params")
    _validate_rational_list(checkpoint["momentum"], "checkpoint.momentum")

    pending = checkpoint["pending"]
    assert type(pending) is dict, "checkpoint.pending must be a JSON object"
    for step, bucket in pending.items():
        _require_step_key(step, "checkpoint.pending key")
        _require_list(bucket, f"checkpoint.pending[{step!r}]")
        for i, event in enumerate(bucket):
            path = f"checkpoint.pending[{step!r}][{i}]"
            _require_exact_keys(event, {"worker_id", "step", "poisoned", "gradient"}, path)
            _require_int(event["worker_id"], f"{path}.worker_id")
            _require_int(event["step"], f"{path}.step")
            _require_bool(event["poisoned"], f"{path}.poisoned")
            # The normalized pending shape couples the two fields: a poisoned
            # contribution has no gradient to average, so retaining the values
            # the input happened to supply (or that were parsed before the
            # poison was noticed) is a schema violation, not a harmless extra.
            if event["poisoned"]:
                assert event["gradient"] is None, (
                    f"{path}.gradient must be null when poisoned is true, got "
                    f"{event['gradient']!r}"
                )
            else:
                assert event["gradient"] is not None, (
                    f"{path}.gradient must be present when poisoned is false"
                )
                _validate_rational_list(event["gradient"], f"{path}.gradient")

    rejections = checkpoint["rejections"]
    assert type(rejections) is dict, "checkpoint.rejections must be a JSON object"
    for step, bucket in rejections.items():
        _require_step_key(step, "checkpoint.rejections key")
        _validate_rejection_list(bucket, f"checkpoint.rejections[{step!r}]")

    _validate_rejection_list(checkpoint["all_rejections"], "checkpoint.all_rejections")


def run_cli_with_checkpoint(config: dict, events: list, checkpoint_in: dict = None):
    """Runs one CLI invocation over in-memory (config, events[, checkpoint_in])
    and returns (out_json, checkpoint_out_json). Memoized: repeated calls
    with equal content reuse the previous subprocess's result instead of
    re-invoking `java`."""
    key = _digest(config, events, checkpoint_in)
    if key in _RESULT_CACHE:
        return _RESULT_CACHE[key]

    config_path = _CACHE_DIR / f"{key}.config.json"
    log_path = _CACHE_DIR / f"{key}.log.json"
    out_path = _CACHE_DIR / f"{key}.out.json"
    ckpt_out_path = _CACHE_DIR / f"{key}.ckpt_out.json"
    config_path.write_text(json.dumps(config))
    log_path.write_text(json.dumps(events))

    args = [
        "--config", str(config_path),
        "--log", str(log_path),
        "--out", str(out_path),
        "--checkpoint-out", str(ckpt_out_path),
    ]
    if checkpoint_in is not None:
        ckpt_in_path = _CACHE_DIR / f"{key}.ckpt_in.json"
        ckpt_in_path.write_text(json.dumps(checkpoint_in))
        args += ["--checkpoint-in", str(ckpt_in_path)]

    result = run_main(*args)
    assert result.returncode == 0, result.stderr

    output = json.loads(_read_no_follow(out_path))
    checkpoint = json.loads(_read_no_follow(ckpt_out_path))
    validate_output_schema(output)
    validate_checkpoint_schema(checkpoint)
    _RESULT_CACHE[key] = (output, checkpoint)
    return output, checkpoint


def run_cli(config: dict, events: list, checkpoint_in: dict = None) -> dict:
    output, _ = run_cli_with_checkpoint(config, events, checkpoint_in)
    return output


def run_cli_raw_text_uncached(config: dict, events: list, jvm_args=()) -> str:
    """Always spawns a fresh `java` process (bypassing the memoization
    cache) and returns the raw --out file text, for determinism checks that
    must actually re-execute the program rather than reuse a cached
    result. `jvm_args` are passed to the JVM itself, so a caller can pin an
    ambient setting (a default locale, say) and check that the graded output
    does not move with it."""
    _UNCACHED_CALL_COUNTER[0] += 1
    key = _digest(config, events, "raw-uncached", _UNCACHED_CALL_COUNTER[0])
    config_path = _CACHE_DIR / f"{key}.config.json"
    log_path = _CACHE_DIR / f"{key}.log.json"
    out_path = _CACHE_DIR / f"{key}.out.json"
    config_path.write_text(json.dumps(config))
    log_path.write_text(json.dumps(events))
    result = run_main(
        "--config", str(config_path),
        "--log", str(log_path),
        "--out", str(out_path),
        jvm_args=jvm_args,
    )
    assert result.returncode == 0, result.stderr
    return _read_no_follow(out_path)


def run_checkpoint_split(config: dict, events: list, split: int):
    """Splits `events` at `split`, runs the head with --checkpoint-out and
    the tail with --checkpoint-in, and returns
    (resumed_out_json, head_out_json, head_checkpoint_json, resumed_checkpoint_json)."""
    key = _digest(config, events, split, "split")
    if key in _RESULT_CACHE:
        return _RESULT_CACHE[key]
    head, tail = events[:split], events[split:]
    head_out, head_checkpoint = run_cli_with_checkpoint(config, head)
    resumed_out, resumed_checkpoint = run_cli_with_checkpoint(config, tail, checkpoint_in=head_checkpoint)
    result = (resumed_out, head_out, head_checkpoint, resumed_checkpoint)
    _RESULT_CACHE[key] = result
    return result


def _read_no_follow(path: Path) -> str:
    """Reads a file the graded process just wrote by a fixed, predictable
    name inside the world-writable cache directory. Refuses to follow it if
    it is a symlink: the compiled agent CLI is the only expected writer of
    that name, but nothing stops candidate code from instead creating a
    symlink there pointing at a root-owned fixture, which this (root)
    session would otherwise transparently dereference on read."""
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r") as handle:
        return handle.read()


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def assert_canonical_fraction(value: str):
    """Same canonical-form rule the schema validators apply, exposed for
    tests that want to check a hand-picked value directly."""
    _require_canonical_rational(value, "value")


def all_numeric_values(output: dict):
    values = []
    for shard in output["final_state"]["shards"]:
        values.extend(shard["params"])
        values.extend(shard["momentum"])
    for entry in output["trace"]:
        if entry["l1_norm"] is not None:
            values.append(entry["l1_norm"])
            values.append(entry["clip_scale"])
    return values
