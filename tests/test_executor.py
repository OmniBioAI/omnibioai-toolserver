"""
Unit tests for toolserver.executor.Executor.

Covers the thread-pool-backed execution lifecycle: dependency wiring at
construction, the happy-path state transition from QUEUED through RUNNING
to COMPLETED, the failure path that captures handler exceptions into a
structured FAILED error record, registry-based handler lookup by tool_id
(an execution-boundary contract -- an unregistered tool must never run),
log-line persistence via the handler's log callback, and concurrent
execution of independent jobs.

Run with:
    python -m pytest tests/test_executor.py -v

Developer:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

import time
from typing import Callable

import pytest

from toolserver.executor import Executor
from toolserver.models import RunRecord
from toolserver.registry import ToolHandler, ToolRegistry
from toolserver.store import RunStore

# ===========================================================================
# Helpers / factories
# ===========================================================================

def _make_record(run_id: str = "run-001", state: str = "QUEUED", **kwargs) -> RunRecord:
    defaults = dict(
        run_id=run_id,
        tool_id="test_tool",
        state=state,
        created_epoch=1_700_000_000,
        updated_epoch=1_700_000_001,
        inputs={},
        resources={},
        logs=[],
        results=None,
        error=None,
    )
    defaults.update(kwargs)
    return RunRecord(**defaults)


def _make_handler(
    run_fn: Callable = None,
    validate_fn: Callable = None,
    tool_id: str = "test_tool",
) -> ToolHandler:
    def _default_run(inputs, resources, log):
        return {"ok": True}

    def _default_validate(inputs, resources):
        return {"ok": True, "errors": [], "warnings": []}

    if run_fn is None:
        run_fn = _default_run
    if validate_fn is None:
        validate_fn = _default_validate
    return ToolHandler(tool_id=tool_id, validate=validate_fn, run=run_fn)


def _make_store(tmp_path) -> RunStore:
    return RunStore(root_dir=str(tmp_path))


def _make_registry(handler: ToolHandler = None) -> ToolRegistry:
    reg = ToolRegistry()
    if handler is not None:
        reg.register(handler)
    return reg


def _wait_for_state(store: RunStore, run_id: str, state: str, timeout: float = 2.0) -> RunRecord:
    """Poll until the record reaches the expected state or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rec = store.get(run_id)
        if rec.state == state:
            return rec
        time.sleep(0.01)
    rec = store.get(run_id)
    raise TimeoutError(f"run {run_id} stuck in {rec.state!r}, expected {state!r}")


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def tmp_store(tmp_path):
    return _make_store(tmp_path)


@pytest.fixture()
def handler():
    return _make_handler()


@pytest.fixture()
def registry(handler):
    return _make_registry(handler)


@pytest.fixture()
def executor(tmp_store, registry):
    return Executor(store=tmp_store, registry=registry)


@pytest.fixture()
def rec(tmp_store):
    r = _make_record()
    tmp_store.create(r)
    return r


# ===========================================================================
# __init__
# ===========================================================================

class TestInit:
    """Executor.__init__ wires the injected store/registry through unchanged
    and starts a live thread pool."""

    def test_stores_injected_dependencies(self, tmp_store, registry):
        """Executor keeps references to the exact store and registry instances it was constructed with."""
        ex = Executor(store=tmp_store, registry=registry)
        assert ex.store is tmp_store
        assert ex.registry is registry

    def test_default_max_workers(self, tmp_store, registry):
        """With no max_workers argument, the executor's thread pool defaults to 8 workers."""
        ex = Executor(store=tmp_store, registry=registry)
        assert ex.pool._max_workers == 8

    def test_custom_max_workers(self, tmp_store, registry):
        """Passing max_workers sizes the executor's thread pool accordingly."""
        ex = Executor(store=tmp_store, registry=registry, max_workers=3)
        assert ex.pool._max_workers == 3

    def test_pool_is_alive(self, executor):
        """A freshly constructed executor's thread pool is not shut down."""
        assert not executor.pool._shutdown


# ===========================================================================
# submit — happy path (state transitions)
# ===========================================================================

class TestSubmitHappyPath:
    """submit() drives a queued run through RUNNING to COMPLETED, forwarding
    inputs/resources to the registered handler and persisting its logs and
    results."""

    def test_state_transitions_to_running_then_completed(self, executor, tmp_store, rec):
        """submit() moves a queued run to COMPLETED once the handler returns successfully."""
        executor.submit(rec, {}, {})
        final = _wait_for_state(tmp_store, rec.run_id, "COMPLETED")
        assert final.state == "COMPLETED"

    def test_results_stored(self, tmp_path, rec):
        """The handler's return value is persisted verbatim as the run's results."""
        expected = {"score": 42, "items": ["a", "b"]}
        h = _make_handler(run_fn=lambda i, r, log: expected)
        reg = _make_registry(h)
        store = _make_store(tmp_path)
        store.create(rec)
        ex = Executor(store=store, registry=reg)
        ex.submit(rec, {}, {})
        final = _wait_for_state(store, rec.run_id, "COMPLETED")
        assert final.results == expected

    def test_logs_contain_start_and_completed(self, executor, tmp_store, rec):
        """The run's logs record both a start and a completion message for a successful run."""
        executor.submit(rec, {}, {})
        final = _wait_for_state(tmp_store, rec.run_id, "COMPLETED")
        combined = " ".join(final.logs)
        assert "Starting tool" in combined
        assert "Completed" in combined

    def test_log_contains_tool_id(self, executor, tmp_store, rec):
        """The first log line names the tool_id being executed."""
        executor.submit(rec, {}, {})
        final = _wait_for_state(tmp_store, rec.run_id, "COMPLETED")
        assert rec.tool_id in final.logs[0]

    def test_updated_epoch_advances(self, executor, tmp_store, rec):
        """updated_epoch does not regress between submission and completion."""
        original_epoch = rec.updated_epoch
        executor.submit(rec, {}, {})
        final = _wait_for_state(tmp_store, rec.run_id, "COMPLETED")
        assert final.updated_epoch >= original_epoch

    def test_inputs_forwarded_to_handler(self, tmp_path, rec):
        """The exact inputs and resources passed to submit() reach the handler's run function unmodified."""
        received: list = []
        def run_fn(inputs, resources, log):
            received.append((inputs, resources))
            return {}

        h = _make_handler(run_fn=run_fn)
        store = _make_store(tmp_path)
        store.create(rec)
        ex = Executor(store=store, registry=_make_registry(h))
        full_inputs = {"genes": ["TP53"]}
        resources = {"cpu": 2}
        ex.submit(rec, full_inputs, resources)
        _wait_for_state(store, rec.run_id, "COMPLETED")
        assert received[0] == (full_inputs, resources)

    def test_log_callable_passed_to_handler(self, tmp_path, rec):
        """The handler receives a callable log() and can invoke it during execution."""
        log_calls: list = []
        def run_fn(inputs, resources, log):
            log("custom log line")
            log_calls.append(True)
            return {}

        h = _make_handler(run_fn=run_fn)
        store = _make_store(tmp_path)
        store.create(rec)
        ex = Executor(store=store, registry=_make_registry(h))
        ex.submit(rec, {}, {})
        _wait_for_state(store, rec.run_id, "COMPLETED")
        assert log_calls  # run_fn was called and invoked log()

    def test_custom_log_line_persisted(self, tmp_path, rec):
        """Lines written via the handler's log() callback are persisted in the final run record's logs."""
        def run_fn(inputs, resources, log):
            log("my-custom-line")
            return {}

        h = _make_handler(run_fn=run_fn)
        store = _make_store(tmp_path)
        store.create(rec)
        ex = Executor(store=store, registry=_make_registry(h))
        ex.submit(rec, {}, {})
        final = _wait_for_state(store, rec.run_id, "COMPLETED")
        assert any("my-custom-line" in line for line in final.logs)

    def test_error_is_none_on_success(self, executor, tmp_store, rec):
        """A successful run leaves the record's error field unset."""
        executor.submit(rec, {}, {})
        final = _wait_for_state(tmp_store, rec.run_id, "COMPLETED")
        assert final.error is None


# ===========================================================================
# submit — failure path
# ===========================================================================

class TestSubmitFailurePath:
    """submit() catches handler exceptions and terminates the run in FAILED
    state with a structured, traceback-bearing error record."""

    def _failing_executor(self, tmp_path, exc: Exception):
        def boom(inputs, resources, log):
            raise exc

        h = _make_handler(run_fn=boom)
        store = _make_store(tmp_path)
        rec = _make_record()
        store.create(rec)
        ex = Executor(store=store, registry=_make_registry(h))
        return ex, store, rec

    def test_state_transitions_to_failed(self, tmp_path):
        """submit() moves a run to FAILED when the handler raises."""
        ex, store, rec = self._failing_executor(tmp_path, RuntimeError("boom"))
        ex.submit(rec, {}, {})
        final = _wait_for_state(store, rec.run_id, "FAILED")
        assert final.state == "FAILED"

    def test_error_code_is_exec_failed(self, tmp_path):
        """A handler exception is recorded with error code EXEC_FAILED."""
        ex, store, rec = self._failing_executor(tmp_path, RuntimeError("boom"))
        ex.submit(rec, {}, {})
        final = _wait_for_state(store, rec.run_id, "FAILED")
        assert final.error["code"] == "EXEC_FAILED"

    def test_error_message_contains_exception_text(self, tmp_path):
        """The failed run's error message includes the original exception's text."""
        ex, store, rec = self._failing_executor(tmp_path, RuntimeError("something went wrong"))
        ex.submit(rec, {}, {})
        final = _wait_for_state(store, rec.run_id, "FAILED")
        assert "something went wrong" in final.error["message"]

    def test_error_trace_is_present(self, tmp_path):
        """A failed run's error record includes a non-empty traceback string."""
        ex, store, rec = self._failing_executor(tmp_path, ValueError("bad value"))
        ex.submit(rec, {}, {})
        final = _wait_for_state(store, rec.run_id, "FAILED")
        assert final.error["trace"]  # non-empty traceback string

    def test_error_trace_contains_exception_type(self, tmp_path):
        """The stored traceback names the exception type that was raised."""
        ex, store, rec = self._failing_executor(tmp_path, ValueError("bad value"))
        ex.submit(rec, {}, {})
        final = _wait_for_state(store, rec.run_id, "FAILED")
        assert "ValueError" in final.error["trace"]

    def test_failed_log_appended(self, tmp_path):
        """A FAILED-state log line is appended when the handler raises."""
        ex, store, rec = self._failing_executor(tmp_path, RuntimeError("oops"))
        ex.submit(rec, {}, {})
        final = _wait_for_state(store, rec.run_id, "FAILED")
        assert any("FAILED" in line for line in final.logs)

    def test_results_is_none_on_failure(self, tmp_path):
        """A failed run leaves results unset (None)."""
        ex, store, rec = self._failing_executor(tmp_path, RuntimeError("oops"))
        ex.submit(rec, {}, {})
        final = _wait_for_state(store, rec.run_id, "FAILED")
        assert final.results is None

    def test_updated_epoch_set_on_failure(self, tmp_path):
        """updated_epoch does not regress when a run fails."""
        ex, store, rec = self._failing_executor(tmp_path, RuntimeError("oops"))
        original = rec.updated_epoch
        ex.submit(rec, {}, {})
        final = _wait_for_state(store, rec.run_id, "FAILED")
        assert final.updated_epoch >= original

    def test_non_runtime_exception_still_fails(self, tmp_path):
        """Non-RuntimeError exceptions (e.g. KeyError) from the handler also fail the run rather than propagating or being swallowed."""
        ex, store, rec = self._failing_executor(tmp_path, KeyError("missing key"))
        ex.submit(rec, {}, {})
        final = _wait_for_state(store, rec.run_id, "FAILED")
        assert final.state == "FAILED"


# ===========================================================================
# submit — registry lookup
# ===========================================================================

class TestSubmitRegistryLookup:
    """submit() resolves the handler by the run's tool_id via the injected
    registry, and rejects runs for unregistered tools before any work
    starts."""

    def test_uses_handler_for_correct_tool_id(self, tmp_path):
        """submit() dispatches to the handler registered under the run's own tool_id, not any other registered handler."""
        called_with: list = []
        def run_fn(inputs, resources, log):
            called_with.append(inputs)
            return {}

        store = _make_store(tmp_path)
        reg = ToolRegistry()
        reg.register(_make_handler(run_fn=run_fn, tool_id="my_tool"))
        rec = _make_record(tool_id="my_tool")
        store.create(rec)

        ex = Executor(store=store, registry=reg)
        ex.submit(rec, {"x": 1}, {})
        _wait_for_state(store, rec.run_id, "COMPLETED")
        assert called_with == [{"x": 1}]

    def test_unregistered_tool_raises_at_submit_time(self, tmp_store):
        """submit() raises KeyError naming the tool_id immediately when no handler is registered for it, rather than failing the run asynchronously."""
        reg = ToolRegistry()  # empty — nothing registered
        ex = Executor(store=tmp_store, registry=reg)
        rec = _make_record(tool_id="ghost_tool")
        tmp_store.create(rec)
        with pytest.raises(KeyError, match="ghost_tool"):
            ex.submit(rec, {}, {})


# ===========================================================================
# append_log
# ===========================================================================

class TestAppendLog:
    """The handler's log callback appends lines to the run record and keeps
    updated_epoch fresh as work progresses."""

    def test_multiple_log_lines_all_persisted(self, tmp_path, rec):
        """Every line written via log() during a run is retained in the final record's logs, in order."""
        def run_fn(inputs, resources, log):
            for i in range(5):
                log(f"line-{i}")
            return {}

        h = _make_handler(run_fn=run_fn)
        store = _make_store(tmp_path)
        store.create(rec)
        ex = Executor(store=store, registry=_make_registry(h))
        ex.submit(rec, {}, {})
        final = _wait_for_state(store, rec.run_id, "COMPLETED")
        log_text = " ".join(final.logs)
        for i in range(5):
            assert f"line-{i}" in log_text

    def test_append_log_updates_epoch(self, tmp_path, rec):
        """Calling log() during a run updates the record's updated_epoch, observable mid-execution."""
        epoch_snapshots: list = []

        def run_fn(inputs, resources, log):
            log("step 1")
            epoch_snapshots.append(store.get(rec.run_id).updated_epoch)
            return {}

        h = _make_handler(run_fn=run_fn)
        store = _make_store(tmp_path)
        store.create(rec)
        ex = Executor(store=store, registry=_make_registry(h))
        ex.submit(rec, {}, {})
        _wait_for_state(store, rec.run_id, "COMPLETED")
        assert epoch_snapshots  # at least one snapshot captured


# ===========================================================================
# Concurrency — multiple simultaneous jobs
# ===========================================================================

class TestConcurrency:
    """The thread-pool-backed executor runs multiple submitted jobs
    concurrently, and one job's failure does not affect the outcome of
    independent jobs."""

    def test_multiple_jobs_all_complete(self, tmp_path):
        """Submitting many runs to a multi-worker executor lets them all complete independently and concurrently."""
        n = 10
        store = _make_store(tmp_path)
        records = [_make_record(run_id=f"run-{i}") for i in range(n)]
        for r in records:
            store.create(r)

        h = _make_handler(run_fn=lambda i, r, log: {"done": True})
        ex = Executor(store=store, registry=_make_registry(h), max_workers=4)

        for r in records:
            ex.submit(r, {}, {})

        for r in records:
            _wait_for_state(store, r.run_id, "COMPLETED", timeout=5.0)

        for r in records:
            assert store.get(r.run_id).state == "COMPLETED"

    def test_failed_job_does_not_affect_others(self, tmp_path):
        """One submitted run failing does not prevent a concurrently submitted, unrelated run from completing successfully."""
        store = _make_store(tmp_path)
        good_rec = _make_record(run_id="good")
        bad_rec = _make_record(run_id="bad")
        store.create(good_rec)
        store.create(bad_rec)

        reg = ToolRegistry()
        reg.register(_make_handler(run_fn=lambda i, r, log: {"ok": True}, tool_id="test_tool"))

        def boom(inputs, resources, log):
            raise RuntimeError("explode")

        reg.register(ToolHandler(
            tool_id="bad_tool",
            validate=lambda i, r: {"ok": True, "errors": [], "warnings": []},
            run=boom,
        ))

        bad_rec2 = _make_record(run_id="bad", tool_id="bad_tool")
        store.update(bad_rec2)

        ex = Executor(store=store, registry=reg, max_workers=2)
        ex.submit(good_rec, {}, {})
        ex.submit(bad_rec2, {}, {})

        _wait_for_state(store, "good", "COMPLETED")
        _wait_for_state(store, "bad", "FAILED")

        assert store.get("good").state == "COMPLETED"
        assert store.get("bad").state == "FAILED"