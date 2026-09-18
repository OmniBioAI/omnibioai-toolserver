"""
Unit tests for toolserver_app endpoints.

Follows the same fixture pattern as conftest.py:
  - TOOLSERVER_RUN_STORE_DIR monkeypatched to tmp_path
  - toolserver.tools._run monkeypatched to avoid real HTTP calls
  - TestClient wraps create_app()

Run with:
    python -m pytest tests/test_app.py -v

Developer:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from toolserver.models import RunRecord
from toolserver.store import RunStore

from .conftest import _authorize

# ===========================================================================
# Shared fake run / helpers
# ===========================================================================

FAKE_RESULTS = {
    "ok": True,
    "results": {
        "WikiPathways_2024_Human": {
            "library": "WikiPathways_2024_Human",
            "columns": ["rank", "term", "p_value"],
            "items": [{"rank": 1, "term": "DummyPathway", "p_value": 1e-6}],
            "n_terms": 1,
        }
    },
}


def _fake_run(inputs, resources, log):
    log("FAKE RUN called")
    time.sleep(0.01)
    return FAKE_RESULTS


def _failing_run(inputs, resources, log):
    raise RuntimeError("deliberate failure")


def _make_client(monkeypatch, tmp_path, run_fn=None):
    """
    Build a (TestClient, RunStore) pair.

    create_app() hardcodes run_store_dir = "out/runs" and does not read any
    env var, so we cannot redirect it via monkeypatch.setenv.  Instead we
    patch toolserver_app.RunStore so that create_app() gets a real RunStore
    pointed at tmp_path, and we capture that same instance to return here.
    That way the test store IS the app store — no directory mismatch.
    """
    import toolserver.tools as tools_mod
    monkeypatch.setattr(tools_mod, "_run", run_fn or _fake_run, raising=True)

    captured: list[RunStore] = []

    original_RunStore = RunStore

    def capturing_RunStore(root_dir):
        instance = original_RunStore(root_dir=str(tmp_path / "runs"))
        captured.append(instance)
        return instance

    import toolserver_app
    monkeypatch.setattr(toolserver_app, "RunStore", capturing_RunStore)

    from toolserver_app import create_app
    app = create_app()
    _authorize(app)  # HIPAA-V2-019: see conftest.py's own comment
    client = TestClient(app, raise_server_exceptions=False)

    # The store create_app() is actually using
    store = captured[0]
    return client, store


def _wait_for_state(store: RunStore, run_id: str, state: str, timeout: float = 3.0) -> RunRecord:
    """Poll until the record reaches `state`.  Uses try_get so it is safe to
    call immediately after submit() before the record has been flushed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rec = store.try_get(run_id)
        if rec is not None and rec.state == state:
            return rec
        time.sleep(0.02)
    rec = store.try_get(run_id)
    actual = rec.state if rec else "NOT_FOUND"
    raise TimeoutError(f"{run_id} stuck in {actual!r}, expected {state!r}")


VALID_INPUTS = {"genes": ["TP53", "BRCA1"]}
VALID_BODY = {"tool_id": "enrichr_pathway", "inputs": VALID_INPUTS, "resources": {}}


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def client(monkeypatch, tmp_path):
    c, _ = _make_client(monkeypatch, tmp_path)
    return c


@pytest.fixture()
def ctx(monkeypatch, tmp_path):
    """Returns (client, store) for tests that need to inspect stored state."""
    return _make_client(monkeypatch, tmp_path)


# ===========================================================================
# GET /health
# ===========================================================================

class TestHealth:
    """GET /health reports service liveness with a stable response shape."""

    def test_200(self, client):
        """Health check responds with HTTP 200."""
        assert client.get("/health").status_code == 200

    def test_ok_true(self, client):
        """Health check reports ok: true when the service is up."""
        assert client.get("/health").json()["ok"] is True

    def test_service_key_present(self, client):
        """Health response includes a service identifier field."""
        assert "service" in client.get("/health").json()


# ===========================================================================
# GET /capabilities
# ===========================================================================

class TestCapabilities:
    """GET /capabilities advertises the engines and tool catalog callers can rely on."""

    def test_200(self, client):
        """Capabilities endpoint responds with HTTP 200."""
        assert client.get("/capabilities").status_code == 200

    def test_engines_present(self, client):
        """Capabilities response includes an engines field."""
        assert "engines" in client.get("/capabilities").json()

    def test_tools_present(self, client):
        """Capabilities response includes a tools field."""
        assert "tools" in client.get("/capabilities").json()

    def test_enrichr_pathway_listed(self, client):
        """enrichr_pathway is advertised in the tool catalog."""
        tool_ids = [t["tool_id"] for t in client.get("/capabilities").json()["tools"]]
        assert "enrichr_pathway" in tool_ids

    def test_enrichr_pathway_version_v1(self, client):
        """enrichr_pathway is advertised at version v1."""
        tools = client.get("/capabilities").json()["tools"]
        tool = next(t for t in tools if t["tool_id"] == "enrichr_pathway")
        assert tool["version"] == "v1"

    def test_enrichr_pathway_has_libraries_default_feature(self, client):
        """enrichr_pathway advertises a libraries_default feature."""
        tools = client.get("/capabilities").json()["tools"]
        tool = next(t for t in tools if t["tool_id"] == "enrichr_pathway")
        assert "libraries_default" in tool["features"]


# ===========================================================================
# POST /validate
# ===========================================================================

class TestValidate:
    """POST /validate checks a run request's shape without creating a run."""

    def test_valid_inputs_ok_true(self, client):
        """A well-formed request for a known tool validates as ok: true."""
        resp = client.post("/validate", json={
            "tool_id": "enrichr_pathway",
            "inputs": VALID_INPUTS,
            "resources": {},
        })
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_unknown_tool_ok_false(self, client):
        """Validating an unregistered tool_id reports ok: false."""
        resp = client.post("/validate", json={"tool_id": "ghost", "inputs": {}, "resources": {}})
        assert resp.json()["ok"] is False

    def test_unknown_tool_error_code(self, client):
        """Validating an unregistered tool_id reports an UNKNOWN_TOOL error code."""
        resp = client.post(
            "/validate", json={"tool_id": "ghost", "inputs": {}, "resources": {}}
        )
        errors = resp.json()["errors"]
        assert any(e["code"] == "UNKNOWN_TOOL" for e in errors)

    def test_empty_genes_list_ok_false(self, client):
        """An empty genes list fails validation."""
        resp = client.post("/validate", json={
            "tool_id": "enrichr_pathway",
            "inputs": {"genes": []},
            "resources": {},
        })
        assert resp.json()["ok"] is False

    def test_empty_genes_list_errors_non_empty(self, client):
        """An empty genes list produces at least one validation error."""
        resp = client.post("/validate", json={
            "tool_id": "enrichr_pathway",
            "inputs": {"genes": []},
            "resources": {},
        })
        assert len(resp.json()["errors"]) > 0

    def test_empty_genes_error_references_genes_field(self, client):
        """The empty-genes validation error identifies the genes field."""
        resp = client.post("/validate", json={
            "tool_id": "enrichr_pathway",
            "inputs": {"genes": []},
            "resources": {},
        })
        fields = [e.get("field") for e in resp.json()["errors"]]
        assert "genes" in fields

    def test_missing_genes_key_ok_false(self, client):
        """Omitting the genes key entirely fails validation."""
        resp = client.post("/validate", json={
            "tool_id": "enrichr_pathway",
            "inputs": {},
            "resources": {},
        })
        assert resp.json()["ok"] is False

    def test_genes_non_list_ok_false(self, client):
        """A non-list genes value (e.g. a bare string) fails validation."""
        resp = client.post("/validate", json={
            "tool_id": "enrichr_pathway",
            "inputs": {"genes": "TP53"},
            "resources": {},
        })
        assert resp.json()["ok"] is False

    def test_warnings_key_always_present(self, client):
        """The response always includes a warnings key, even when valid."""
        resp = client.post("/validate", json={
            "tool_id": "enrichr_pathway",
            "inputs": VALID_INPUTS,
            "resources": {},
        })
        assert "warnings" in resp.json()

    def test_missing_tool_id_returns_422(self, client):
        """Omitting the required tool_id field is rejected at the HTTP layer with 422."""
        assert client.post("/validate", json={"inputs": VALID_INPUTS}).status_code == 422

    def test_omitting_inputs_does_not_422(self, client):
        """Omitting inputs is legal because the model defaults it to an empty dict."""
        # inputs has default_factory=dict, so omitting is legal at the HTTP layer
        resp = client.post("/validate", json={"tool_id": "enrichr_pathway"})
        assert resp.status_code == 200


# ===========================================================================
# POST /runs
# ===========================================================================

class TestCreateRun:
    """POST /runs validates the request, persists a run record, and executes it asynchronously."""

    def test_returns_run_id(self, client):
        """A successful submission returns a run_id."""
        assert "run_id" in client.post("/runs", json=VALID_BODY).json()

    def test_run_id_has_ts_prefix(self, client):
        """Generated run_id values are namespaced with a ts_ prefix."""
        assert client.post("/runs", json=VALID_BODY).json()["run_id"].startswith("ts_")

    def test_run_ids_are_unique(self, client):
        """Repeated submissions each get a distinct run_id."""
        ids = {client.post("/runs", json=VALID_BODY).json()["run_id"] for _ in range(10)}
        assert len(ids) == 10

    def test_empty_genes_returns_400(self, client):
        """An empty genes list is rejected with HTTP 400 before a run is created."""
        resp = client.post("/runs", json={
            "tool_id": "enrichr_pathway",
            "inputs": {"genes": []},
            "resources": {},
        })
        assert resp.status_code == 400

    def test_empty_genes_error_code_validation_failed(self, client):
        """An empty genes list is rejected with a VALIDATION_FAILED error code."""
        resp = client.post("/runs", json={
            "tool_id": "enrichr_pathway",
            "inputs": {"genes": []},
            "resources": {},
        })
        assert resp.json()["error"]["code"] == "VALIDATION_FAILED"

    def test_unknown_tool_returns_400(self, client):
        """Submitting a run for an unregistered tool_id is rejected with HTTP 400."""
        resp = client.post(
            "/runs", json={"tool_id": "ghost", "inputs": {}, "resources": {}}
        )
        assert resp.status_code == 400

    def test_missing_tool_id_returns_422(self, client):
        """Omitting the required tool_id field is rejected at the HTTP layer with 422."""
        assert client.post("/runs", json={"inputs": VALID_INPUTS}).status_code == 422

    def test_record_created_in_store(self, ctx):
        """A successful submission persists a run record in the run store."""
        client, store = ctx
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        assert store.try_get(run_id) is not None

    def test_record_tool_id_is_enrichr_pathway(self, ctx):
        """The persisted run record retains the submitted tool_id."""
        client, store = ctx
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        assert store.get(run_id).tool_id == "enrichr_pathway"

    def test_record_has_queued_log(self, ctx):
        """The persisted run record starts with a log line noting it was queued."""
        client, store = ctx
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        assert any("Queued" in line for line in store.get(run_id).logs)

    def test_resources_stored_on_record(self, ctx):
        """The submitted resources dict is persisted verbatim on the run record."""
        client, store = ctx
        resources = {"cpu": 4}
        run_id = client.post("/runs", json={**VALID_BODY, "resources": resources}).json()["run_id"]
        assert store.get(run_id).resources == resources

    def test_inputs_stored_as_lightweight_placeholder(self, ctx):
        """Full run inputs are not persisted on the record; only a placeholder summary is."""
        client, store = ctx
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        assert store.get(run_id).inputs == {"summary": "stored externally"}

    def test_run_eventually_completes(self, ctx):
        """A submitted run reaches the COMPLETED state once execution finishes."""
        client, store = ctx
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        _wait_for_state(store, run_id, "COMPLETED")

    def test_failing_run_reaches_failed_state(self, monkeypatch, tmp_path):
        """A run whose tool execution raises reaches the FAILED state."""
        client, store = _make_client(monkeypatch, tmp_path, run_fn=_failing_run)
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        _wait_for_state(store, run_id, "FAILED")

    def test_status_200_on_success(self, client):
        """A valid run submission responds with HTTP 200."""
        assert client.post("/runs", json=VALID_BODY).status_code == 200


# ===========================================================================
# GET /runs/{run_id}
# ===========================================================================

class TestGetRun:
    """GET /runs/{run_id} reports a run's current state, failing closed for unknown ids."""

    def test_200_for_known_run(self, client):
        """A known run_id responds with HTTP 200."""
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        assert client.get(f"/runs/{run_id}").status_code == 200

    def test_run_id_echoed(self, client):
        """The response echoes back the queried run_id."""
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        assert client.get(f"/runs/{run_id}").json()["run_id"] == run_id

    def test_state_is_valid_run_state(self, client):
        """The reported state is always one of the documented run states."""
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        state = client.get(f"/runs/{run_id}").json()["state"]
        assert state in ("QUEUED", "RUNNING", "COMPLETED", "FAILED")

    def test_updated_epoch_present(self, client):
        """The response includes an updated_epoch timestamp."""
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        assert "updated_epoch" in client.get(f"/runs/{run_id}").json()

    def test_unknown_run_id_state_is_failed(self, client):
        """An unrecognized run_id is reported as FAILED rather than erroring."""
        assert client.get("/runs/ghost-run-999").json()["state"] == "FAILED"

    def test_unknown_run_id_has_message(self, client):
        """An unrecognized run_id's response includes an explanatory message."""
        assert "message" in client.get("/runs/ghost-run-999").json()

    def test_state_completed_after_execution(self, ctx):
        """Once execution finishes, the state endpoint reflects COMPLETED."""
        client, store = ctx
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        _wait_for_state(store, run_id, "COMPLETED")
        assert client.get(f"/runs/{run_id}").json()["state"] == "COMPLETED"

    def test_state_failed_after_execution_error(self, monkeypatch, tmp_path):
        """Once execution raises, the state endpoint reflects FAILED."""
        client, store = _make_client(monkeypatch, tmp_path, run_fn=_failing_run)
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        _wait_for_state(store, run_id, "FAILED")
        assert client.get(f"/runs/{run_id}").json()["state"] == "FAILED"


# ===========================================================================
# GET /runs/{run_id}/logs
# ===========================================================================

class TestGetLogs:
    """GET /runs/{run_id}/logs returns a tailable, newline-joined log stream."""

    def test_200_for_known_run(self, client):
        """A known run_id responds with HTTP 200."""
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        assert client.get(f"/runs/{run_id}/logs").status_code == 200

    def test_run_id_echoed(self, client):
        """The response echoes back the queried run_id."""
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        assert client.get(f"/runs/{run_id}/logs").json()["run_id"] == run_id

    def test_logs_field_is_string(self, client):
        """The logs field is returned as a single joined string, not a list."""
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        assert isinstance(client.get(f"/runs/{run_id}/logs").json()["logs"], str)

    def test_unknown_run_contains_unknown_run_message(self, client):
        """An unrecognized run_id's logs field explains the run is unknown."""
        assert "unknown run" in client.get("/runs/ghost-999/logs").json()["logs"]

    def test_fake_run_log_line_appears_after_completion(self, ctx):
        """Log lines emitted by the tool's execution are present once it completes."""
        client, store = ctx
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        _wait_for_state(store, run_id, "COMPLETED")
        logs = client.get(f"/runs/{run_id}/logs").json()["logs"]
        assert "FAKE RUN called" in logs

    def test_tail_limits_line_count(self, ctx):
        """The tail query parameter caps the number of log lines returned."""
        client, store = ctx
        rec = RunRecord(
            run_id="log-many", tool_id="enrichr_pathway", state="COMPLETED",
            created_epoch=1_700_000_000, updated_epoch=1_700_000_001,
            organization_id="test-org",
            inputs={}, resources={}, logs=[f"line-{i}" for i in range(50)],
            results={"ok": True}, error=None,
        )
        store.create(rec)
        data = client.get("/runs/log-many/logs?tail=5").json()
        assert len(data["logs"].strip().splitlines()) == 5

    def test_tail_returns_last_lines(self, ctx):
        """Tailing returns the most recent lines, not the earliest ones."""
        client, store = ctx
        rec = RunRecord(
            run_id="log-tail", tool_id="enrichr_pathway", state="COMPLETED",
            created_epoch=1_700_000_000, updated_epoch=1_700_000_001,
            organization_id="test-org",
            inputs={}, resources={}, logs=[f"line-{i}" for i in range(10)],
            results={"ok": True}, error=None,
        )
        store.create(rec)
        data = client.get("/runs/log-tail/logs?tail=3").json()
        assert "line-9" in data["logs"]
        assert "line-0" not in data["logs"]

    def test_default_tail_is_200_lines(self, ctx):
        """With no tail parameter, the log stream defaults to the last 200 lines."""
        client, store = ctx
        rec = RunRecord(
            run_id="log-default", tool_id="enrichr_pathway", state="COMPLETED",
            created_epoch=1_700_000_000, updated_epoch=1_700_000_001,
            organization_id="test-org",
            inputs={}, resources={}, logs=[f"line-{i}" for i in range(300)],
            results={"ok": True}, error=None,
        )
        store.create(rec)
        data = client.get("/runs/log-default/logs").json()
        assert len(data["logs"].strip().splitlines()) == 200

    def test_tail_zero_returns_all_lines(self, ctx):
        """Line 120→122: tail=0 is falsy → if branch skipped → all lines returned."""
        client, store = ctx
        rec = RunRecord(
            run_id="log-tail-zero", tool_id="enrichr_pathway", state="COMPLETED",
            created_epoch=1_700_000_000, updated_epoch=1_700_000_001,
            organization_id="test-org",
            inputs={}, resources={}, logs=[f"line-{i}" for i in range(5)],
            results={"ok": True}, error=None,
        )
        store.create(rec)
        data = client.get("/runs/log-tail-zero/logs?tail=0").json()
        assert len(data["logs"].strip().splitlines()) == 5


# ===========================================================================
# GET /runs/{run_id}/results
# ===========================================================================

class TestGetResults:
    """GET /runs/{run_id}/results gates access to results by run state."""

    def _inject(self, store: RunStore, run_id: str, state: str,
                results=None, error=None) -> None:
        """Seed the store directly with a run record in an arbitrary state,
        bypassing the /runs endpoint so state-gating can be tested in isolation."""
        store.create(RunRecord(
            run_id=run_id, tool_id="enrichr_pathway", state=state,
            created_epoch=1_700_000_000, updated_epoch=1_700_000_001,
            organization_id="test-org",
            inputs={}, resources={}, logs=[], results=results, error=error,
        ))

    def test_unknown_run_not_found_code(self, client):
        """An unrecognized run_id reports ok: false with a NOT_FOUND error code."""
        data = client.get("/runs/ghost-999/results").json()
        assert data["ok"] is False
        assert data["error"]["code"] == "NOT_FOUND"

    def test_queued_run_not_ready_code(self, ctx):
        """A QUEUED run reports ok: false with a NOT_READY error code."""
        client, store = ctx
        self._inject(store, "q-run", "QUEUED")
        data = client.get("/runs/q-run/results").json()
        assert data["ok"] is False
        assert data["error"]["code"] == "NOT_READY"

    def test_running_run_not_ready_code(self, ctx):
        """A RUNNING run reports ok: false with a NOT_READY error code."""
        client, store = ctx
        self._inject(store, "r-run", "RUNNING")
        data = client.get("/runs/r-run/results").json()
        assert data["ok"] is False
        assert data["error"]["code"] == "NOT_READY"

    def test_not_ready_echoes_state(self, ctx):
        """A not-ready response still echoes the run's actual current state."""
        client, store = ctx
        self._inject(store, "s-run", "RUNNING")
        assert client.get("/runs/s-run/results").json()["state"] == "RUNNING"

    def test_failed_run_not_ready(self, ctx):
        """A FAILED run's results are reported as not ok rather than returning stale data."""
        client, store = ctx
        self._inject(store, "f-run", "FAILED",
                     error={"code": "EXEC_FAILED", "message": "oops", "trace": ""})
        assert client.get("/runs/f-run/results").json()["ok"] is False

    def test_completed_returns_stored_results(self, ctx):
        """A COMPLETED run returns exactly the results stored on its record."""
        client, store = ctx
        expected = {"score": 99, "label": "hit"}
        self._inject(store, "done-run", "COMPLETED", results=expected)
        assert client.get("/runs/done-run/results").json() == expected

    def test_completed_none_results_returns_default_ok(self, ctx):
        """A COMPLETED run with no stored results still returns a default ok: true body."""
        client, store = ctx
        self._inject(store, "done-empty", "COMPLETED", results=None)
        assert client.get("/runs/done-empty/results").json()["ok"] is True

    def test_e2e_results_ok_true(self, ctx):
        """End-to-end: a submitted run's results are ok: true once execution completes."""
        client, store = ctx
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        _wait_for_state(store, run_id, "COMPLETED")
        assert client.get(f"/runs/{run_id}/results").json()["ok"] is True

    def test_e2e_results_contain_wikipathways_key(self, ctx):
        """End-to-end: the completed run's results contain the tool's expected result key."""
        client, store = ctx
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        _wait_for_state(store, run_id, "COMPLETED")
        data = client.get(f"/runs/{run_id}/results").json()
        assert "WikiPathways_2024_Human" in data.get("results", {})

    def test_e2e_failed_run_results_not_ready(self, monkeypatch, tmp_path):
        """End-to-end: a run whose execution raises never exposes results as ok."""
        client, store = _make_client(monkeypatch, tmp_path, run_fn=_failing_run)
        run_id = client.post("/runs", json=VALID_BODY).json()["run_id"]
        _wait_for_state(store, run_id, "FAILED")
        data = client.get(f"/runs/{run_id}/results").json()
        assert data["ok"] is False


# ===========================================================================
# /register_tools — REGISTER_TOOLS_AUTHORIZATION_MODEL_UNRESOLVED
# ===========================================================================
# HIPAA-V2-019: the runtime /register_tools HTTP endpoint (registering an
# arbitrary caller-supplied non-http "stub" tool_def, then submitting a
# run against it to reach the stub's NotImplementedError) is no longer
# reachable through the public API at all -- this endpoint is
# unconditionally disabled (see toolserver_app.py's own comment). That
# specific stub-registration code path in the old register_tools_endpoint
# body was deleted along with the rest of that handler, not merely left
# untested; the equivalent-strength replacement (proving the disabled
# endpoint's exact fail-closed shape) lives in
# test_toolserver_app_coverage.py's register_tools tests.

class TestRegisterToolsDisabled:
    """POST /register_tools is unconditionally disabled, with no delegated permission able to unlock it."""

    def test_register_tools_returns_501_even_when_authenticated(self, ctx):
        """An authenticated caller (this file's fixtures always are, via
        conftest.py's dependency override) gets the exact same denial as
        an unauthenticated one -- there is no delegated permission that
        can unlock this endpoint today."""
        client, _ = ctx
        resp = client.post("/register_tools", json={"tools": [{"tool_id": "stub_exec_tool"}]})
        assert resp.status_code == 501

    def test_register_tools_disabled_leaves_tool_unregistered(self, ctx):
        """Since /register_tools is a no-op, a "registered" tool_id never
        actually enters the registry -- a run submitted against it fails
        closed with UNKNOWN_TOOL, not a stub-execution NotImplementedError."""
        client, _ = ctx
        client.post("/register_tools", json={"tools": [{"tool_id": "stub_exec_tool"}]})
        resp = client.post("/runs", json={
            "tool_id": "stub_exec_tool", "inputs": {}, "resources": {},
        })
        assert resp.status_code == 400
        assert resp.json()["error"]["details"]["errors"][0]["code"] == "UNKNOWN_TOOL"