"""Hermetic unit tests for the FastAPI application factory and route handlers.

These tests call the synchronous route functions directly.  That keeps the
coverage deterministic in environments where the installed HTTP test client
does not complete a request, while still exercising the application boundary
logic and response contracts.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.routing import APIRoute

from toolserver.models import RunCreateRequest, RunRecord, ValidateRequest
from toolserver.store import RunStore
from toolserver_app import RegisterToolsRequest, _new_run_id, create_app


def _endpoint(app, path: str, method: str = "GET"):
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == path and method in route.methods:
            return route.endpoint
    raise AssertionError(f"route not found: {method} {path}")


@pytest.fixture()
def app_context(monkeypatch, tmp_path):
    """Create an app with a temporary store and an executor that does no work."""
    import toolserver_app

    monkeypatch.setenv("TOOLS_YAML_PATH", str(tmp_path / "missing.yaml"))
    real_store = RunStore
    store = real_store(str(tmp_path / "runs"))
    monkeypatch.setattr(toolserver_app, "RunStore", lambda _root: store)

    class FakeExecutor:
        def __init__(self, store, registry, max_workers):
            self.store = store
            self.registry = registry
            self.submissions = []
            executors.append(self)

        def submit(self, rec, full_inputs, resources):
            self.submissions.append((rec, full_inputs, resources))

    executors = []
    monkeypatch.setattr(toolserver_app, "Executor", FakeExecutor)
    app = create_app()
    return app, store, executors[0]


def test_new_run_id_has_expected_prefix_and_unique_token():
    first = _new_run_id()
    second = _new_run_id()
    assert first.startswith("ts_")
    assert len(first) > len("ts_")
    assert first != second


def test_create_app_missing_yaml_keeps_legacy_capabilities(app_context):
    app, _, _ = app_context
    capabilities = _endpoint(app, "/capabilities")()
    tool_ids = {item["tool_id"] for item in capabilities["tools"]}
    assert "enrichr_pathway" in tool_ids
    assert capabilities["engines"] == ["http_toolserver"]


def test_create_app_loads_tools_when_yaml_exists(monkeypatch, tmp_path):
    import toolserver_app

    yaml_path = tmp_path / "tools.yaml"
    yaml_path.write_text("- tool_id: yaml_tool\n")
    loaded = MagicMock()
    monkeypatch.setenv("TOOLS_YAML_PATH", str(yaml_path))
    monkeypatch.setattr(toolserver_app, "load_tools_from_yaml", loaded)
    monkeypatch.setattr(toolserver_app, "RunStore", lambda _root: RunStore(str(tmp_path / "runs")))
    monkeypatch.setattr(toolserver_app, "Executor", lambda **kwargs: MagicMock())

    create_app()
    loaded.assert_called_once()
    assert loaded.call_args.args[1] == str(yaml_path)


def test_validate_returns_success_and_unknown_tool_error(app_context):
    app, _, _ = app_context
    validate = _endpoint(app, "/validate", "POST")

    valid = validate(ValidateRequest(tool_id="enrichr_pathway", inputs={"genes": ["TP53"]}))
    assert valid["ok"] is True

    unknown = validate(ValidateRequest(tool_id="missing", inputs={}))
    assert unknown["ok"] is False
    assert unknown["errors"][0]["code"] == "UNKNOWN_TOOL"


def test_create_run_rejects_invalid_input_and_queues_valid_run(app_context):
    app, store, _ = app_context
    create_run = _endpoint(app, "/runs", "POST")

    invalid = create_run(RunCreateRequest(tool_id="enrichr_pathway", inputs={}))
    assert invalid.status_code == 400
    assert json.loads(invalid.body)["error"]["code"] == "VALIDATION_FAILED"

    valid = create_run(
        RunCreateRequest(
            tool_id="enrichr_pathway",
            inputs={"genes": ["TP53"]},
            resources={"cpu": 1},
        )
    )
    run_id = valid["run_id"]
    record = store.get(run_id)
    assert record.state == "QUEUED"
    assert record.inputs == {"summary": "stored externally"}


def test_run_status_logs_and_results_cover_unknown_pending_and_completed(app_context):
    app, store, _ = app_context
    get_run = _endpoint(app, "/runs/{run_id}")
    get_logs = _endpoint(app, "/runs/{run_id}/logs")
    get_results = _endpoint(app, "/runs/{run_id}/results")

    assert get_run("unknown")["state"] == "FAILED"
    assert get_logs("unknown")["logs"] == "[unknown] unknown run"
    assert get_results("unknown")["error"]["code"] == "NOT_FOUND"

    pending = RunRecord(
        run_id="pending",
        tool_id="enrichr_pathway",
        state="RUNNING",
        created_epoch=1,
        updated_epoch=2,
        logs=["one", "two", "three"],
    )
    store.create(pending)
    assert get_run("pending")["state"] == "RUNNING"
    assert get_logs("pending", tail=2)["logs"] == "two\nthree"
    assert get_logs("pending", tail=0)["logs"] == "one\ntwo\nthree"
    assert get_results("pending")["error"]["code"] == "NOT_READY"

    pending.state = "COMPLETED"
    pending.results = None
    store.update(pending)
    assert get_results("pending") == {"ok": True, "results": {}}

    pending.results = {"ok": True, "results": {"items": []}}
    store.update(pending)
    assert get_results("pending") == pending.results


def test_register_tools_skips_invalid_entries_and_registers_http_and_stub(app_context):
    app, _, executor = app_context
    register = _endpoint(app, "/register_tools", "POST")
    capabilities = _endpoint(app, "/capabilities")

    response = register(
        RegisterToolsRequest(
            tools=[
                {},
                {"tool_id": "http_tool", "http": {"url": "http://example.test"}},
                {"tool_id": "stub_tool", "features": {"local": True}},
            ]
        )
    )
    assert response == {"ok": True, "registered": 2}
    ids = {item["tool_id"] for item in capabilities()["tools"]}
    assert {"http_tool", "stub_tool"}.issubset(ids)
    assert executor.registry.get("stub_tool").validate({}, {})["ok"] is True
    with pytest.raises(NotImplementedError, match="stub_tool"):
        executor.registry.get("stub_tool").run({}, {}, lambda _: None)


def test_register_tools_empty_list_and_health(app_context):
    app, _, _ = app_context
    register = _endpoint(app, "/register_tools", "POST")
    health = _endpoint(app, "/health")
    assert register(RegisterToolsRequest(tools=[])) == {"ok": True, "registered": 0}
    assert health() == {"ok": True, "service": "omnibioai-toolserver"}
