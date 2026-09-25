# tests/test_toolserver_app_coverage.py
"""
Coverage-focused tests for toolserver_app.py's app-creation branches
(tools YAML present/absent) and the disabled /register_tools endpoint.

Developer:
    Manish Kumar <manish@omnibioai.org>
"""
import pytest
import yaml
from fastapi.testclient import TestClient
from unittest.mock import patch
import os


# ── helper ──────────────────────────────────────────────────────────────────
def get_client(tools_yaml_exists=False):
    """Create a TestClient with controlled YAML path."""
    yaml_path = "configs/tools.example.yaml" if tools_yaml_exists else "/nonexistent/path.yaml"
    with patch.dict(os.environ, {"TOOLS_YAML_PATH": yaml_path}):
        from toolserver_app import create_app
        return TestClient(create_app())


# ── Line 42: YAML not found warning ─────────────────────────────────────────
def test_create_app_yaml_not_found(capsys):
    """Covers the else branch (line 42) when tools YAML is missing."""
    client = get_client(tools_yaml_exists=False)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "only legacy tools loaded" in captured.out


# ── Line 120→122: get_results when state != COMPLETED ───────────────────────
# Replace test_get_results_not_ready and add test for line 160

# ── /register_tools: anonymous callers are rejected ──────────────────────────
# HIPAA-V2-019: registration requires Auth's service-only
# toolserver_registration credential from an allowlisted service identity
# (TES). The full authenticated contract is in
# tests/test_toolserver_registration_auth.py; these keep the anonymous cases
# next to the rest of the app coverage.
def test_register_tools_anonymous_is_rejected():
    """Reject an anonymous, well-formed HTTP-tool registration with 401."""
    from toolserver_app import create_app
    client = TestClient(create_app())

    resp = client.post("/register_tools", json={"tools": [{
        "tool_id": "my_http_tool",
        "version": "v1",
        "features": {"async": True},
        "http": {
            "url": "http://example.com/run",
            "method": "POST"
        }
    }]})
    assert resp.status_code == 401


def test_register_tools_anonymous_rejected_for_empty_tools_list():
    """Reject anonymous registration with 401 even when the tools list is empty."""
    from toolserver_app import create_app
    client = TestClient(create_app())

    resp = client.post("/register_tools", json={"tools": []})
    assert resp.status_code == 401


def test_register_tools_anonymous_does_not_mutate_registry():
    """An anonymous request can never add a tool, whatever its body."""
    from toolserver_app import create_app
    client = TestClient(create_app())

    client.post("/register_tools", json={"tools": [
        {},
        {"tool_id": "should_never_register", "http": {"url": "http://example.com"}},
    ]})
    caps = client.get("/capabilities").json()
    tool_ids = {t["tool_id"] for t in caps["tools"]}
    assert "should_never_register" not in tool_ids


# ── Line 42: YAML file found → load_tools_from_yaml called ──────────────────
def test_create_app_yaml_found(tmp_path):
    """Covers line 42: when TOOLS_YAML_PATH exists, load_tools_from_yaml is called."""
    tool_defs = [{
        "tool_id": "yaml_loaded_tool",
        "version": "v2",
        "http": {"method": "GET", "url": "https://api.example.com/test"},
        "inputs": [],
        "response_map": {},
    }]
    yaml_file = tmp_path / "tools.yaml"
    yaml_file.write_text(yaml.dump(tool_defs))

    with patch.dict(os.environ, {"TOOLS_YAML_PATH": str(yaml_file)}):
        from toolserver_app import create_app
        client = TestClient(create_app())

    tools = client.get("/capabilities").json()["tools"]
    tool_ids = [t["tool_id"] for t in tools]
    assert "yaml_loaded_tool" in tool_ids