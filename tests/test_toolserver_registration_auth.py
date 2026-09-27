"""HIPAA-V2-019: POST /register_tools accepts only TES's service-only
registration credential.

Exercises the real app and the real dependency chain
(toolserver.security.require_tes_registration ->
iam_client.registration.require_toolserver_registration ->
AsyncIAMClient.validate_toolserver_registration). Most tests mock that last
method, the same network boundary tests/test_toolserver_delegated_auth.py
mocks for delegated execution; the cross-credential tests mock the raw Auth
HTTP call instead, so a single token is judged by both introspection
endpoints exactly as production would judge it.

Developer:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from iam_client import AsyncIAMClient, DelegatedExecutionIdentity, ServiceRegistrationIdentity

import toolserver.security as security_mod

TES = "omni_client_tesTESTtesTESTtesTEST"
TES_IDENTITY = ServiceRegistrationIdentity(calling_service=TES, organization="1", registration_id="reg-1")
OTHER_IDENTITY = ServiceRegistrationIdentity(
    calling_service="omni_client_otherservice", organization="1", registration_id="reg-2"
)
TOKEN = "registration.jwt.token"


def _http_tool(tool_id: str) -> dict:
    return {
        "tool_id": tool_id,
        "version": "v1",
        "features": {},
        "http": {"url": "http://example.invalid/api", "method": "GET"},
        "inputs": [{"name": "q", "type": "string", "required": True}],
    }


def _client(monkeypatch, tmp_path, *, allow=TES):
    import toolserver_app
    from toolserver.store import RunStore

    monkeypatch.setattr(
        toolserver_app, "RunStore", lambda root_dir: RunStore(root_dir=str(tmp_path / "runs"))
    )
    if allow is None:
        monkeypatch.delenv(security_mod.REGISTRATION_CLIENT_IDS_ENV, raising=False)
    else:
        monkeypatch.setenv(security_mod.REGISTRATION_CLIENT_IDS_ENV, allow)
    return TestClient(toolserver_app.create_app(), raise_server_exceptions=False)


def _mock_registration(monkeypatch, identity):
    fake = MagicMock()
    fake.validate_toolserver_registration = AsyncMock(return_value=identity)
    fake.validate_delegated_execution = AsyncMock(return_value=None)
    monkeypatch.setattr(security_mod, "get_iam_client", lambda: fake)
    return fake


def _tool_ids(client) -> set[str]:
    return {t["tool_id"] for t in client.get("/capabilities").json()["tools"]}


# 1. anonymous
def test_anonymous_registration_rejected_and_registry_unchanged(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    fake = _mock_registration(monkeypatch, TES_IDENTITY)
    before = _tool_ids(client)
    resp = client.post("/register_tools", json={"tools": [_http_tool("anon_tool")]})
    assert resp.status_code == 401
    assert _tool_ids(client) == before
    fake.validate_toolserver_registration.assert_not_awaited()


@pytest.mark.parametrize("header", ["Basic abc", "Bearer", "Token x.y.z", "Bearer a b"])
def test_malformed_authorization_rejected(monkeypatch, tmp_path, header):
    client = _client(monkeypatch, tmp_path)
    _mock_registration(monkeypatch, TES_IDENTITY)
    resp = client.post(
        "/register_tools", json={"tools": [_http_tool("t")]}, headers={"Authorization": header}
    )
    assert resp.status_code == 401


# 2-4. invalid / expired / wrong audience: Auth introspection answers valid=false
def test_invalid_expired_or_wrong_audience_credential_rejected(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _mock_registration(monkeypatch, None)
    resp = client.post(
        "/register_tools", json={"tools": [_http_tool("t")]}, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert resp.status_code == 401
    assert "t" not in _tool_ids(client)


# 6. non-TES service identity
def test_valid_credential_from_a_non_tes_service_is_forbidden(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _mock_registration(monkeypatch, OTHER_IDENTITY)
    resp = client.post(
        "/register_tools", json={"tools": [_http_tool("t")]}, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert resp.status_code == 403
    assert "t" not in _tool_ids(client)


@pytest.mark.parametrize("allow", [None, "", " , "])
def test_unconfigured_allowlist_denies_even_tes(monkeypatch, tmp_path, allow):
    client = _client(monkeypatch, tmp_path, allow=allow)
    _mock_registration(monkeypatch, TES_IDENTITY)
    resp = client.post(
        "/register_tools", json={"tools": [_http_tool("t")]}, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert resp.status_code == 403


# 7-8. valid TES credential registers every HTTP tool it sends
def test_valid_tes_credential_registers_every_http_tool(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path, allow=f"someone_else,{TES}")
    fake = _mock_registration(monkeypatch, TES_IDENTITY)
    before = _tool_ids(client)
    tools = [_http_tool(f"http_tool_{i}") for i in range(1716)]
    resp = client.post("/register_tools", json={"tools": tools}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "registered": 1716, "skipped": 0}
    assert _tool_ids(client) == before | {t["tool_id"] for t in tools}
    fake.validate_toolserver_registration.assert_awaited_once_with(TOKEN)


def test_entries_without_an_http_block_are_skipped_not_stubbed(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _mock_registration(monkeypatch, TES_IDENTITY)
    resp = client.post(
        "/register_tools",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"tools": [_http_tool("ok_tool"), {"tool_id": "no_http"}, {"http": {"url": "x"}}]},
    )
    assert resp.json() == {"ok": True, "registered": 1, "skipped": 2}
    ids = _tool_ids(client)
    assert "ok_tool" in ids and "no_http" not in ids


# 12. no credential in logs or responses
def test_registration_never_logs_or_echoes_the_credential(monkeypatch, tmp_path, capsys):
    client = _client(monkeypatch, tmp_path)
    _mock_registration(monkeypatch, TES_IDENTITY)
    ok = client.post(
        "/register_tools", json={"tools": [_http_tool("t")]}, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    _mock_registration(monkeypatch, None)
    bad = client.post(
        "/register_tools", json={"tools": [_http_tool("t")]}, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    out = capsys.readouterr()
    assert TOKEN not in out.out + out.err
    assert TOKEN not in ok.text and TOKEN not in bad.text
    assert f"service={TES}" in out.out


# 9-11, 13. one token judged by both Auth introspection endpoints, as in production
def _auth_http(registration_valid: bool, delegated_valid: bool):
    def response(body):
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = body
        return r

    async def post(url, json=None, timeout=None):
        if url.endswith("/service/delegations/toolserver/registration/introspect"):
            return response(
                {
                    "valid": registration_valid,
                    "client_id": TES,
                    "organization_id": "1",
                    "scopes": ["toolserver.register"],
                    "registration_id": "reg-1",
                }
                if registration_valid
                else {"valid": False}
            )
        if url.endswith("/service/delegations/toolserver/introspect"):
            return response(
                {
                    "valid": delegated_valid,
                    "client_id": TES,
                    "user_id": "42",
                    "organization_id": "1",
                    "permissions": ["workflow.execute", "runs.read"],
                    "delegation_id": "d-1",
                }
                if delegated_valid
                else {"valid": False}
            )
        raise AssertionError(f"unexpected Auth call {url}")

    iam = AsyncIAMClient(base_url="http://auth-service:8001", redis_url="redis://localhost:6379")
    iam.http = MagicMock()
    iam.http.post = AsyncMock(side_effect=post)
    return iam


def test_registration_credential_cannot_run_validate_or_read_runs(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    iam = _auth_http(registration_valid=True, delegated_valid=False)
    monkeypatch.setattr(security_mod, "get_iam_client", lambda: iam)
    h = {"Authorization": f"Bearer {TOKEN.replace('registration', 'aaa')}"}
    assert client.post("/register_tools", json={"tools": [_http_tool("t")]}, headers=h).status_code == 200
    body = {"tool_id": "t", "inputs": {"q": "x"}, "resources": {}}
    assert client.post("/runs", json=body, headers=h).status_code == 401
    assert client.post("/validate", json=body, headers=h).status_code == 401
    assert client.get("/runs/ts_x", headers=h).status_code == 401
    assert client.get("/runs/ts_x/logs", headers=h).status_code == 401
    assert client.get("/runs/ts_x/results", headers=h).status_code == 401


def test_delegated_user_credential_cannot_register_tools(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    iam = _auth_http(registration_valid=False, delegated_valid=True)
    monkeypatch.setattr(security_mod, "get_iam_client", lambda: iam)
    h = {"Authorization": "Bearer aaa.bbb.ccc"}
    assert client.post("/register_tools", json={"tools": [_http_tool("t")]}, headers=h).status_code == 401
    # The same delegated credential still works for execution, unchanged.
    assert (
        client.post(
            "/validate",
            json={"tool_id": "enrichr_pathway", "inputs": {"genes": ["TP53"]}, "resources": {}},
            headers=h,
        ).status_code
        == 200
    )


def test_anonymous_runs_and_validate_remain_rejected(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    body = {"tool_id": "enrichr_pathway", "inputs": {"genes": ["TP53"]}, "resources": {}}
    assert client.post("/runs", json=body).status_code == 401
    assert client.post("/validate", json=body).status_code == 401


def test_delegated_identity_type_unchanged():
    """The delegated-execution identity used by /runs and /validate still
    carries a user principal; the registration identity deliberately has none."""
    assert "initiating_user" in DelegatedExecutionIdentity.__dataclass_fields__
    assert "initiating_user" not in ServiceRegistrationIdentity.__dataclass_fields__
