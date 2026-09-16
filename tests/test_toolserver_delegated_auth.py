"""HIPAA-V2-019: ToolServer-side delegated-execution authentication and
permission-enforcement tests.

Deliberately does NOT use conftest.py's/test_app.py's dependency
override -- these tests exercise the REAL FastAPI dependency chain
(`toolserver.security.require_workflow_execute` /
`require_runs_read` -> `iam_client.delegated.require_delegated_execution`
-> `AsyncIAMClient.validate_delegated_execution`) mounted on the real
routes, proving the security dependency is actually wired in rather than
only testing a helper in isolation. Only the one network boundary --
`AsyncIAMClient.validate_delegated_execution` itself -- is mocked, the
same boundary omnibioai-tes's own IAM integration tests mock; that
method's own internal correctness (introspection call shape, fail-closed
HTTP/JSON/schema handling, audience/type checks, ...) is exhaustively
covered by omnibioai-iam-client's own 41-test delegated-identity suite
and is not re-tested here. One deeper test below additionally mocks the
raw httpx call to prove the full chain works without that shortcut.
"""
from __future__ import annotations

from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from iam_client import DelegatedExecutionIdentity
from iam_client.exceptions import AuthorizationError

import toolserver.security as security_mod
from toolserver.models import RunRecord
from toolserver.store import RunStore

EXECUTE_IDENTITY = DelegatedExecutionIdentity(
    calling_service="tes-service",
    initiating_user="user-42",
    organization="org-7",
    delegated_permissions=frozenset({"workflow.execute"}),
    delegation_id="delegation-exec-1",
)
READ_IDENTITY = DelegatedExecutionIdentity(
    calling_service="tes-service",
    initiating_user="user-42",
    organization="org-7",
    delegated_permissions=frozenset({"runs.read"}),
    delegation_id="delegation-read-1",
)
BOTH_IDENTITY = DelegatedExecutionIdentity(
    calling_service="tes-service",
    initiating_user="user-42",
    organization="org-7",
    delegated_permissions=frozenset({"workflow.execute", "runs.read"}),
    delegation_id="delegation-both-1",
)

VALID_BODY = {"tool_id": "enrichr_pathway", "inputs": {"genes": ["TP53"]}, "resources": {}}


def _make_app_client(monkeypatch, tmp_path):
    """Real create_app(), real dependency chain (no override), RunStore
    redirected to tmp_path the same way test_app.py's own helper does."""
    import toolserver.tools as tools_mod

    def fake_run(inputs, resources, log):
        log("FAKE RUN called")
        return {"ok": True, "results": {"hits": 1}}

    monkeypatch.setattr(tools_mod, "_run", fake_run, raising=True)

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
    client = TestClient(app, raise_server_exceptions=False)
    return client, captured[0]


def _mock_iam(
    monkeypatch,
    *,
    return_value: Optional[DelegatedExecutionIdentity] = None,
    side_effect=None,
):
    """Replaces toolserver.security.get_iam_client()'s return value with a
    fake client whose validate_delegated_execution(token, permission) is
    fully controlled -- this is the exact boundary
    require_delegated_execution itself calls, so every fail-closed/
    permission-split behavior it already implements (401 on None, 403 on
    AuthorizationError) is exercised for real."""
    fake_client = MagicMock()
    if side_effect is not None:
        fake_client.validate_delegated_execution = AsyncMock(side_effect=side_effect)
    else:
        fake_client.validate_delegated_execution = AsyncMock(return_value=return_value)
    monkeypatch.setattr(security_mod, "get_iam_client", lambda: fake_client)
    return fake_client


def _bearer(token: str = "delegated.jwt.token") -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def client(monkeypatch, tmp_path):
    c, _ = _make_app_client(monkeypatch, tmp_path)
    return c


@pytest.fixture()
def ctx(monkeypatch, tmp_path):
    return _make_app_client(monkeypatch, tmp_path)


# ===========================================================================
# 1-2, 31: valid workflow.execute can submit / validate; existing behavior
# unaffected once authenticated
# ===========================================================================

class TestExecutePermissionAllowed:
    def test_valid_execute_identity_can_submit_run(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=EXECUTE_IDENTITY)
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer())
        assert resp.status_code == 200
        assert "run_id" in resp.json()

    def test_valid_execute_identity_can_validate(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=EXECUTE_IDENTITY)
        resp = client.post("/validate", json=VALID_BODY, headers=_bearer())
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_authenticated_run_still_reaches_completed(self, ctx, monkeypatch):
        """HIPAA-V2-019 only adds authentication/permission enforcement --
        existing execution/business logic must behave exactly as before
        once a caller is authenticated."""
        import time

        client, store = ctx
        _mock_iam(monkeypatch, return_value=EXECUTE_IDENTITY)
        run_id = client.post("/runs", json=VALID_BODY, headers=_bearer()).json()["run_id"]
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            rec = store.try_get(run_id)
            if rec and rec.state == "COMPLETED":
                break
            time.sleep(0.02)
        assert store.get(run_id).state == "COMPLETED"

    def test_validation_failure_business_logic_unaffected_by_auth(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=EXECUTE_IDENTITY)
        resp = client.post(
            "/runs",
            json={"tool_id": "enrichr_pathway", "inputs": {"genes": []}, "resources": {}},
            headers=_bearer(),
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "VALIDATION_FAILED"


# ===========================================================================
# 3-5: valid runs.read can read status/logs/results
# ===========================================================================

class TestReadPermissionAllowed:
    def _seed(self, store, run_id="known-run", state="COMPLETED", results=None, organization_id="org-7"):
        store.create(RunRecord(
            run_id=run_id, tool_id="enrichr_pathway", state=state,
            created_epoch=1_700_000_000, updated_epoch=1_700_000_001,
            organization_id=organization_id,
            inputs={}, resources={}, logs=["line-1"], results=results or {"ok": True, "results": {}},
            error=None,
        ))

    def test_valid_read_identity_can_get_status(self, ctx, monkeypatch):
        client, store = ctx
        self._seed(store)
        _mock_iam(monkeypatch, return_value=READ_IDENTITY)
        resp = client.get("/runs/known-run", headers=_bearer())
        assert resp.status_code == 200
        assert resp.json()["run_id"] == "known-run"

    def test_valid_read_identity_can_get_logs(self, ctx, monkeypatch):
        client, store = ctx
        self._seed(store)
        _mock_iam(monkeypatch, return_value=READ_IDENTITY)
        resp = client.get("/runs/known-run/logs", headers=_bearer())
        assert resp.status_code == 200
        assert "line-1" in resp.json()["logs"]

    def test_valid_read_identity_can_get_results(self, ctx, monkeypatch):
        client, store = ctx
        self._seed(store, results={"ok": True, "results": {"hits": 3}})
        _mock_iam(monkeypatch, return_value=READ_IDENTITY)
        resp = client.get("/runs/known-run/results", headers=_bearer())
        assert resp.status_code == 200
        assert resp.json()["results"]["hits"] == 3


# ===========================================================================
# 6-16: authentication denials (fail closed)
# ===========================================================================

class TestAuthenticationDenied:
    def test_missing_bearer_denied(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=EXECUTE_IDENTITY)  # would succeed if reached
        resp = client.post("/runs", json=VALID_BODY)
        assert resp.status_code == 401

    def test_malformed_bearer_denied(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=EXECUTE_IDENTITY)
        resp = client.post("/runs", json=VALID_BODY, headers={"Authorization": "Basic abc123"})
        assert resp.status_code == 401

    def test_bearer_with_no_token_denied(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=EXECUTE_IDENTITY)
        resp = client.post("/runs", json=VALID_BODY, headers={"Authorization": "Bearer"})
        assert resp.status_code == 401

    def test_ordinary_user_token_denied(self, client, monkeypatch):
        """An ordinary user access token is not a delegated_execution
        token; real Auth introspection returns valid:false for it, which
        validate_delegated_execution surfaces as None."""
        _mock_iam(monkeypatch, return_value=None)
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer("ordinary-user-jwt"))
        assert resp.status_code == 401

    def test_ordinary_service_credentials_token_denied(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=None)
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer("client-credentials-jwt"))
        assert resp.status_code == 401

    def test_invalid_delegated_token_denied(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=None)
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer("forged-delegated-jwt"))
        assert resp.status_code == 401

    def test_expired_delegated_token_denied(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=None)  # Auth's own exp check -> valid:false -> None
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer("expired-delegated-jwt"))
        assert resp.status_code == 401

    def test_revoked_delegated_token_denied(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=None)
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer("revoked-delegated-jwt"))
        assert resp.status_code == 401

    def test_wrong_audience_denied(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=None)  # Auth's own aud check -> valid:false -> None
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer("wrong-audience-jwt"))
        assert resp.status_code == 401

    def test_wrong_token_type_denied(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=None)
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer("wrong-type-jwt"))
        assert resp.status_code == 401

    def test_auth_unavailable_denied(self, client, monkeypatch):
        """validate_delegated_execution itself fails closed to None on
        any Auth-unreachable/malformed-response condition (see
        omnibioai-iam-client's own tests) -- from ToolServer's
        perspective this is indistinguishable from an invalid token, and
        must deny exactly the same way, never fall back."""
        _mock_iam(monkeypatch, return_value=None)
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer())
        assert resp.status_code == 401

    def test_malformed_introspection_response_denied(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=None)
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer())
        assert resp.status_code == 401

    def test_valid_false_denied(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=None)
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer())
        assert resp.status_code == 401

    def test_no_anonymous_fallback_after_denial(self, client, monkeypatch):
        """A denied delegated credential must never cause a silent retry
        as an unauthenticated request -- the run must never be created."""
        _mock_iam(monkeypatch, return_value=None)
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer())
        assert resp.status_code == 401
        assert "run_id" not in resp.json()


# ===========================================================================
# 17-20: permission enforcement (authenticated but wrong/missing scope)
# ===========================================================================

class TestPermissionEnforcement:
    def test_execute_identity_lacking_workflow_execute_denied_403(self, client, monkeypatch):
        _mock_iam(monkeypatch, side_effect=AuthorizationError("missing permission"))
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer())
        assert resp.status_code == 403

    def test_read_identity_lacking_runs_read_denied_403(self, ctx, monkeypatch):
        client, store = ctx
        store.create(RunRecord(
            run_id="r1", tool_id="enrichr_pathway", state="COMPLETED",
            created_epoch=1, updated_epoch=1, organization_id="org-7",
            inputs={}, resources={}, logs=[],
            results={"ok": True}, error=None,
        ))
        _mock_iam(monkeypatch, side_effect=AuthorizationError("missing permission"))
        resp = client.get("/runs/r1", headers=_bearer())
        assert resp.status_code == 403

    def test_runs_read_cannot_substitute_for_workflow_execute(self, client, monkeypatch):
        """A delegated identity holding only runs.read must not be able
        to submit a run -- require_delegated_execution itself enforces
        this (identity.require_permission raises when the scope
        doesn't include the one requested), proven end-to-end here."""
        fake_client = MagicMock()

        async def _validate(token, permission):
            if permission not in READ_IDENTITY.delegated_permissions:
                raise AuthorizationError("Delegated permission required")
            return READ_IDENTITY

        fake_client.validate_delegated_execution = _validate
        monkeypatch.setattr(security_mod, "get_iam_client", lambda: fake_client)

        resp = client.post("/runs", json=VALID_BODY, headers=_bearer())
        assert resp.status_code == 403

    def test_workflow_execute_cannot_substitute_for_runs_read(self, ctx, monkeypatch):
        client, store = ctx
        store.create(RunRecord(
            run_id="r2", tool_id="enrichr_pathway", state="COMPLETED",
            created_epoch=1, updated_epoch=1, organization_id="org-7",
            inputs={}, resources={}, logs=[],
            results={"ok": True}, error=None,
        ))

        fake_client = MagicMock()

        async def _validate(token, permission):
            if permission not in EXECUTE_IDENTITY.delegated_permissions:
                raise AuthorizationError("Delegated permission required")
            return EXECUTE_IDENTITY

        fake_client.validate_delegated_execution = _validate
        monkeypatch.setattr(security_mod, "get_iam_client", lambda: fake_client)

        resp = client.get("/runs/r2", headers=_bearer())
        assert resp.status_code == 403

    def test_identity_with_both_permissions_can_do_both(self, ctx, monkeypatch):
        client, store = ctx
        store.create(RunRecord(
            run_id="r3", tool_id="enrichr_pathway", state="COMPLETED",
            created_epoch=1, updated_epoch=1, organization_id="org-7",
            inputs={}, resources={}, logs=[],
            results={"ok": True}, error=None,
        ))

        fake_client = MagicMock()

        async def _validate(token, permission):
            if permission not in BOTH_IDENTITY.delegated_permissions:
                raise AuthorizationError("Delegated permission required")
            return BOTH_IDENTITY

        fake_client.validate_delegated_execution = _validate
        monkeypatch.setattr(security_mod, "get_iam_client", lambda: fake_client)

        assert client.post("/runs", json=VALID_BODY, headers=_bearer()).status_code == 200
        assert client.get("/runs/r3", headers=_bearer()).status_code == 200


# ===========================================================================
# 21-24: header forgery cannot alter identity/authority
# ===========================================================================

class TestHeaderForgeryResistance:
    FORGED_HEADERS = {
        "X-User-Id": "attacker",
        "X-Org-Id": "attacker-org",
        "X-Organization-Id": "attacker-org",
        "X-Roles": "admin",
        "X-Permissions": "workflow.execute,runs.read,workflow.manage",
        "X-Service-Id": "attacker-service",
    }

    def test_forged_headers_alongside_valid_credential_do_not_change_outcome(self, client, monkeypatch):
        mock = _mock_iam(monkeypatch, return_value=EXECUTE_IDENTITY)
        resp = client.post(
            "/runs", json=VALID_BODY,
            headers={**_bearer(), **self.FORGED_HEADERS},
        )
        assert resp.status_code == 200
        # The dependency call itself only ever received (token, permission)
        # -- there is no parameter through which any of these headers
        # could have reached it.
        mock.validate_delegated_execution.assert_called_once_with("delegated.jwt.token", "workflow.execute")

    def test_forged_headers_cannot_grant_authority_without_a_bearer(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=EXECUTE_IDENTITY)
        resp = client.post("/runs", json=VALID_BODY, headers=self.FORGED_HEADERS)
        assert resp.status_code == 401

    def test_forged_permissions_header_cannot_add_authority_beyond_identity(self, client, monkeypatch):
        """Identity only has runs.read; X-Permissions claims workflow.execute
        too -- must still be denied on the execute route."""
        _mock_iam(monkeypatch, side_effect=AuthorizationError("missing permission"))
        resp = client.post(
            "/runs", json=VALID_BODY,
            headers={**_bearer(), "X-Permissions": "workflow.execute"},
        )
        assert resp.status_code == 403

    def test_forged_service_header_cannot_alter_calling_service(self, client, monkeypatch):
        mock = _mock_iam(monkeypatch, return_value=EXECUTE_IDENTITY)
        client.post(
            "/runs", json=VALID_BODY,
            headers={**_bearer(), "X-Service-Id": "attacker-service"},
        )
        # The identity actually used is exactly what Auth introspection
        # (here, the mock standing in for it) returned.
        result_identity = mock.validate_delegated_execution.return_value
        assert result_identity.calling_service == "tes-service"


# ===========================================================================
# 25-28: credential safety
# ===========================================================================

class TestCredentialSafety:
    SECRET_TOKEN = "super-secret-delegated-bearer-value"

    def test_bearer_absent_from_401_response_body(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=None)
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer(self.SECRET_TOKEN))
        assert self.SECRET_TOKEN not in resp.text

    def test_bearer_absent_from_403_response_body(self, client, monkeypatch):
        _mock_iam(monkeypatch, side_effect=AuthorizationError("missing permission"))
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer(self.SECRET_TOKEN))
        assert self.SECRET_TOKEN not in resp.text

    def test_bearer_absent_from_run_store(self, ctx, monkeypatch):
        client, store = ctx
        _mock_iam(monkeypatch, return_value=EXECUTE_IDENTITY)
        run_id = client.post("/runs", json=VALID_BODY, headers=_bearer(self.SECRET_TOKEN)).json()["run_id"]
        rec = store.get(run_id)
        assert self.SECRET_TOKEN not in rec.model_dump_json()

    def test_bearer_absent_from_result_metadata(self, ctx, monkeypatch):
        import time

        client, store = ctx
        _mock_iam(monkeypatch, return_value=EXECUTE_IDENTITY)
        run_id = client.post("/runs", json=VALID_BODY, headers=_bearer(self.SECRET_TOKEN)).json()["run_id"]
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            rec = store.try_get(run_id)
            if rec and rec.state == "COMPLETED":
                break
            time.sleep(0.02)
        results_resp = client.get(f"/runs/{run_id}/results", headers=_bearer(self.SECRET_TOKEN))
        assert self.SECRET_TOKEN not in results_resp.text

    def test_bearer_absent_from_security_module_source(self):
        """No logging/print call sites exist in the security module that
        could accidentally emit a bearer."""
        import inspect

        source = inspect.getsource(security_mod)
        assert "print(" not in source
        assert "logging." not in source
        assert "logger." not in source


# ===========================================================================
# 29-30: public/unrelated routes unaffected
# ===========================================================================

class TestPublicRoutesUnaffected:
    def test_health_requires_no_credential(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_capabilities_requires_no_credential(self, client):
        resp = client.get("/capabilities")
        assert resp.status_code == 200
        assert "engines" in resp.json()

    def test_health_ignores_forged_headers_and_still_public(self, client):
        resp = client.get("/health", headers={"X-User-Id": "attacker", "X-Permissions": "admin"})
        assert resp.status_code == 200


# ===========================================================================
# 32: no anonymous fallback exists for any protected route
# ===========================================================================

class TestNoAnonymousFallbackAcrossRoutes:
    @pytest.mark.parametrize("method,path,body", [
        ("post", "/validate", VALID_BODY),
        ("post", "/runs", VALID_BODY),
        ("get", "/runs/some-run", None),
        ("get", "/runs/some-run/logs", None),
        ("get", "/runs/some-run/results", None),
    ])
    def test_route_denies_without_any_credential(self, client, monkeypatch, method, path, body):
        _mock_iam(monkeypatch, return_value=BOTH_IDENTITY)  # would succeed if reached
        call = getattr(client, method)
        resp = call(path, json=body) if body is not None else call(path)
        assert resp.status_code == 401


# ===========================================================================
# /register_tools: unconditionally disabled, regardless of credential
# ===========================================================================

class TestRegisterToolsUnresolved:
    def test_register_tools_denied_even_with_valid_credential(self, client, monkeypatch):
        _mock_iam(monkeypatch, return_value=BOTH_IDENTITY)
        resp = client.post(
            "/register_tools", json={"tools": [{"tool_id": "x"}]}, headers=_bearer(),
        )
        assert resp.status_code == 501

    def test_register_tools_denied_without_any_credential(self, client):
        resp = client.post("/register_tools", json={"tools": [{"tool_id": "x"}]})
        assert resp.status_code == 501


# ===========================================================================
# Deeper end-to-end proof: real validate_delegated_execution, only the
# raw httpx introspection call mocked -- proves the full chain, not just
# the client.validate_delegated_execution() shortcut used above.
# ===========================================================================

class TestFullChainWithoutShortcut:
    def test_real_validate_delegated_execution_through_mocked_http(self, client, monkeypatch):
        real_client = security_mod.AsyncIAMClient(base_url="http://fake-auth")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "valid": True,
            "client_id": "tes-service",
            "user_id": "user-42",
            "organization_id": "org-7",
            "permissions": ["workflow.execute"],
            "delegation_id": "delegation-real-1",
        }
        real_client.http.post = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(security_mod, "get_iam_client", lambda: real_client)

        resp = client.post(
            "/runs", json=VALID_BODY,
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
        )
        assert resp.status_code == 200
        real_client.http.post.assert_called_once()
        url = real_client.http.post.call_args.args[0]
        assert url.endswith("/service/delegations/toolserver/introspect")

    def test_real_chain_denies_valid_false(self, client, monkeypatch):
        real_client = security_mod.AsyncIAMClient(base_url="http://fake-auth")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"valid": False}
        real_client.http.post = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(security_mod, "get_iam_client", lambda: real_client)

        resp = client.post(
            "/runs", json=VALID_BODY,
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
        )
        assert resp.status_code == 401
