"""HIPAA-V2-001 (ToolServer portion): authoritative RunRecord organization
ownership enforcement.

Deliberately mirrors tests/test_toolserver_delegated_auth.py's fixture
pattern (real create_app(), real dependency chain, only
AsyncIAMClient.validate_delegated_execution mocked) -- these tests prove
ownership is enforced on the real, mounted routes, not on a helper
tested in isolation. Permission enforcement itself (401 vs 403) is
already covered there; this file focuses on the orthogonal condition
this task adds: a valid, correctly-permissioned delegated identity from
the WRONG organization must still be denied.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from iam_client import DelegatedExecutionIdentity
from iam_client.exceptions import AuthorizationError

import toolserver.security as security_mod
from toolserver.models import RunRecord
from toolserver.store import RunStore

ORG_A = "org-a"
ORG_B = "org-b"

EXECUTE_A = DelegatedExecutionIdentity(
    calling_service="tes-service", initiating_user="user-a1", organization=ORG_A,
    delegated_permissions=frozenset({"workflow.execute"}), delegation_id="d-a-exec",
)
READ_A = DelegatedExecutionIdentity(
    calling_service="tes-service", initiating_user="user-a1", organization=ORG_A,
    delegated_permissions=frozenset({"runs.read"}), delegation_id="d-a-read",
)
READ_A_OTHER_USER = DelegatedExecutionIdentity(
    calling_service="tes-service", initiating_user="user-a2", organization=ORG_A,
    delegated_permissions=frozenset({"runs.read"}), delegation_id="d-a-read-2",
)
EXECUTE_A_NO_PERMS = DelegatedExecutionIdentity(
    calling_service="tes-service", initiating_user="user-a1", organization=ORG_A,
    delegated_permissions=frozenset(), delegation_id="d-a-none",
)
READ_B = DelegatedExecutionIdentity(
    calling_service="tes-service", initiating_user="user-b1", organization=ORG_B,
    delegated_permissions=frozenset({"runs.read"}), delegation_id="d-b-read",
)
READ_B_NO_PERMS = DelegatedExecutionIdentity(
    calling_service="tes-service", initiating_user="user-b1", organization=ORG_B,
    delegated_permissions=frozenset(), delegation_id="d-b-none",
)

VALID_BODY = {"tool_id": "enrichr_pathway", "inputs": {"genes": ["TP53"]}, "resources": {}}


def _make_app_client(monkeypatch, tmp_path):
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


def _mock_iam_fixed(monkeypatch, identity: DelegatedExecutionIdentity):
    """Mimics the real contract: returns `identity` only when the
    requested permission is one it actually holds, else raises
    AuthorizationError -- exactly what require_delegated_execution's real
    dependency does against a real Auth introspection response."""
    fake_client = MagicMock()

    async def _validate(token, permission):
        if permission not in identity.delegated_permissions:
            raise AuthorizationError("Delegated permission required")
        return identity

    fake_client.validate_delegated_execution = _validate
    monkeypatch.setattr(security_mod, "get_iam_client", lambda: fake_client)
    return fake_client


def _bearer(token: str = "delegated.jwt.token") -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def ctx(monkeypatch, tmp_path):
    return _make_app_client(monkeypatch, tmp_path)


@pytest.fixture()
def client(ctx):
    return ctx[0]


def _submit(client, identity_setter, **headers) -> str:
    resp = client.post("/runs", json=VALID_BODY, headers={**_bearer(), **headers})
    assert resp.status_code == 200, resp.text
    return resp.json()["run_id"]


def _seed(store, run_id, *, organization_id, state="COMPLETED", results=None, logs=None):
    store.create(RunRecord(
        run_id=run_id, tool_id="enrichr_pathway", state=state,
        created_epoch=1_700_000_000, updated_epoch=1_700_000_001,
        organization_id=organization_id,
        inputs={}, resources={}, logs=logs or ["seeded log line"],
        results=results if results is not None else {"ok": True, "results": {"secret": "org-a-data"}},
        error=None,
    ))


# ===========================================================================
# 1-2: new run stores the authenticated organization, sourced from identity
# ===========================================================================

class TestOwnershipAssignedAtCreation:
    def test_new_run_stores_authenticated_organization(self, ctx, monkeypatch):
        client, store = ctx
        _mock_iam_fixed(monkeypatch, EXECUTE_A)
        run_id = _submit(client, EXECUTE_A)
        assert store.get(run_id).organization_id == ORG_A

    def test_organization_comes_from_delegated_identity_not_a_default(self, ctx, monkeypatch):
        client, store = ctx
        identity = DelegatedExecutionIdentity(
            calling_service="tes-service", initiating_user="user-x", organization="org-unusual-42",
            delegated_permissions=frozenset({"workflow.execute"}), delegation_id="d-x",
        )
        _mock_iam_fixed(monkeypatch, identity)
        run_id = _submit(client, identity)
        assert store.get(run_id).organization_id == "org-unusual-42"


# ===========================================================================
# 3-5: forgery cannot override ownership at creation
# ===========================================================================

class TestCreationForgeryResistance:
    def test_body_organization_id_in_inputs_cannot_override(self, ctx, monkeypatch):
        client, store = ctx
        _mock_iam_fixed(monkeypatch, EXECUTE_A)
        body = {
            "tool_id": "enrichr_pathway",
            "inputs": {"genes": ["TP53"], "organization_id": ORG_B},
            "resources": {},
        }
        resp = client.post("/runs", json=body, headers=_bearer())
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]
        assert store.get(run_id).organization_id == ORG_A

    def test_metadata_organization_id_in_resources_cannot_override(self, ctx, monkeypatch):
        client, store = ctx
        _mock_iam_fixed(monkeypatch, EXECUTE_A)
        body = {
            "tool_id": "enrichr_pathway",
            "inputs": {"genes": ["TP53"]},
            "resources": {"organization_id": ORG_B, "org_id": ORG_B},
        }
        resp = client.post("/runs", json=body, headers=_bearer())
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]
        assert store.get(run_id).organization_id == ORG_A

    def test_forged_org_headers_cannot_override_at_creation(self, ctx, monkeypatch):
        client, store = ctx
        _mock_iam_fixed(monkeypatch, EXECUTE_A)
        run_id = _submit(
            client, EXECUTE_A,
            **{"X-Org-Id": ORG_B, "X-Organization-Id": ORG_B},
        )
        assert store.get(run_id).organization_id == ORG_A


# ===========================================================================
# 6-7: serialization / deserialization preserve ownership
# ===========================================================================

class TestSerializationPreservesOwnership:
    def test_serialization_round_trip_preserves_organization(self):
        rec = RunRecord(
            run_id="ser-1", tool_id="t", state="QUEUED",
            created_epoch=1, updated_epoch=1, organization_id=ORG_A,
        )
        restored = RunRecord.model_validate_json(rec.model_dump_json())
        assert restored.organization_id == ORG_A

    def test_deserialization_from_disk_preserves_organization(self, ctx, monkeypatch):
        """Forces a real read-from-disk path (not the in-process cache) by
        constructing a second RunStore instance against the same
        directory -- proving persistence, not just an in-memory object
        reference, carries the owner."""
        client, store = ctx
        _mock_iam_fixed(monkeypatch, EXECUTE_A)
        run_id = _submit(client, EXECUTE_A)

        fresh_store = RunStore(root_dir=str(store.root))
        assert fresh_store.get(run_id).organization_id == ORG_A


# ===========================================================================
# 8-10, 15: same-organization read access allowed (including a different
# user in the same org)
# ===========================================================================

class TestSameOrgReadAllowed:
    def test_status_same_org_allowed(self, ctx, monkeypatch):
        client, store = ctx
        _seed(store, "run-1", organization_id=ORG_A, state="RUNNING")
        _mock_iam_fixed(monkeypatch, READ_A)
        resp = client.get("/runs/run-1", headers=_bearer())
        assert resp.status_code == 200
        assert resp.json()["run_id"] == "run-1"

    def test_logs_same_org_allowed(self, ctx, monkeypatch):
        client, store = ctx
        _seed(store, "run-2", organization_id=ORG_A, logs=["hello-from-org-a"])
        _mock_iam_fixed(monkeypatch, READ_A)
        resp = client.get("/runs/run-2/logs", headers=_bearer())
        assert resp.status_code == 200
        assert "hello-from-org-a" in resp.json()["logs"]

    def test_results_same_org_allowed(self, ctx, monkeypatch):
        client, store = ctx
        _seed(store, "run-3", organization_id=ORG_A, results={"ok": True, "results": {"hits": 7}})
        _mock_iam_fixed(monkeypatch, READ_A)
        resp = client.get("/runs/run-3/results", headers=_bearer())
        assert resp.status_code == 200
        assert resp.json()["results"]["hits"] == 7

    def test_same_org_different_user_allowed(self, ctx, monkeypatch):
        """Organization isolation, not creator-only isolation: a
        different user in the SAME org with valid runs.read must still
        be able to read the run."""
        client, store = ctx
        _mock_iam_fixed(monkeypatch, EXECUTE_A)
        run_id = _submit(client, EXECUTE_A)  # created by user-a1

        _mock_iam_fixed(monkeypatch, READ_A_OTHER_USER)  # read by user-a2, same org
        resp = client.get(f"/runs/{run_id}", headers=_bearer())
        assert resp.status_code == 200


# ===========================================================================
# 11-14: cross-tenant denial -- the central requirement
# ===========================================================================

class TestCrossTenantDenied:
    def test_status_wrong_org_denied(self, ctx, monkeypatch):
        client, store = ctx
        _seed(store, "org-a-run", organization_id=ORG_A)
        _mock_iam_fixed(monkeypatch, READ_B)
        resp = client.get("/runs/org-a-run", headers=_bearer())
        assert resp.status_code == 200  # ToolServer's own "unknown run" shape, not a 4xx
        assert resp.json() == {"state": "FAILED", "message": "unknown run"}

    def test_logs_wrong_org_denied(self, ctx, monkeypatch):
        client, store = ctx
        _seed(store, "org-a-run", organization_id=ORG_A, logs=["ORG-A-SECRET-LOG-LINE"])
        _mock_iam_fixed(monkeypatch, READ_B)
        resp = client.get("/runs/org-a-run/logs", headers=_bearer())
        assert resp.status_code == 200
        assert "ORG-A-SECRET-LOG-LINE" not in resp.text
        assert resp.json()["logs"] == "[org-a-run] unknown run"

    def test_results_wrong_org_denied(self, ctx, monkeypatch):
        client, store = ctx
        _seed(
            store, "org-a-run", organization_id=ORG_A,
            results={"ok": True, "results": {"secret": "ORG-A-RESULT"}},
        )
        _mock_iam_fixed(monkeypatch, READ_B)
        resp = client.get("/runs/org-a-run/results", headers=_bearer())
        assert resp.status_code == 200
        assert "ORG-A-RESULT" not in resp.text
        assert resp.json() == {"ok": False, "error": {"code": "NOT_FOUND", "message": "unknown run"}}

    def test_exact_valid_foreign_run_id_still_denied(self, ctx, monkeypatch):
        """Possession of a real, exact, valid run_id belonging to another
        org is never sufficient authorization on its own."""
        client, store = ctx
        _mock_iam_fixed(monkeypatch, EXECUTE_A)
        real_run_id = _submit(client, EXECUTE_A)  # a genuine org-A run_id, not guessed

        _mock_iam_fixed(monkeypatch, READ_B)
        resp = client.get(f"/runs/{real_run_id}", headers=_bearer())
        assert resp.json()["message"] == "unknown run"


# ===========================================================================
# 16-17: permission AND ownership both required
# ===========================================================================

class TestPermissionAndOwnershipBothRequired:
    def test_correct_org_no_runs_read_denied(self, ctx, monkeypatch):
        client, store = ctx
        _seed(store, "run-x", organization_id=ORG_A)
        _mock_iam_fixed(monkeypatch, EXECUTE_A_NO_PERMS)  # right org, no runs.read
        resp = client.get("/runs/run-x", headers=_bearer())
        assert resp.status_code == 403

    def test_wrong_org_no_runs_read_denied(self, ctx, monkeypatch):
        client, store = ctx
        _seed(store, "run-x", organization_id=ORG_A)
        _mock_iam_fixed(monkeypatch, READ_B_NO_PERMS)  # wrong org AND no runs.read
        resp = client.get("/runs/run-x", headers=_bearer())
        assert resp.status_code == 403

    def test_wrong_org_with_valid_runs_read_denied(self, ctx, monkeypatch):
        client, store = ctx
        _seed(store, "run-x", organization_id=ORG_A)
        _mock_iam_fixed(monkeypatch, READ_B)  # valid permission, wrong org
        resp = client.get("/runs/run-x", headers=_bearer())
        assert resp.status_code == 200
        assert resp.json()["message"] == "unknown run"

    def test_correct_org_with_valid_runs_read_allowed(self, ctx, monkeypatch):
        client, store = ctx
        _seed(store, "run-x", organization_id=ORG_A)
        _mock_iam_fixed(monkeypatch, READ_A)
        resp = client.get("/runs/run-x", headers=_bearer())
        assert resp.status_code == 200
        assert resp.json()["run_id"] == "run-x"


# ===========================================================================
# 18-23: fail-closed on missing/null/empty/malformed ownership
# ===========================================================================

class TestFailClosedOnMissingOwnership:
    def test_ownerless_legacy_status_denied(self, ctx, monkeypatch):
        client, store = ctx
        store.create(RunRecord(
            run_id="legacy-1", tool_id="enrichr_pathway", state="COMPLETED",
            created_epoch=1, updated_epoch=1,  # no organization_id at all -- pre-V2-001 shape
        ))
        _mock_iam_fixed(monkeypatch, READ_A)
        resp = client.get("/runs/legacy-1", headers=_bearer())
        assert resp.json() == {"state": "FAILED", "message": "unknown run"}

    def test_ownerless_legacy_logs_denied(self, ctx, monkeypatch):
        client, store = ctx
        store.create(RunRecord(
            run_id="legacy-2", tool_id="enrichr_pathway", state="COMPLETED",
            created_epoch=1, updated_epoch=1, logs=["legacy log content"],
        ))
        _mock_iam_fixed(monkeypatch, READ_A)
        resp = client.get("/runs/legacy-2/logs", headers=_bearer())
        assert "legacy log content" not in resp.text
        assert resp.json()["logs"] == "[legacy-2] unknown run"

    def test_ownerless_legacy_results_denied(self, ctx, monkeypatch):
        client, store = ctx
        store.create(RunRecord(
            run_id="legacy-3", tool_id="enrichr_pathway", state="COMPLETED",
            created_epoch=1, updated_epoch=1, results={"ok": True, "results": {"secret": "legacy"}},
        ))
        _mock_iam_fixed(monkeypatch, READ_A)
        resp = client.get("/runs/legacy-3/results", headers=_bearer())
        assert "legacy" not in resp.text
        assert resp.json()["error"]["code"] == "NOT_FOUND"

    def test_null_owner_denied(self, ctx, monkeypatch):
        client, store = ctx
        _seed(store, "null-owner", organization_id=None)
        _mock_iam_fixed(monkeypatch, READ_A)
        resp = client.get("/runs/null-owner", headers=_bearer())
        assert resp.json()["message"] == "unknown run"

    def test_empty_owner_denied(self, ctx, monkeypatch):
        client, store = ctx
        _seed(store, "empty-owner", organization_id="")
        _mock_iam_fixed(monkeypatch, READ_A)
        resp = client.get("/runs/empty-owner", headers=_bearer())
        assert resp.json()["message"] == "unknown run"

    def test_malformed_owner_denied(self, ctx, monkeypatch):
        """A garbage/corrupted owner value that doesn't match any real
        identity's organization must deny exactly like a genuine
        cross-tenant mismatch -- it must never be treated as a wildcard
        or coerced into a match."""
        client, store = ctx
        _seed(store, "malformed-owner", organization_id="\x00\x01-not-a-real-org-##")
        _mock_iam_fixed(monkeypatch, READ_A)
        resp = client.get("/runs/malformed-owner", headers=_bearer())
        assert resp.json()["message"] == "unknown run"

    def test_ownerless_run_never_becomes_readable_via_reader_identity(self, ctx, monkeypatch):
        """Explicitly rules out the forbidden shortcut of inferring
        ownership from whoever happens to be reading."""
        client, store = ctx
        _seed(store, "legacy-4", organization_id=None)
        # Even the "obviously legitimate-looking" reader must be denied --
        # there is no current-reader-becomes-owner fallback anywhere.
        _mock_iam_fixed(monkeypatch, READ_A)
        assert client.get("/runs/legacy-4", headers=_bearer()).json()["message"] == "unknown run"
        _mock_iam_fixed(monkeypatch, READ_B)
        assert client.get("/runs/legacy-4", headers=_bearer()).json()["message"] == "unknown run"


# ===========================================================================
# 24-26: ownership immutability across the run lifecycle
# ===========================================================================

class TestOwnershipImmutability:
    def test_status_update_does_not_alter_owner(self, ctx, monkeypatch):
        client, store = ctx
        _seed(store, "life-1", organization_id=ORG_A, state="QUEUED")
        rec = store.get("life-1")
        rec.state = "RUNNING"
        rec.updated_epoch = 999
        store.update(rec)
        assert store.get("life-1").organization_id == ORG_A

    def test_result_update_does_not_alter_owner(self, ctx, monkeypatch):
        client, store = ctx
        _seed(store, "life-2", organization_id=ORG_A, state="RUNNING", results=None)
        rec = store.get("life-2")
        rec.state = "COMPLETED"
        rec.results = {"ok": True, "results": {"new": "data"}}
        store.update(rec)
        assert store.get("life-2").organization_id == ORG_A

    def test_full_execution_lifecycle_preserves_owner(self, ctx, monkeypatch):
        client, store = ctx
        _mock_iam_fixed(monkeypatch, EXECUTE_A)
        run_id = _submit(client, EXECUTE_A)

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            rec = store.try_get(run_id)
            if rec and rec.state == "COMPLETED":
                break
            time.sleep(0.02)

        final = store.get(run_id)
        assert final.state == "COMPLETED"
        assert final.organization_id == ORG_A

        _mock_iam_fixed(monkeypatch, READ_A)
        assert client.get(f"/runs/{run_id}", headers=_bearer()).status_code == 200
        _mock_iam_fixed(monkeypatch, READ_B)
        assert client.get(f"/runs/{run_id}", headers=_bearer()).json()["message"] == "unknown run"


# ===========================================================================
# 27-29: credential safety and identity-field substitution guards
# ===========================================================================

class TestOwnershipFieldSafety:
    def test_bearer_not_persisted_alongside_owner(self, ctx, monkeypatch):
        client, store = ctx
        _mock_iam_fixed(monkeypatch, EXECUTE_A)
        secret_token = "super-secret-bearer-value-xyz"
        resp = client.post("/runs", json=VALID_BODY, headers=_bearer(secret_token))
        run_id = resp.json()["run_id"]
        rec = store.get(run_id)
        assert secret_token not in rec.model_dump_json()

    def test_calling_service_not_substituted_for_organization(self, ctx, monkeypatch):
        identity = DelegatedExecutionIdentity(
            calling_service="a-completely-different-value", initiating_user="user-a1",
            organization=ORG_A, delegated_permissions=frozenset({"workflow.execute"}),
            delegation_id="d-svc-check",
        )
        client, store = ctx
        _mock_iam_fixed(monkeypatch, identity)
        run_id = _submit(client, identity)
        rec = store.get(run_id)
        assert rec.organization_id == ORG_A
        assert rec.organization_id != identity.calling_service

    def test_initiating_user_not_substituted_for_organization(self, ctx, monkeypatch):
        identity = DelegatedExecutionIdentity(
            calling_service="tes-service", initiating_user="a-completely-different-value",
            organization=ORG_A, delegated_permissions=frozenset({"workflow.execute"}),
            delegation_id="d-user-check",
        )
        client, store = ctx
        _mock_iam_fixed(monkeypatch, identity)
        run_id = _submit(client, identity)
        rec = store.get(run_id)
        assert rec.organization_id == ORG_A
        assert rec.organization_id != identity.initiating_user


# ===========================================================================
# 30-32: unaffected/unchanged surfaces
# ===========================================================================

class TestUnaffectedSurfaces:
    def test_health_unaffected(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_capabilities_unaffected(self, client):
        resp = client.get("/capabilities")
        assert resp.status_code == 200

    def test_validate_protected_but_creates_no_run_record(self, ctx, monkeypatch):
        client, store = ctx
        _mock_iam_fixed(monkeypatch, EXECUTE_A)
        resp = client.post("/validate", json=VALID_BODY, headers=_bearer())
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        # /validate is not a persisted tenant resource -- no RunRecord,
        # and therefore no ownership semantics, are created by it.
        assert len(list(store.root.glob("*.json"))) == 0

    def test_validate_still_denied_without_credential(self, client):
        resp = client.post("/validate", json=VALID_BODY)
        assert resp.status_code == 401

    def test_register_tools_remains_fail_closed(self, client, monkeypatch):
        _mock_iam_fixed(monkeypatch, EXECUTE_A)
        resp = client.post("/register_tools", json={"tools": [{"tool_id": "x"}]}, headers=_bearer())
        assert resp.status_code == 501
        assert "REGISTER_TOOLS_AUTHORIZATION_MODEL_UNRESOLVED" in resp.json()["detail"]


# ===========================================================================
# 34: no direct logs/results bypass of RunRecord-based ownership
# ===========================================================================

class TestNoOwnershipBypass:
    def test_logs_route_never_reads_store_directly_without_authorization(self, ctx, monkeypatch):
        """Regression guard for the exact bypass class this task calls
        out: a route must not be able to construct/return log content
        from anything other than the ownership-checked RunRecord. Seeds
        a foreign-org run with unmistakable sentinel content and asserts
        it never appears anywhere in the denial response."""
        client, store = ctx
        _seed(store, "bypass-check", organization_id=ORG_A, logs=["UNMISTAKABLE-SENTINEL-VALUE-42"])
        _mock_iam_fixed(monkeypatch, READ_B)
        resp = client.get("/runs/bypass-check/logs", headers=_bearer())
        assert "UNMISTAKABLE-SENTINEL-VALUE-42" not in resp.text

    def test_results_route_never_reads_store_directly_without_authorization(self, ctx, monkeypatch):
        client, store = ctx
        _seed(store, "bypass-check-2", organization_id=ORG_A,
              results={"ok": True, "results": {"marker": "UNMISTAKABLE-SENTINEL-VALUE-99"}})
        _mock_iam_fixed(monkeypatch, READ_B)
        resp = client.get("/runs/bypass-check-2/results", headers=_bearer())
        assert "UNMISTAKABLE-SENTINEL-VALUE-99" not in resp.text

    def test_status_route_never_reveals_state_of_foreign_run(self, ctx, monkeypatch):
        client, store = ctx
        _seed(store, "bypass-check-3", organization_id=ORG_A, state="RUNNING")
        _mock_iam_fixed(monkeypatch, READ_B)
        resp = client.get("/runs/bypass-check-3", headers=_bearer())
        assert resp.json().get("state") != "RUNNING"
