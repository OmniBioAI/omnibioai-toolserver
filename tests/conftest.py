from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from iam_client import DelegatedExecutionIdentity

from toolserver.security import require_runs_read, require_workflow_execute

# HIPAA-V2-019: every /runs*, /validate route now requires a real
# delegated-execution credential, independently verified via IAM. This
# fixture (and test_app.py's own equivalent below) overrides those two
# FastAPI dependencies with a fixed, already-authorized fake identity so
# the bulk of this repo's existing tests -- which exercise business
# logic (validation, execution, run state, logs, results), not
# authentication itself -- don't need their own Authorization header or
# real Auth/IAM connectivity. Authentication/authorization behavior
# itself (missing/invalid/expired credentials, permission enforcement,
# header forgery, Auth outage, ...) is covered by
# tests/test_toolserver_delegated_auth.py, which deliberately does NOT
# apply this override and exercises the real dependency chain instead.
FAKE_EXECUTE_IDENTITY = DelegatedExecutionIdentity(
    calling_service="tes-test",
    initiating_user="test-user",
    organization="test-org",
    delegated_permissions=frozenset({"workflow.execute"}),
    delegation_id="test-delegation-execute",
)
FAKE_READ_IDENTITY = DelegatedExecutionIdentity(
    calling_service="tes-test",
    initiating_user="test-user",
    organization="test-org",
    delegated_permissions=frozenset({"runs.read"}),
    delegation_id="test-delegation-read",
)


def _authorize(app) -> None:
    app.dependency_overrides[require_workflow_execute] = lambda: FAKE_EXECUTE_IDENTITY
    app.dependency_overrides[require_runs_read] = lambda: FAKE_READ_IDENTITY


@pytest.fixture()
def client(monkeypatch, tmp_path):
    # Ensure RunStore writes into temp
    monkeypatch.setenv("TOOLSERVER_RUN_STORE_DIR", str(tmp_path / "runs"))

    # Patch the tool run function to avoid network
    import toolserver.tools as tools_mod

    def fake_run(inputs, resources, log):
        log("FAKE RUN called")
        time.sleep(0.01)
        return {
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

    monkeypatch.setattr(tools_mod, "_run", fake_run, raising=True)

    from toolserver_app import create_app

    app = create_app()
    _authorize(app)
    return TestClient(app)
