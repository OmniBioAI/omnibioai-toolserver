"""HIPAA-V2-019: ToolServer-side delegated-execution authentication and
permission enforcement.

ToolServer independently verifies every protected request's bearer
credential through Auth's dedicated delegation-introspection contract
(`POST /service/delegations/toolserver/introspect`), consumed exclusively
via the shared `omnibioai-iam-client` package's public
`iam_client.delegated` API. This module deliberately does not
reimplement JWT parsing, Auth introspection, audience/type checking, or
permission validation -- all of that already lives in Auth and
omnibioai-iam-client; duplicating it here would create a second,
divergent security boundary.

Two fixed dependencies are exposed, matching the exact permission
vocabulary already established by TES/Auth (Auth's
delegated_execution_service.py::ACCEPTED_PERMISSIONS = {"workflow.execute",
"runs.read"} -- no permission is invented here):

* `require_workflow_execute` -- run submission, validation.
* `require_runs_read` -- run status, logs, results.

Both are plain, importable module-level functions (not factory-returned
closures bound at route-registration time) specifically so:

1. the underlying `AsyncIAMClient` is constructed lazily, on first use,
   never at import/app-creation time (see get_iam_client's own
   docstring for why); and
2. tests can override them via FastAPI's `app.dependency_overrides`,
   exactly like every other IAM-consuming service in this workspace
   (see omnibioai-tes's identical `get_identity`/`require_permission`
   pattern) -- so a test proves the real dependency is actually mounted
   on a route rather than only exercising a helper in isolation.

There is no generic "any delegated identity" dependency, and no fallback
to ordinary user or client-credentials validation: a bearer that isn't a
valid `delegated_execution` token for exactly the required permission is
always denied, never re-interpreted as some other credential type.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Header
from iam_client import AsyncIAMClient, DelegatedExecutionIdentity
from iam_client.delegated import require_delegated_execution

# Matches every other IAM_URL consumer in this workspace (omnibioai-tes,
# omnibioai-api-gateway, ...) -- already wired into Studio's compose file
# for this exact service (`toolserver.environment.IAM_URL: http://auth-
# service:8001`), so no deployment/Compose change is needed for this to
# resolve once a delegation-capable image is deployed.
IAM_URL = os.environ.get("IAM_URL", "http://auth-service:8001")

# AsyncIAMClient's constructor requires a redis_url, but the delegated-
# execution path used exclusively by this module never touches it:
# validate_delegated_execution (iam_client/delegated.py) never reads or
# writes the IAM Redis cache -- only get_user/validate_remote/
# get_cached_user do, none of which this file ever calls. Kept
# configurable rather than hardcoded in case a future consumer of this
# same client instance needs it.
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

WORKFLOW_EXECUTE = "workflow.execute"
RUNS_READ = "runs.read"

_client: Optional[AsyncIAMClient] = None


def get_iam_client() -> AsyncIAMClient:
    """Lazy module-level singleton -- constructing AsyncIAMClient must not
    happen at import time (it opens a Redis/httpx client bound to the
    running event loop; same established reasoning as
    omnibioai-tes/service/security/iam.py's identical get_iam_client()),
    and every request should share one client/connection pool rather
    than opening fresh ones per call."""
    global _client
    if _client is None:
        _client = AsyncIAMClient(base_url=IAM_URL, redis_url=REDIS_URL)
    return _client


async def require_workflow_execute(
    authorization: Optional[str] = Header(default=None),
) -> DelegatedExecutionIdentity:
    """Protects run submission/validation. Denies (401) missing/malformed/
    invalid/expired/revoked/wrong-audience/wrong-type delegated
    credentials, and denies (403) a valid delegated identity that lacks
    `workflow.execute` -- both raised by the shared
    iam_client.delegated.require_delegated_execution dependency this
    thinly wraps, not reimplemented here."""
    dependency = require_delegated_execution(get_iam_client(), WORKFLOW_EXECUTE)
    return await dependency(authorization=authorization)


async def require_runs_read(
    authorization: Optional[str] = Header(default=None),
) -> DelegatedExecutionIdentity:
    """Protects run status/logs/results. Same fail-closed/401-vs-403
    behavior as require_workflow_execute, scoped to `runs.read`."""
    dependency = require_delegated_execution(get_iam_client(), RUNS_READ)
    return await dependency(authorization=authorization)
