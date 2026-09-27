from __future__ import annotations

import os
import secrets
import time
from typing import Any, Dict, List

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from iam_client import DelegatedExecutionIdentity, ServiceRegistrationIdentity
from pydantic import BaseModel

from toolserver.adapters.http_tool_executor import make_run, make_validate
from toolserver.executor import Executor
from toolserver.models import RunCreateRequest, RunRecord, ValidateRequest
from toolserver.registry import ToolHandler, ToolRegistry
from toolserver.security import require_runs_read, require_tes_registration, require_workflow_execute
from toolserver.store import RunStore
from toolserver.tools import load_tools_from_yaml, register_tools


class RegisterToolsRequest(BaseModel):
    tools: List[Dict[str, Any]]


def _new_run_id() -> str:
    return f"ts_{secrets.token_urlsafe(10)}"


def create_app() -> FastAPI:
    app = FastAPI(title="omnibioai-toolserver", root_path="/_svc/toolserver")

    run_store_dir = "out/runs"
    store = RunStore(run_store_dir)

    registry = ToolRegistry()

    # 1) Register legacy enrichr_pathway handler (keeps existing behaviour)
    register_tools(registry)

    # 2) Auto-register all YAML-declared HTTP tools (zero Python per tool)
    tools_yaml = os.environ.get("TOOLS_YAML_PATH", "configs/tools.example.yaml")
    if os.path.exists(tools_yaml):
        load_tools_from_yaml(registry, tools_yaml)
    else:
        print(f"[toolserver] WARNING: tools YAML not found at '{tools_yaml}' — only legacy tools loaded")

    executor = Executor(store=store, registry=registry, max_workers=8)

    # ----------------
    # Capabilities
    # ----------------
    # Public/unauthenticated by design: this is server metadata (which
    # engines/tool_ids/versions/feature flags are registered), not
    # tenant, run, or PHI data -- no user/organization/execution context
    # is exposed here, so there is nothing for delegated authorization to
    # gate. Kept explicitly out of the protected surface, matching
    # HIPAA-V2-019 Section 15's "retain only deliberately low-sensitivity
    # health/readiness endpoints without authentication."
    @app.get("/capabilities")
    def capabilities():
        return registry.capabilities().model_dump()

    def _validate_tool(req: ValidateRequest) -> Dict[str, Any]:
        try:
            h = registry.get(req.tool_id)
        except KeyError as e:
            return {"ok": False, "errors": [{"code": "UNKNOWN_TOOL", "message": str(e)}], "warnings": []}
        return h.validate(req.inputs, req.resources)

    # HIPAA-V2-001: single authoritative gate for every tenant-owned read
    # route (status/logs/results). Returns the RunRecord only when it
    # both exists AND its persisted `organization_id` exactly matches the
    # caller's verified delegated identity; returns None for every other
    # case -- unknown run_id, wrong-tenant run, and missing/empty/
    # malformed ownership are all folded into the exact same None result,
    # deliberately indistinguishable to the caller (this platform's
    # established anti-enumeration posture -- see e.g. omnibioai-tes's
    # _require_same_workspace, which does the same "mismatch looks
    # exactly like not-found" fold). Callers must render None using
    # whichever "unknown run" shape that specific route already used
    # before V2-001 -- this function does not decide the HTTP response.
    #
    # `not rec.organization_id` (falsy) catches both None and "" in one
    # check; anything else that isn't an exact match to
    # `identity.organization` denies via the equality check right after.
    # A legacy/ownerless record can never become readable merely because
    # the caller holds a valid runs.read permission -- ownership is a
    # separate, mandatory condition, not something permission satisfies.
    def _authorized_run(run_id: str, identity: DelegatedExecutionIdentity):
        rec = store.try_get(run_id)
        if not rec:
            return None
        if not rec.organization_id or rec.organization_id != identity.organization:
            return None
        return rec

    # ----------------
    # Validate
    # ----------------
    @app.post("/validate")
    def validate(
        req: ValidateRequest,
        identity: DelegatedExecutionIdentity = Depends(require_workflow_execute),
    ):
        return _validate_tool(req)

    # ----------------
    # Submit run
    # ----------------
    @app.post("/runs")
    def create_run(
        req: RunCreateRequest,
        identity: DelegatedExecutionIdentity = Depends(require_workflow_execute),
    ):
        # 1) validate first (shared helper, not the /validate route
        # function itself -- that now carries its own Depends(...) and
        # cannot be called as a plain Python function).
        v = _validate_tool(ValidateRequest(tool_id=req.tool_id, inputs=req.inputs, resources=req.resources))
        if not v.get("ok", False):
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": {"code": "VALIDATION_FAILED", "details": v}},
            )

        run_id = _new_run_id()
        now = int(time.time())

        rec = RunRecord(
            run_id=run_id,
            tool_id=req.tool_id,
            state="QUEUED",
            created_epoch=now,
            updated_epoch=now,
            # HIPAA-V2-001: the ONLY place a run's authoritative owner is
            # ever assigned, and the only acceptable source for it --
            # the caller's own verified delegated identity, never
            # anything from `req` (RunCreateRequest has no organization
            # field at all; even if one were added, or a caller stuffs an
            # "organization_id"-shaped key into `inputs`/`resources`,
            # nothing here ever reads it for ownership purposes).
            organization_id=identity.organization,
            inputs={"summary": "stored externally"},  # keep record lightweight
            resources=req.resources,
            logs=["Queued"],
            results=None,
            error=None,
        )
        store.create(rec)

        # 2) async execution — full inputs passed directly to executor
        executor.submit(store.get(run_id), full_inputs=req.inputs, resources=req.resources)

        return {"run_id": run_id}

    # ----------------
    # Status
    # ----------------
    @app.get("/runs/{run_id}")
    def get_run(
        run_id: str,
        identity: DelegatedExecutionIdentity = Depends(require_runs_read),
    ):
        rec = _authorized_run(run_id, identity)
        if not rec:
            return {"state": "FAILED", "message": "unknown run"}
        return {"run_id": rec.run_id, "state": rec.state, "updated_epoch": rec.updated_epoch}

    # ----------------
    # Logs
    # ----------------
    @app.get("/runs/{run_id}/logs")
    def get_logs(
        run_id: str,
        tail: int = 200,
        identity: DelegatedExecutionIdentity = Depends(require_runs_read),
    ):
        rec = _authorized_run(run_id, identity)
        if not rec:
            return {"run_id": run_id, "logs": f"[{run_id}] unknown run"}
        lines = rec.logs or []
        if tail and tail > 0:
            lines = lines[-tail:]
        return {"run_id": run_id, "logs": "\n".join(lines)}

    # ----------------
    # Results
    # ----------------
    @app.get("/runs/{run_id}/results")
    def get_results(
        run_id: str,
        identity: DelegatedExecutionIdentity = Depends(require_runs_read),
    ):
        rec = _authorized_run(run_id, identity)
        if not rec:
            return {"ok": False, "error": {"code": "NOT_FOUND", "message": "unknown run"}}
        if rec.state != "COMPLETED":
            return {
                "ok": False,
                "error": {"code": "NOT_READY", "message": f"state={rec.state}"},
                "state": rec.state,
            }
        return rec.results or {"ok": True, "results": {}}

    # ----------------
    # Register tools -- TES service identity only
    # ----------------
    # HIPAA-V2-019: registering HTTP-tool definitions is an administrative,
    # code-execution-adjacent capability (a registered handler is later run
    # by /runs), so it is not a workflow.execute/runs.read operation and is
    # never reachable with a user or delegated-execution credential. It
    # requires Auth's service-only `toolserver_registration` credential
    # (scope toolserver.register, ToolServer audience, no user principal)
    # from a service identity listed in TOOLSERVER_REGISTRATION_CLIENT_IDS
    # -- in production, exactly TES, which registers its declarative HTTP
    # tools once at startup. Only tools with an `http` block are accepted;
    # the credential is never logged.
    @app.post("/register_tools")
    def register_tools_endpoint(
        req: RegisterToolsRequest,
        identity: ServiceRegistrationIdentity = Depends(require_tes_registration),  # noqa: B008
    ):
        registered = 0
        skipped = 0
        for tool_def in req.tools:
            tool_id = tool_def.get("tool_id")
            if not tool_id or not isinstance(tool_def.get("http"), dict):
                skipped += 1
                continue
            registry.register(
                ToolHandler(
                    tool_id=tool_id,
                    validate=make_validate(tool_def),
                    run=make_run(tool_def),
                    version=tool_def.get("version", "v1"),
                    features=tool_def.get("features") or {},
                )
            )
            registered += 1
        print(
            f"[toolserver] register_tools: service={identity.calling_service} "
            f"registered={registered} skipped={skipped}"
        )
        return {"ok": True, "registered": registered, "skipped": skipped}

    # ----------------
    # Health
    # ----------------
    @app.get("/health")
    def health():
        return {"ok": True, "service": "omnibioai-toolserver"}

    return app