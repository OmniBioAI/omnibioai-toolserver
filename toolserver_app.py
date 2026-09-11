from __future__ import annotations

import os
import secrets
import time
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from toolserver.adapters.http_tool_executor import make_run, make_validate
from toolserver.executor import Executor
from toolserver.models import RunCreateRequest, RunRecord, ValidateRequest
from toolserver.registry import ToolHandler, ToolRegistry
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
    @app.get("/capabilities")
    def capabilities():
        return registry.capabilities().model_dump()

    # ----------------
    # Validate
    # ----------------
    @app.post("/validate")
    def validate(req: ValidateRequest):
        try:
            h = registry.get(req.tool_id)
        except KeyError as e:
            return {"ok": False, "errors": [{"code": "UNKNOWN_TOOL", "message": str(e)}], "warnings": []}
        return h.validate(req.inputs, req.resources)

    # ----------------
    # Submit run
    # ----------------
    @app.post("/runs")
    def create_run(req: RunCreateRequest):
        # 1) validate first
        v = validate(ValidateRequest(tool_id=req.tool_id, inputs=req.inputs, resources=req.resources))
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
    def get_run(run_id: str):
        rec = store.try_get(run_id)
        if not rec:
            return {"state": "FAILED", "message": "unknown run"}
        return {"run_id": rec.run_id, "state": rec.state, "updated_epoch": rec.updated_epoch}

    # ----------------
    # Logs
    # ----------------
    @app.get("/runs/{run_id}/logs")
    def get_logs(run_id: str, tail: int = 200):
        rec = store.try_get(run_id)
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
    def get_results(run_id: str):
        rec = store.try_get(run_id)
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
    # Register tools (dynamic, zero-config)
    # ----------------
    @app.post("/register_tools")
    def register_tools_endpoint(req: RegisterToolsRequest):
        registered = 0
        for tool_def in req.tools:
            tool_id = tool_def.get("tool_id")
            if not tool_id:
                continue
            if tool_def.get("http"):
                handler = ToolHandler(
                    tool_id=tool_id,
                    validate=make_validate(tool_def),
                    run=make_run(tool_def),
                    version=tool_def.get("version", "v1"),
                    features=tool_def.get("features") or {},
                )
            else:
                def _stub_validate(inputs, resources):
                    return {"ok": True, "errors": [], "warnings": []}

                def _stub_run(inputs, resources, log, _tid=tool_id):
                    raise NotImplementedError(f"Tool {_tid!r} has no http block; cannot execute")

                handler = ToolHandler(
                    tool_id=tool_id,
                    validate=_stub_validate,
                    run=_stub_run,
                    version=tool_def.get("version", "v1"),
                    features=tool_def.get("features") or {},
                )
            registry.register(handler)
            registered += 1
        return {"ok": True, "registered": registered}

    # ----------------
    # Health
    # ----------------
    @app.get("/health")
    def health():
        return {"ok": True, "service": "omnibioai-toolserver"}

    return app