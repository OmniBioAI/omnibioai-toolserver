from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

RunState = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"]


class ValidateRequest(BaseModel):
    tool_id: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    resources: Dict[str, Any] = Field(default_factory=dict)


class RunCreateRequest(BaseModel):
    tool_id: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    resources: Dict[str, Any] = Field(default_factory=dict)


class RunStatusResponse(BaseModel):
    run_id: str
    state: RunState
    updated_epoch: int
    message: Optional[str] = None


class RunRecord(BaseModel):
    run_id: str
    tool_id: str
    state: RunState
    created_epoch: int
    updated_epoch: int

    # HIPAA-V2-001: the authoritative tenant owner, assigned exactly once
    # at creation from the caller's verified `DelegatedExecutionIdentity
    # .organization` (toolserver_app.py::create_run) -- never from request
    # body/query/path, forwarded headers, or run metadata. Optional with
    # a None default -- not because a new protected run may lack one, but
    # so a pre-V2-001 (or otherwise ownerless) on-disk record still
    # deserializes here instead of raising, matching this platform's
    # established "Optional/default None so pre-existing records still
    # model fine" convention (see omnibioai-tes's identical
    # RunRecord.organization_id). A None/empty value is never treated as
    # a wildcard or as implicitly owned by the current reader -- it is a
    # fail-closed deny at the read-authorization layer
    # (toolserver_app.py::_authorized_run), the same way an org mismatch
    # is. Nothing in this codebase ever re-assigns this field after
    # creation (see Executor.submit/work, which always loads and mutates
    # the existing record via RunStore rather than reconstructing one).
    organization_id: Optional[str] = None

    inputs: Dict[str, Any] = Field(default_factory=dict)     # lightweight metadata ok
    resources: Dict[str, Any] = Field(default_factory=dict)

    logs: List[str] = Field(default_factory=list)
    results: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class ToolCapability(BaseModel):
    tool_id: str
    version: Optional[str] = None
    features: Dict[str, Any] = Field(default_factory=dict)


class ServerCapabilities(BaseModel):
    engines: List[str] = Field(default_factory=list)
    tools: List[ToolCapability] = Field(default_factory=list)
    resources: Dict[str, Any] = Field(default_factory=dict)
    storage: Dict[str, Any] = Field(default_factory=dict)
    policies: Dict[str, Any] = Field(default_factory=dict)
