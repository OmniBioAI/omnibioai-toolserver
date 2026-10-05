"""
OmniBioAI toolserver.registry.

Purpose:
    Defines ToolHandler and ToolRegistry for toolserver.registry.

Author:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from .models import ServerCapabilities

ValidateFn = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
RunFn = Callable[[Dict[str, Any], Dict[str, Any], Callable[[str], None]], Dict[str, Any]]


@dataclass(frozen=True)
class ToolHandler:
    tool_id: str
    validate: ValidateFn
    run: RunFn
    version: str = "v1"
    features: Dict[str, Any] = None  # advertised via capabilities


class ToolRegistry:
    def __init__(self) -> None:
        self._handlers: Dict[str, ToolHandler] = {}

    def register(self, h: ToolHandler) -> None:
        self._handlers[h.tool_id] = h

    def get(self, tool_id: str) -> ToolHandler:
        if tool_id not in self._handlers:
            raise KeyError(f"tool not registered: {tool_id}")
        return self._handlers[tool_id]

    def capabilities(self) -> ServerCapabilities:
        tools = []
        for h in self._handlers.values():
            tools.append(
                {
                    "tool_id": h.tool_id,
                    "version": h.version,
                    "features": h.features or {},
                }
            )

        return ServerCapabilities(
            engines=["http_toolserver"],
            tools=tools,  # pydantic will coerce dict -> ToolCapability
            resources={"max_cpu": 8, "max_ram_gb": 32},
            storage={"allowed_uri_schemes": ["inline"]},
            policies={"max_runtime_minutes": 60},
        )
