"""
OmniBioAI toolserver.adapters.http_tool_executor.

Purpose:
    Defines make_validate and make_run for toolserver.adapters.http_tool_executor.

Author:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

import httpx


def _resolve(template: str, inputs: Dict) -> str:
    """Substitute {param} placeholders with actual input values."""
    return re.sub(
        r"\{(\w+)\}",
        lambda m: str(inputs.get(m.group(1), m.group(0))),
        str(template),
    )


def _get_nested(data: Any, path: str) -> Any:
    """
    Resolve dot-notation + array index paths from response JSON.
    e.g. "results[0].name"  or  "organism.scientificName"
    Returns None safely if path doesn't exist.
    """
    if not path:
        return data
    for part in re.split(r"\.|\[(\d+)\]", path):
        if not part:
            continue
        if isinstance(data, list):
            try:
                data = data[int(part)]
            except (IndexError, ValueError):
                return None
        elif isinstance(data, dict):
            data = data.get(part)
        else:
            return None
    return data


def _input_fields(tool_def: Dict) -> List[Dict]:
    """Safely get inputs list — handles None and missing key."""
    fields = tool_def.get("inputs") or []
    return fields if isinstance(fields, list) else []


def _response_map(tool_def: Dict) -> Dict:
    """Safely get response_map — handles None and missing key."""
    rmap = tool_def.get("response_map") or {}
    return rmap if isinstance(rmap, dict) else {}


def _parse_response(resp: httpx.Response) -> Any:
    """
    Parse response — tries JSON first, falls back to plain text.
    Handles APIs like KEGG that return text/plain instead of JSON.
    """
    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        return resp.json()
    try:
        return resp.json()
    except Exception:
        return {"text": resp.text}


def make_validate(tool_def: Dict) -> Callable:
    """Return a validate() function bound to this tool_def."""

    def _validate(inputs: Dict[str, Any], resources: Dict[str, Any]) -> Dict[str, Any]:
        errors = []

        for field in _input_fields(tool_def):
            name     = field["name"]
            required = field.get("required", False)
            ftype    = field.get("type", "string")
            value    = inputs.get(name, field.get("default"))

            if required and (value is None or str(value).strip() == ""):
                errors.append({"field": name, "message": f"'{name}' is required."})
                continue

            if value is not None:
                if ftype == "integer" and not isinstance(value, int):
                    errors.append({"field": name, "message": f"'{name}' must be an integer."})
                elif ftype == "number" and not isinstance(value, (int, float)):
                    errors.append({"field": name, "message": f"'{name}' must be a number."})

        return {"ok": len(errors) == 0, "errors": errors, "warnings": []}

    return _validate


def make_run(tool_def: Dict) -> Callable:
    """Return a run() function bound to this tool_def."""

    def _run(
        inputs: Dict[str, Any],
        resources: Dict[str, Any],
        log: Callable[[str], None],
    ) -> Dict[str, Any]:
        http_cfg = tool_def.get("http")
        if not http_cfg or not isinstance(http_cfg, dict):
            raise ValueError(
                f"[{tool_def.get('tool_id', '?')}] tool_def missing 'http' block"
            )

        method  = http_cfg.get("method", "GET").upper()
        timeout = http_cfg.get("timeout", 15)

        if "url" not in http_cfg:
            raise ValueError(
                f"[{tool_def.get('tool_id', '?')}] tool_def.http missing 'url'"
            )

        # Apply input defaults
        resolved: Dict[str, Any] = {}
        for field in _input_fields(tool_def):
            name = field["name"]
            resolved[name] = inputs.get(name, field.get("default", ""))

        # Build URL — resolve {path_param} placeholders
        url = _resolve(http_cfg["url"], resolved)

        # Build query params — preserve native types (int stays int for httpx)
        raw_params = http_cfg.get("params") or {}
        params: Dict[str, Any] = {}
        for k, v in raw_params.items():
            resolved_v = _resolve(str(v), resolved)
            field_def = next((f for f in _input_fields(tool_def) if f["name"] == k), None)
            if field_def and field_def.get("type") == "integer":
                try:
                    params[k] = int(resolved_v)
                except ValueError:
                    params[k] = resolved_v
            else:
                params[k] = resolved_v

        # Skip empty string params — avoids ?synonym= polluting URLs
        params = {k: v for k, v in params.items() if v != ""}

        # Headers
        headers = dict(http_cfg.get("headers") or {})

        # Body (POST only)
        body_type = http_cfg.get("body_type", "json")
        body_map: Dict[str, Any] = {
            k: _resolve(str(v), resolved)
            for k, v in (http_cfg.get("body_map") or {}).items()
        }

        # Log — mask secret fields
        secret_fields = {
            f["name"] for f in _input_fields(tool_def) if f.get("secret")
        }
        safe_inputs = {
            k: ("***" if k in secret_fields else v)
            for k, v in resolved.items()
        }
        log(f"[{tool_def.get('tool_id', '?')}] {method} {url} inputs={safe_inputs}")

        # Execute
        with httpx.Client(timeout=timeout) as client:
            if method == "GET":
                resp = client.get(url, params=params, headers=headers)
            elif method == "POST" and body_type == "form":
                resp = client.post(url, data=body_map, params=params, headers=headers)
            elif method == "POST":
                resp = client.post(url, json=body_map, params=params, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

        resp.raise_for_status()

        # ✅ Handle both JSON and plain text responses (e.g. KEGG returns text/plain)
        raw = _parse_response(resp)

        # Map response fields via dot-path notation
        result: Dict[str, Any] = {"raw": raw}
        for out_key, json_path in _response_map(tool_def).items():
            result[out_key] = _get_nested(raw, str(json_path) if json_path else "")

        log(f"[{tool_def.get('tool_id', '?')}] completed OK")
        return result

    return _run