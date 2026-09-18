"""Tests for toolserver/adapters/http_tool_executor.py.

Covers the generic HTTP-based tool adapter that ToolServer uses to invoke
externally-defined "http" tools from tool_def configuration: placeholder
substitution (_resolve), dot/array-index response-path extraction
(_get_nested), input validation (make_validate), and request execution
(make_run) -- URL/param/body construction, GET/POST dispatch, response
parsing (including the non-JSON fallback), secret-field log masking, and
the config-error guard clauses that reject malformed tool definitions.

Developer:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

import pytest
import httpx
from unittest.mock import MagicMock, patch, call
from typing import Any, Dict

# ── Import the module under test ──────────────────────────────────────────────
# Adjust the import path to match your project structure, e.g.:
# from toolserver.tool_runner import _resolve, _get_nested, make_validate, make_run
from toolserver.adapters.http_tool_executor import _resolve, _get_nested, _parse_response, make_validate, make_run


# ═════════════════════════════════════════════════════════════════════════════
# _resolve
# ═════════════════════════════════════════════════════════════════════════════

class TestResolve:
    """Substitute {param} placeholders in a template string with resolved
    input values, leaving unmatched placeholders untouched."""

    def test_single_placeholder(self):
        """A single {name} placeholder is substituted with its input value."""
        assert _resolve("Hello, {name}!", {"name": "World"}) == "Hello, World!"

    def test_multiple_placeholders(self):
        """Multiple distinct placeholders in one template are all substituted."""
        result = _resolve("{method} {resource} v{version}", {
            "method": "GET", "resource": "genes", "version": 2
        })
        assert result == "GET genes v2"

    def test_missing_key_leaves_placeholder(self):
        """If a key is absent the original placeholder is preserved."""
        assert _resolve("Hello, {name}!", {}) == "Hello, {name}!"

    def test_integer_value_is_stringified(self):
        """An integer input value is stringified when substituted into the template."""
        assert _resolve("/page/{page}", {"page": 3}) == "/page/3"

    def test_no_placeholders(self):
        """A template with no placeholders is returned unchanged."""
        assert _resolve("/api/v1/health", {}) == "/api/v1/health"

    def test_empty_template(self):
        """An empty template string resolves to an empty string."""
        assert _resolve("", {"key": "val"}) == ""

    def test_partial_substitution(self):
        """Placeholders with a matching input are substituted; placeholders
        without a matching key are left as-is."""
        result = _resolve("{a}/{b}/{c}", {"a": "x", "c": "z"})
        assert result == "x/{b}/z"


# ═════════════════════════════════════════════════════════════════════════════
# _get_nested
# ═════════════════════════════════════════════════════════════════════════════

class TestGetNested:
    """Resolve dot-notation and array-index paths (e.g. "results[0].name")
    against parsed response JSON, returning None for any path segment that
    doesn't exist rather than raising."""

    def test_empty_path_returns_data(self):
        """An empty path returns the input data object itself, unchanged."""
        data = {"key": "value"}
        assert _get_nested(data, "") is data

    def test_simple_key(self):
        """A single top-level dict key is resolved directly."""
        assert _get_nested({"name": "gene1"}, "name") == "gene1"

    def test_nested_dot_path(self):
        """Dot-separated keys traverse nested dicts."""
        data = {"organism": {"scientificName": "Homo sapiens"}}
        assert _get_nested(data, "organism.scientificName") == "Homo sapiens"

    def test_array_index(self):
        """A bracketed index selects an element from a list."""
        data = {"results": ["a", "b", "c"]}
        assert _get_nested(data, "results[1]") == "b"

    def test_array_index_with_nested_key(self):
        """An array index can be followed by a dotted key into the selected element."""
        data = {"results": [{"name": "gene1"}, {"name": "gene2"}]}
        assert _get_nested(data, "results[0].name") == "gene1"

    def test_out_of_bounds_index_returns_none(self):
        """An out-of-range array index returns None instead of raising IndexError."""
        data = {"results": ["only_one"]}
        assert _get_nested(data, "results[5]") is None

    def test_missing_dict_key_returns_none(self):
        """A dict key that isn't present returns None instead of raising KeyError."""
        assert _get_nested({"a": 1}, "b") is None

    def test_deeply_nested(self):
        """Multiple chained dot-separated keys resolve through several nesting levels."""
        data = {"a": {"b": {"c": {"d": 42}}}}
        assert _get_nested(data, "a.b.c.d") == 42

    def test_path_on_non_container_returns_none(self):
        """Applying a path to a non-dict, non-list value returns None."""
        assert _get_nested("just_a_string", "key") is None

    def test_none_data_with_path_returns_none(self):
        """Applying a non-empty path to None data returns None instead of raising."""
        assert _get_nested(None, "key") is None


# ═════════════════════════════════════════════════════════════════════════════
# make_validate
# ═════════════════════════════════════════════════════════════════════════════

TOOL_DEF_VALIDATE = {
    "tool_id": "test_tool",
    "inputs": [
        {"name": "query",  "type": "string",  "required": True},
        {"name": "limit",  "type": "integer", "required": False, "default": 10},
        {"name": "score",  "type": "number",  "required": False},
        {"name": "optional", "type": "string", "required": False},
    ],
}


class TestMakeValidate:
    """Validate tool inputs against a tool_def's declared fields: required-ness,
    blank-string rejection for required strings, and integer/number type checks."""

    @pytest.fixture
    def validate(self):
        return make_validate(TOOL_DEF_VALIDATE)

    def test_valid_inputs_pass(self, validate):
        """Inputs satisfying all required/type constraints validate with no errors."""
        result = validate({"query": "BRCA1", "limit": 5}, {})
        assert result["ok"] is True
        assert result["errors"] == []

    def test_missing_required_field_fails(self, validate):
        """Omitting a required field fails validation with an error naming that field."""
        result = validate({}, {})
        assert result["ok"] is False
        assert any(e["field"] == "query" for e in result["errors"])

    def test_blank_string_for_required_fails(self, validate):
        """A required string field containing only whitespace fails validation
        as if the field were missing."""
        result = validate({"query": "   "}, {})
        assert result["ok"] is False

    def test_wrong_type_integer_fails(self, validate):
        """A non-integer value for an integer-typed field fails validation
        with an error naming that field."""
        result = validate({"query": "BRCA1", "limit": "not_an_int"}, {})
        assert result["ok"] is False
        assert any(e["field"] == "limit" for e in result["errors"])

    def test_wrong_type_number_fails(self, validate):
        """A non-numeric value for a number-typed field fails validation
        with an error naming that field."""
        result = validate({"query": "BRCA1", "score": "high"}, {})
        assert result["ok"] is False
        assert any(e["field"] == "score" for e in result["errors"])

    def test_integer_accepted_for_number_field(self, validate):
        """An integer value is accepted for a number-typed field."""
        result = validate({"query": "BRCA1", "score": 7}, {})
        assert result["ok"] is True

    def test_float_accepted_for_number_field(self, validate):
        """A float value is accepted for a number-typed field."""
        result = validate({"query": "BRCA1", "score": 3.14}, {})
        assert result["ok"] is True

    def test_optional_field_missing_is_ok(self, validate):
        """Omitting a non-required field with no default still validates successfully."""
        result = validate({"query": "BRCA1"}, {})
        assert result["ok"] is True

    def test_default_satisfies_optional_field(self, validate):
        """limit has a default — omitting it should still be fine."""
        result = validate({"query": "BRCA1"}, {})
        assert result["ok"] is True

    def test_no_inputs_defined(self):
        """A tool_def with an empty inputs list validates any input dict as OK."""
        validate = make_validate({"tool_id": "empty", "inputs": []})
        assert validate({}, {})["ok"] is True

    def test_warnings_always_empty(self, validate):
        """Validation never populates the warnings list."""
        result = validate({"query": "test"}, {})
        assert result["warnings"] == []

    def test_multiple_errors_reported(self, validate):
        """Both a required-string error and a type error can surface together."""
        bad = make_validate({
            "tool_id": "t",
            "inputs": [
                {"name": "a", "type": "string",  "required": True},
                {"name": "b", "type": "integer", "required": True},
            ],
        })
        result = bad({"b": "oops"}, {})
        assert result["ok"] is False
        # 'a' is missing AND 'b' has wrong type
        fields = [e["field"] for e in result["errors"]]
        assert "a" in fields
        assert "b" in fields


# ═════════════════════════════════════════════════════════════════════════════
# make_run  –  shared fixtures
# ═════════════════════════════════════════════════════════════════════════════

BASE_TOOL_DEF: Dict[str, Any] = {
    "tool_id": "gene_search",
    "inputs": [
        {"name": "query", "type": "string", "required": True, "default": ""},
        {"name": "limit", "type": "integer", "required": False, "default": 10},
    ],
    "http": {
        "method": "GET",
        "url": "https://api.example.com/genes",
        "params": {"q": "{query}", "limit": "{limit}"},
        "headers": {"Accept": "application/json"},
        "timeout": 5,
    },
    "response_map": {
        "names": "results[0].name",
        "total": "metadata.total",
    },
}


def _make_mock_response(json_data: Any, status_code: int = 200) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ═════════════════════════════════════════════════════════════════════════════
# make_run  –  GET tests
# ═════════════════════════════════════════════════════════════════════════════

class TestMakeRunGet:
    """Execute a GET-method tool: build the resolved URL/params, apply input
    defaults, mask secret fields in the log line, and map the raw response
    through response_map."""

    @pytest.fixture
    def run(self):
        return make_run(BASE_TOOL_DEF)

    @pytest.fixture
    def log(self):
        return MagicMock()

    def _patch_get(self, json_data):
        mock_resp = _make_mock_response(json_data)
        return patch("httpx.Client.get", return_value=mock_resp)

    def test_get_returns_raw_and_mapped(self, run, log):
        """The parsed response is returned unmapped under 'raw', and also
        mapped through response_map into the tool's own named output fields."""
        json_data = {
            "results": [{"name": "BRCA1"}],
            "metadata": {"total": 42},
        }
        with self._patch_get(json_data):
            result = run({"query": "BRCA1", "limit": 5}, {}, log)

        assert result["raw"] == json_data
        assert result["names"] == "BRCA1"
        assert result["total"] == 42

    def test_get_url_placeholder_resolved(self, run, log):
        """A {placeholder} embedded in the URL template is resolved from the
        validated input values before the GET request is issued."""
        tool_def = {
            **BASE_TOOL_DEF,
            "http": {
                **BASE_TOOL_DEF["http"],
                "url": "https://api.example.com/genes/{query}",
                "params": {},
            },
        }
        run2 = make_run(tool_def)
        json_data = {"results": [], "metadata": {"total": 0}}
        mock_resp = _make_mock_response(json_data)

        with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
            run2({"query": "TP53", "limit": 10}, {}, log)
            called_url = mock_get.call_args[0][0]
            assert "TP53" in called_url

    def test_default_values_applied(self, run, log):
        """An omitted optional input falls back to its declared default when
        the request params are built."""
        json_data = {"results": [], "metadata": {"total": 0}}
        mock_resp = _make_mock_response(json_data)

        with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
            run({"query": "MYC"}, {}, log)   # limit not provided → default 10
            _, kwargs = mock_get.call_args
            assert kwargs["params"]["limit"] == 10

    def test_log_called_twice(self, run, log):
        """A successful run logs exactly twice: once for the outgoing
        request and once for the received response."""
        with self._patch_get({"results": [], "metadata": {}}):
            run({"query": "X"}, {}, log)
        assert log.call_count == 2

    def test_log_masks_secret_fields(self, log):
        """An input field flagged secret=True has its value replaced with
        '***' in the log line, so it never appears in plaintext in logs."""
        tool_def = {
            **BASE_TOOL_DEF,
            "inputs": [
                {"name": "api_key", "type": "string", "required": True, "secret": True, "default": ""},
                {"name": "query",   "type": "string", "required": True, "default": ""},
            ],
        }
        run = make_run(tool_def)
        with self._patch_get({}):
            run({"api_key": "supersecret", "query": "gene1"}, {}, log)

        first_log_call = log.call_args_list[0][0][0]
        assert "supersecret" not in first_log_call
        assert "***" in first_log_call

    def test_raise_for_status_called(self, run, log):
        """The response's raise_for_status() is always invoked, so a non-2xx
        HTTP status is rejected rather than treated as a successful result."""
        mock_resp = _make_mock_response({})
        with patch("httpx.Client.get", return_value=mock_resp):
            run({"query": "X"}, {}, log)
        mock_resp.raise_for_status.assert_called_once()

    def test_http_error_propagates(self, run, log):
        """An HTTPStatusError raised by raise_for_status() propagates out of
        run() to the caller rather than being swallowed."""
        mock_resp = _make_mock_response({}, status_code=404)
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_resp
        )
        with patch("httpx.Client.get", return_value=mock_resp):
            with pytest.raises(httpx.HTTPStatusError):
                run({"query": "X"}, {}, log)

    def test_response_map_missing_path_returns_none(self, run, log):
        """If response_map path doesn't exist in raw, value should be None."""
        with self._patch_get({"results": [], "metadata": {}}):
            result = run({"query": "X"}, {}, log)
        assert result["names"] is None
        assert result["total"] is None


# ═════════════════════════════════════════════════════════════════════════════
# make_run  –  POST tests
# ═════════════════════════════════════════════════════════════════════════════

POST_TOOL_DEF: Dict[str, Any] = {
    "tool_id": "create_job",
    "inputs": [
        {"name": "sequence", "type": "string", "required": True, "default": ""},
        {"name": "model",    "type": "string", "required": False, "default": "v1"},
    ],
    "http": {
        "method": "POST",
        "url": "https://api.example.com/jobs",
        "body_type": "json",
        "body_map": {"seq": "{sequence}", "model": "{model}"},
        "params": {},
        "headers": {},
        "timeout": 10,
    },
    "response_map": {"job_id": "id"},
}

POST_FORM_TOOL_DEF: Dict[str, Any] = {
    **POST_TOOL_DEF,
    "tool_id": "create_job_form",
    "http": {**POST_TOOL_DEF["http"], "body_type": "form"},
}


class TestMakeRunPost:
    """Execute a POST-method tool: build the request body via body_map
    {placeholder} substitution, honor body_type (json/form) when choosing
    how to send it, and map the response through response_map."""

    @pytest.fixture
    def log(self):
        return MagicMock()

    def test_post_json_body_sent(self, log):
        """body_type 'json' sends the body_map-substituted body as the
        request's JSON payload, and the response is mapped as usual."""
        run = make_run(POST_TOOL_DEF)
        mock_resp = _make_mock_response({"id": "job-123"})

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            result = run({"sequence": "ATCG", "model": "v2"}, {}, log)
            _, kwargs = mock_post.call_args
            assert kwargs["json"] == {"seq": "ATCG", "model": "v2"}

        assert result["job_id"] == "job-123"

    def test_post_form_body_sent(self, log):
        """body_type 'form' sends the body_map-substituted body via the
        form-encoded 'data' kwarg instead of 'json'."""
        run = make_run(POST_FORM_TOOL_DEF)
        mock_resp = _make_mock_response({"id": "job-456"})

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            run({"sequence": "GCTA", "model": "v1"}, {}, log)
            _, kwargs = mock_post.call_args
            assert "data" in kwargs
            assert kwargs["data"]["seq"] == "GCTA"

    def test_post_default_body_type_is_json(self, log):
        """Omitting body_type from the http block defaults the POST body
        encoding to JSON."""
        tool_def = {
            **POST_TOOL_DEF,
            "http": {k: v for k, v in POST_TOOL_DEF["http"].items() if k != "body_type"},
        }
        run = make_run(tool_def)
        mock_resp = _make_mock_response({"id": "x"})

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            run({"sequence": "TTTT"}, {}, log)
            _, kwargs = mock_post.call_args
            assert "json" in kwargs


# ═════════════════════════════════════════════════════════════════════════════
# make_run  –  unsupported method
# ═════════════════════════════════════════════════════════════════════════════

class TestMakeRunUnsupportedMethod:
    """An http.method outside the supported GET/POST set is rejected before
    any request is attempted, rather than silently falling through."""

    def test_unsupported_method_raises(self):
        """A DELETE (or any unsupported) HTTP method raises ValueError
        naming the unsupported method, before any request is sent."""
        tool_def = {
            **BASE_TOOL_DEF,
            "http": {**BASE_TOOL_DEF["http"], "method": "DELETE"},
        }
        run = make_run(tool_def)
        with pytest.raises(ValueError, match="Unsupported HTTP method"):
            run({"query": "X"}, {}, MagicMock())


# ═════════════════════════════════════════════════════════════════════════════
# make_run  –  default method fallback
# ═════════════════════════════════════════════════════════════════════════════

class TestMakeRunDefaultMethod:
    """Omitting 'method' from the http block defaults the request to GET
    rather than raising or refusing to run."""

    def test_default_method_is_get(self):
        """With no 'method' key in the http block, the run still dispatches
        the request via httpx.Client.get."""
        tool_def = {
            **BASE_TOOL_DEF,
            "http": {k: v for k, v in BASE_TOOL_DEF["http"].items() if k != "method"},
        }
        run = make_run(tool_def)
        mock_resp = _make_mock_response({})

        with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
            run({"query": "X"}, {}, MagicMock())
            assert mock_get.called


# ═════════════════════════════════════════════════════════════════════════════
# _parse_response  (lines 59-64)
# ═════════════════════════════════════════════════════════════════════════════

class TestParseResponse:
    """Parse an httpx Response body into a dict: an application/json
    content-type or a response whose body still parses as JSON returns the
    parsed JSON; a genuine JSON-parse failure falls back to raw text."""

    def test_json_content_type_returns_parsed_json(self):
        """Line 60: content-type is application/json → early return via resp.json()."""
        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "application/json; charset=utf-8"}
        mock_resp.json.return_value = {"key": "value"}
        assert _parse_response(mock_resp) == {"key": "value"}

    def test_non_json_content_type_falls_back_to_json(self):
        """Line 62: content-type is not application/json but json() still succeeds."""
        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.json.return_value = {"data": 42}
        assert _parse_response(mock_resp) == {"data": 42}

    def test_json_parse_failure_returns_text(self):
        """Lines 63-64: json() raises → fall back to {'text': resp.text}."""
        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.json.side_effect = Exception("not valid JSON")
        mock_resp.text = "gene\tname\nTP53\ttumour protein"
        result = _parse_response(mock_resp)
        assert result == {"text": "gene\tname\nTP53\ttumour protein"}


# ═════════════════════════════════════════════════════════════════════════════
# make_run — config-error guard-clauses (lines 103-114)
# ═════════════════════════════════════════════════════════════════════════════

class TestMakeRunConfigErrors:
    """A malformed or absent 'http' block in a tool_def is rejected at
    run-time with ValueError, before any network request is attempted."""

    def test_raises_when_http_block_absent(self):
        """Line 104: tool_def has no 'http' key → ValueError."""
        run = make_run({"tool_id": "no_http"})
        with pytest.raises(ValueError, match="missing 'http' block"):
            run({}, {}, MagicMock())

    def test_raises_when_http_block_is_none(self):
        """Line 104: tool_def['http'] = None → ValueError."""
        run = make_run({"tool_id": "null_http", "http": None})
        with pytest.raises(ValueError, match="missing 'http' block"):
            run({}, {}, MagicMock())

    def test_raises_when_http_block_not_a_dict(self):
        """Line 104: tool_def['http'] is a non-dict truthy value → ValueError."""
        run = make_run({"tool_id": "bad_http", "http": "GET /genes"})
        with pytest.raises(ValueError, match="missing 'http' block"):
            run({}, {}, MagicMock())

    def test_raises_when_url_missing_from_http_block(self):
        """Line 112: http block exists but has no 'url' key → ValueError."""
        run = make_run({"tool_id": "no_url", "http": {"method": "GET"}})
        with pytest.raises(ValueError, match="missing 'url'"):
            run({}, {}, MagicMock())


# ═════════════════════════════════════════════════════════════════════════════
# make_run — integer param conversion failure (lines 132-135)
# ═════════════════════════════════════════════════════════════════════════════

class TestMakeRunIntegerParamFallback:
    """An integer-typed param whose input value fails int() conversion is
    not dropped or rejected -- it's kept in the request as its original
    string value."""

    def test_non_numeric_integer_param_kept_as_string(self):
        """Lines 134-135: int() conversion fails → param value kept as string."""
        tool_def = {
            "tool_id": "int_test",
            "inputs": [{"name": "count", "type": "integer", "required": False, "default": ""}],
            "http": {
                "method": "GET",
                "url": "https://api.example.com/data",
                "params": {"count": "{count}"},
            },
            "response_map": {},
        }
        run = make_run(tool_def)
        mock_resp = _make_mock_response({})

        with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
            run({"count": "not-a-number"}, {}, MagicMock())
            _, kwargs = mock_get.call_args
            assert kwargs["params"]["count"] == "not-a-number"