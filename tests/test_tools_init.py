"""Tests for toolserver/tools/__init__.py, the tool registry bootstrap.

Covers register_tools's registration of the two built-in tools
(enrichr_pathway, david_annotation) and load_tools_from_yaml's parsing of
a YAML tool catalog into registered HTTP-tool handlers, including its
file-not-found handling and its rules for skipping catalog entries that
are missing a tool_id or an http block.

Developer:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from toolserver.tools import load_tools_from_yaml, register_tools


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _make_registry():
    """Return a simple mock registry with a .register() recorder."""
    registry = MagicMock()
    registry.registered = []
    registry.register.side_effect = lambda handler: registry.registered.append(handler)
    return registry


def _write_yaml(tmp_path: Path, data) -> Path:
    p = tmp_path / "tools.yaml"
    p.write_text(yaml.dump(data))
    return p


# ═════════════════════════════════════════════════════════════════════════════
# register_tools
# ═════════════════════════════════════════════════════════════════════════════

class TestRegisterTools:
    """register_tools's registration of the two built-in tool handlers."""

    def test_registers_enrichr_pathway(self):
        """The built-in enrichr_pathway tool is registered under that tool_id."""
        registry = _make_registry()
        register_tools(registry)

        ids = [h.tool_id for h in registry.registered]
        assert "enrichr_pathway" in ids

    def test_registers_david_annotation(self):
        """The built-in david_annotation tool is registered under that tool_id."""
        registry = _make_registry()
        register_tools(registry)

        ids = [h.tool_id for h in registry.registered]
        assert "david_annotation" in ids

    def test_registers_both_builtin_tools(self):
        """Exactly the two built-in tools are registered, no more and no fewer."""
        registry = _make_registry()
        register_tools(registry)

        assert registry.register.call_count == 2

    def test_enrichr_pathway_has_correct_version(self):
        """The enrichr_pathway handler is registered with version v1."""
        registry = _make_registry()
        register_tools(registry)
        assert registry.registered[0].version == "v1"

    def test_enrichr_pathway_has_default_libraries_feature(self):
        """The enrichr_pathway handler's default feature set includes its default libraries."""
        registry = _make_registry()
        register_tools(registry)
        features = registry.registered[0].features
        assert "libraries_default" in features
        assert "WikiPathways_2024_Human" in features["libraries_default"]
        assert "Reactome_2022" in features["libraries_default"]

    def test_enrichr_pathway_validate_and_run_are_callable(self):
        """The registered enrichr_pathway handler exposes callable validate and run hooks."""
        registry = _make_registry()
        register_tools(registry)
        handler = registry.registered[0]
        assert callable(handler.validate)
        assert callable(handler.run)


# ═════════════════════════════════════════════════════════════════════════════
# load_tools_from_yaml — file-not-found
# ═════════════════════════════════════════════════════════════════════════════

class TestLoadToolsFromYamlFileNotFound:
    """load_tools_from_yaml's error handling when the YAML catalog file is missing."""

    def test_raises_file_not_found(self, tmp_path):
        """A missing YAML catalog path raises FileNotFoundError with a descriptive message."""
        registry = _make_registry()
        with pytest.raises(FileNotFoundError, match="tools YAML not found"):
            load_tools_from_yaml(registry, str(tmp_path / "missing.yaml"))

    def test_error_message_contains_path(self, tmp_path):
        """The FileNotFoundError message includes the specific missing file's path."""
        registry = _make_registry()
        bad_path = str(tmp_path / "no_such_file.yaml")
        with pytest.raises(FileNotFoundError, match="no_such_file.yaml"):
            load_tools_from_yaml(registry, bad_path)


# ═════════════════════════════════════════════════════════════════════════════
# load_tools_from_yaml — happy paths
# ═════════════════════════════════════════════════════════════════════════════

MINIMAL_HTTP_TOOL = {
    "tool_id": "gene_search",
    "version": "v2",
    "features": {"max_results": 100},
    "http": {
        "method": "GET",
        "url": "https://api.example.com/genes",
        "params": {"q": "{query}"},
        "timeout": 5,
    },
    "inputs": [{"name": "query", "type": "string", "required": True}],
    "response_map": {},
}


class TestLoadToolsFromYamlHappyPath:
    """load_tools_from_yaml's registration of well-formed HTTP-tool catalog entries."""

    def test_registers_single_http_tool(self, tmp_path):
        """A single well-formed HTTP-tool entry is registered under its tool_id."""
        registry = _make_registry()
        p = _write_yaml(tmp_path, [MINIMAL_HTTP_TOOL])
        load_tools_from_yaml(registry, str(p))

        assert len(registry.registered) == 1
        assert registry.registered[0].tool_id == "gene_search"

    def test_registered_handler_version(self, tmp_path):
        """The registered handler's version matches the catalog entry's version field."""
        registry = _make_registry()
        p = _write_yaml(tmp_path, [MINIMAL_HTTP_TOOL])
        load_tools_from_yaml(registry, str(p))
        assert registry.registered[0].version == "v2"

    def test_registered_handler_features(self, tmp_path):
        """The registered handler's features match the catalog entry's features field."""
        registry = _make_registry()
        p = _write_yaml(tmp_path, [MINIMAL_HTTP_TOOL])
        load_tools_from_yaml(registry, str(p))
        assert registry.registered[0].features == {"max_results": 100}

    def test_validate_and_run_are_callable(self, tmp_path):
        """A handler built from a YAML entry exposes callable validate and run hooks."""
        registry = _make_registry()
        p = _write_yaml(tmp_path, [MINIMAL_HTTP_TOOL])
        load_tools_from_yaml(registry, str(p))
        handler = registry.registered[0]
        assert callable(handler.validate)
        assert callable(handler.run)

    def test_registers_multiple_http_tools(self, tmp_path):
        """Multiple HTTP-tool entries are all registered, preserving catalog order."""
        tools = [
            {**MINIMAL_HTTP_TOOL, "tool_id": "tool_a"},
            {**MINIMAL_HTTP_TOOL, "tool_id": "tool_b"},
            {**MINIMAL_HTTP_TOOL, "tool_id": "tool_c"},
        ]
        registry = _make_registry()
        p = _write_yaml(tmp_path, tools)
        load_tools_from_yaml(registry, str(p))
        assert len(registry.registered) == 3
        ids = [h.tool_id for h in registry.registered]
        assert ids == ["tool_a", "tool_b", "tool_c"]

    def test_default_version_when_not_specified(self, tmp_path):
        """An entry with no version field defaults to v1."""
        tool = {k: v for k, v in MINIMAL_HTTP_TOOL.items() if k != "version"}
        registry = _make_registry()
        p = _write_yaml(tmp_path, [tool])
        load_tools_from_yaml(registry, str(p))
        assert registry.registered[0].version == "v1"

    def test_default_features_when_not_specified(self, tmp_path):
        """An entry with no features field defaults to an empty features dict."""
        tool = {k: v for k, v in MINIMAL_HTTP_TOOL.items() if k != "features"}
        registry = _make_registry()
        p = _write_yaml(tmp_path, [tool])
        load_tools_from_yaml(registry, str(p))
        assert registry.registered[0].features == {}

    def test_prints_loaded_count(self, tmp_path, capsys):
        """Loading the catalog prints a summary line with the number of HTTP tools loaded."""
        p = _write_yaml(tmp_path, [MINIMAL_HTTP_TOOL])
        load_tools_from_yaml(_make_registry(), str(p))
        out = capsys.readouterr().out
        assert "1" in out
        assert "HTTP tools" in out


# ═════════════════════════════════════════════════════════════════════════════
# load_tools_from_yaml — skipping rules
# ═════════════════════════════════════════════════════════════════════════════

class TestLoadToolsFromYamlSkipping:
    """load_tools_from_yaml's rules for skipping malformed or non-HTTP catalog entries."""

    def test_skips_tool_without_http_block(self, tmp_path):
        """An entry with no http block is skipped; entries with one are still registered."""
        tools = [
            {"tool_id": "legacy_tool", "inputs": []},   # no 'http' key
            MINIMAL_HTTP_TOOL,
        ]
        registry = _make_registry()
        p = _write_yaml(tmp_path, tools)
        load_tools_from_yaml(registry, str(p))

        assert len(registry.registered) == 1
        assert registry.registered[0].tool_id == "gene_search"

    def test_skips_tool_without_tool_id(self, tmp_path, capsys):
        """An entry missing tool_id is skipped and a warning is printed, without aborting the load."""
        tools = [
            {"http": MINIMAL_HTTP_TOOL["http"], "inputs": []},  # missing tool_id
            MINIMAL_HTTP_TOOL,
        ]
        registry = _make_registry()
        p = _write_yaml(tmp_path, tools)
        load_tools_from_yaml(registry, str(p))

        # Only the valid tool is registered
        assert len(registry.registered) == 1
        # A warning should be printed
        out = capsys.readouterr().out
        assert "WARNING" in out

    def test_empty_yaml_registers_nothing(self, tmp_path):
        """An empty YAML file (parses to None) registers no tools and does not raise."""
        p = tmp_path / "tools.yaml"
        p.write_text("")          # empty file → yaml.safe_load returns None
        registry = _make_registry()
        load_tools_from_yaml(registry, str(p))
        registry.register.assert_not_called()

    def test_yaml_with_only_non_http_tools_registers_nothing(self, tmp_path):
        """A catalog containing only entries without an http block registers nothing."""
        tools = [
            {"tool_id": "a", "inputs": []},
            {"tool_id": "b", "inputs": []},
        ]
        registry = _make_registry()
        p = _write_yaml(tmp_path, tools)
        load_tools_from_yaml(registry, str(p))
        registry.register.assert_not_called()

    def test_mixed_tools_only_http_ones_registered(self, tmp_path):
        """In a mixed catalog, only the HTTP-tool entries are registered, in their original order."""
        tools = [
            {"tool_id": "legacy"},                       # no http
            {**MINIMAL_HTTP_TOOL, "tool_id": "http_1"},
            {"tool_id": "also_legacy"},                  # no http
            {**MINIMAL_HTTP_TOOL, "tool_id": "http_2"},
        ]
        registry = _make_registry()
        p = _write_yaml(tmp_path, tools)
        load_tools_from_yaml(registry, str(p))
        ids = [h.tool_id for h in registry.registered]
        assert ids == ["http_1", "http_2"]
