"""Guard: tests must exercise the same pure-Python code the Docker image ships.

An in-place compiled extension (for example a stray .so from an old Cython
build) would be imported instead of the .py source and silently change what
the suite tests.
"""

from __future__ import annotations

import importlib
import pkgutil

import toolserver


def test_toolserver_modules_load_from_python_source():
    compiled = []
    for info in pkgutil.walk_packages(toolserver.__path__, prefix="toolserver."):
        module = importlib.import_module(info.name)
        origin = getattr(module, "__file__", "") or ""
        if not origin.endswith(".py"):
            compiled.append(f"{info.name}: {origin}")
    assert not compiled, "modules not loaded from .py source:\n" + "\n".join(compiled)
