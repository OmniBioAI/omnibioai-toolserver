"""
OmniBioAI toolserver.executor.

Purpose:
    Defines Executor with submit methods for toolserver.executor.

Author:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from .models import RunRecord
from .registry import ToolRegistry
from .store import RunStore


class Executor:
    """
    Background execution in a thread pool.
    Good for REST calls + light CPU tasks. For heavy CPU, move to Celery/Slurm later.
    """
    def __init__(self, store: RunStore, registry: ToolRegistry, max_workers: int = 8) -> None:
        self.store = store
        self.registry = registry
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, rec: RunRecord, full_inputs: Dict[str, Any], resources: Dict[str, Any]) -> None:
        handler = self.registry.get(rec.tool_id)

        def append_log(line: str) -> None:
            r = self.store.get(rec.run_id)
            r.logs.append(line)
            r.updated_epoch = int(time.time())
            self.store.update(r)

        def work():
            r = self.store.get(rec.run_id)
            r.state = "RUNNING"
            r.updated_epoch = int(time.time())
            self.store.update(r)

            try:
                append_log(f"Starting tool: {rec.tool_id}")
                res = handler.run(full_inputs, resources, append_log)
                r = self.store.get(rec.run_id)
                r.results = res
                r.state = "COMPLETED"
                r.updated_epoch = int(time.time())
                append_log("Completed")
                self.store.update(r)
            except Exception as e:
                tb = traceback.format_exc(limit=20)
                r = self.store.get(rec.run_id)
                r.state = "FAILED"
                r.error = {"code": "EXEC_FAILED", "message": str(e), "trace": tb}
                r.updated_epoch = int(time.time())
                r.logs.append(f"FAILED: {e}")
                self.store.update(r)

        self.pool.submit(work)
