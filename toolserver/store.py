"""
OmniBioAI toolserver.store.

Purpose:
    Defines RunStore with create, get, update and try_get methods for toolserver.store.

Author:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, Optional

from .models import RunRecord


class RunStore:
    """
    File-backed store (MVP) so runs survive restarts.
    Writes one JSON file per run_id.

    Safe enough for dev/single-host. Move to Redis/Postgres later.
    """
    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: Dict[str, RunRecord] = {}

    def _path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def create(self, rec: RunRecord) -> None:
        with self._lock:
            self._cache[rec.run_id] = rec
            self._path(rec.run_id).write_text(rec.model_dump_json(indent=2))

    def get(self, run_id: str) -> RunRecord:
        with self._lock:
            if run_id in self._cache:
                return self._cache[run_id]

            p = self._path(run_id)
            if not p.exists():
                raise KeyError(f"run not found: {run_id}")
            rec = RunRecord(**json.loads(p.read_text()))
            self._cache[run_id] = rec
            return rec

    def update(self, rec: RunRecord) -> None:
        with self._lock:
            self._cache[rec.run_id] = rec
            tmp = self._path(rec.run_id).with_suffix(".json.tmp")
            tmp.write_text(rec.model_dump_json(indent=2))
            os.replace(tmp, self._path(rec.run_id))

    def try_get(self, run_id: str) -> Optional[RunRecord]:
        try:
            return self.get(run_id)
        except KeyError:
            return None
