"""
Tests for toolserver.store.RunStore, the JSON-file-backed persistence layer
for run records.

Covers construction/directory creation, path derivation, create/get/update/
try_get behavior against both the in-memory cache and disk, the atomic
(write-tmp-then-os.replace) write path used by update(), and thread safety
under concurrent creates/updates/reads via the store's shared lock.

Developer:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from toolserver.models import RunRecord
from toolserver.store import RunStore

# ===========================================================================
# Helpers
# ===========================================================================

def _make_record(run_id: str = "run-001", **kwargs) -> RunRecord:
    """Build a minimal valid RunRecord matching the real schema."""
    defaults = dict(
        run_id=run_id,
        tool_id="enrichr_pathway",
        state="QUEUED",
        created_epoch=1_700_000_000,
        updated_epoch=1_700_000_001,
        inputs={"genes": ["TP53", "BRCA1"]},
        resources={},
        logs=[],
        results=None,
        error=None,
    )
    defaults.update(kwargs)
    return RunRecord(**defaults)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def store(tmp_path: Path) -> RunStore:
    return RunStore(root_dir=str(tmp_path))


@pytest.fixture()
def rec() -> RunRecord:
    return _make_record()


# ===========================================================================
# __init__ / construction
# ===========================================================================

class TestInit:
    """RunStore's constructor creates its storage directory as needed and
    starts with a clean in-memory cache and lock."""

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        """Constructing a RunStore over a non-existent nested path creates
        that directory tree."""
        target = tmp_path / "deep" / "nested" / "store"
        assert not target.exists()
        RunStore(root_dir=str(target))
        assert target.is_dir()

    def test_accepts_existing_directory(self, tmp_path: Path) -> None:
        """Constructing a RunStore over an already-existing directory does
        not raise."""
        # Must not raise if the directory already exists
        RunStore(root_dir=str(tmp_path))
        RunStore(root_dir=str(tmp_path))

    def test_cache_starts_empty(self, store: RunStore) -> None:
        """A freshly constructed RunStore's in-memory cache is empty."""
        assert store._cache == {}

    def test_lock_is_created(self, store: RunStore) -> None:
        """A freshly constructed RunStore has a lock ready for
        synchronized cache/file access."""
        assert store._lock is not None


# ===========================================================================
# _path
# ===========================================================================

class TestPath:
    """RunStore._path() deterministically derives each run's on-disk JSON
    file location from its run_id."""

    def test_returns_json_file_inside_root(self, store: RunStore) -> None:
        """_path() returns a "<run_id>.json" file located directly inside
        the store's root directory."""
        p = store._path("abc-123")
        assert p.parent == store.root
        assert p.name == "abc-123.json"

    def test_different_ids_give_different_paths(self, store: RunStore) -> None:
        """_path() maps distinct run_ids to distinct file paths."""
        assert store._path("id-1") != store._path("id-2")


# ===========================================================================
# create
# ===========================================================================

class TestCreate:
    """RunStore.create() persists a new run record to both the in-memory
    cache and a JSON file on disk."""

    def test_file_written(self, store: RunStore, rec: RunRecord) -> None:
        """create() writes a JSON file for the run at its derived path."""
        store.create(rec)
        assert store._path(rec.run_id).exists()

    def test_file_contains_valid_json(self, store: RunStore, rec: RunRecord) -> None:
        """create() writes valid JSON whose run_id field matches the
        record."""
        store.create(rec)
        raw = store._path(rec.run_id).read_text()
        data = json.loads(raw)
        assert data["run_id"] == rec.run_id

    def test_record_added_to_cache(self, store: RunStore, rec: RunRecord) -> None:
        """create() stores the exact same record object in the in-memory
        cache, keyed by run_id."""
        store.create(rec)
        assert rec.run_id in store._cache
        assert store._cache[rec.run_id] is rec

    def test_overwrite_existing_record(self, store: RunStore, rec: RunRecord) -> None:
        """create() called again with the same run_id overwrites both the
        cached record and the on-disk file with the new state."""
        store.create(rec)
        updated = _make_record(run_id=rec.run_id, state="RUNNING")
        store.create(updated)
        assert store._cache[rec.run_id].state == "RUNNING"
        data = json.loads(store._path(rec.run_id).read_text())
        assert data["state"] == "RUNNING"

    def test_multiple_records_written(self, store: RunStore) -> None:
        """create() called for several distinct run_ids writes a separate
        file and cache entry for each."""
        recs = [_make_record(run_id=f"run-{i}") for i in range(5)]
        for r in recs:
            store.create(r)
        for r in recs:
            assert store._path(r.run_id).exists()
            assert r.run_id in store._cache


# ===========================================================================
# get
# ===========================================================================

class TestGet:
    """RunStore.get() serves cached records without touching disk, falls
    back to loading and caching from disk on a cache miss, and raises
    KeyError naming the missing run_id when no record exists anywhere."""

    def test_returns_cached_record(self, store: RunStore, rec: RunRecord) -> None:
        """get() returns the identical cached object, not a reloaded
        copy, when the run_id is already in cache."""
        store.create(rec)
        result = store.get(rec.run_id)
        assert result is rec  # exact same object from cache

    def test_loads_from_disk_when_not_in_cache(self, store: RunStore, rec: RunRecord) -> None:
        """get() falls back to reading the on-disk file when the run_id
        has been evicted from the cache."""
        store.create(rec)
        store._cache.clear()  # evict from cache
        result = store.get(rec.run_id)
        assert result.run_id == rec.run_id

    def test_disk_load_populates_cache(self, store: RunStore, rec: RunRecord) -> None:
        """get() re-populates the cache after loading a record from
        disk."""
        store.create(rec)
        store._cache.clear()
        store.get(rec.run_id)
        assert rec.run_id in store._cache

    def test_raises_key_error_for_missing_id(self, store: RunStore) -> None:
        """get() raises KeyError with a "run not found" message for a
        run_id with no cache entry or file."""
        with pytest.raises(KeyError, match="run not found"):
            store.get("does-not-exist")

    def test_key_error_message_contains_run_id(self, store: RunStore) -> None:
        """get()'s KeyError message includes the specific missing
        run_id, not just a generic message."""
        missing_id = "ghost-run-999"
        with pytest.raises(KeyError, match=missing_id):
            store.get(missing_id)

    def test_round_trip_preserves_fields(self, store: RunStore) -> None:
        """A record's fields (state, inputs, etc.) survive a
        create()-then-disk-load round trip unchanged."""
        rec = _make_record(run_id="rt-1", state="COMPLETED", inputs={"genes": ["MYC"]})
        store.create(rec)
        store._cache.clear()
        loaded = store.get("rt-1")
        assert loaded.run_id == "rt-1"
        assert loaded.state == "COMPLETED"
        assert loaded.inputs == {"genes": ["MYC"]}

    def test_disk_file_with_extra_whitespace_still_loads(
        self, store: RunStore, rec: RunRecord
    ) -> None:
        """get() parses the indented, multi-line JSON produced by
        model_dump_json(indent=2) without error."""
        store.create(rec)
        store._cache.clear()
        # model_dump_json with indent=2 produces multi-line JSON; make sure it parses fine
        raw = store._path(rec.run_id).read_text()
        assert "\n" in raw  # indented
        result = store.get(rec.run_id)
        assert result.run_id == rec.run_id


# ===========================================================================
# update
# ===========================================================================

class TestUpdate:
    """RunStore.update() writes through a temp file and a single
    os.replace() so the on-disk record is never left partially written,
    and the change is immediately visible via the cache."""

    def test_cache_updated(self, store: RunStore, rec: RunRecord) -> None:
        """update() replaces the cached record for a run_id with the new
        state."""
        store.create(rec)
        updated = _make_record(run_id=rec.run_id, state="COMPLETED")
        store.update(updated)
        assert store._cache[rec.run_id].state == "COMPLETED"

    def test_file_updated_on_disk(self, store: RunStore, rec: RunRecord) -> None:
        """update() overwrites the on-disk JSON file with the new record
        state."""
        store.create(rec)
        updated = _make_record(run_id=rec.run_id, state="COMPLETED")
        store.update(updated)
        data = json.loads(store._path(rec.run_id).read_text())
        assert data["state"] == "COMPLETED"

    def test_no_tmp_file_left_behind(self, store: RunStore, rec: RunRecord) -> None:
        """update() leaves no "*.json.tmp" file behind after a
        successful write."""
        store.create(rec)
        store.update(_make_record(run_id=rec.run_id, state="COMPLETED"))
        tmp = store._path(rec.run_id).with_suffix(".json.tmp")
        assert not tmp.exists()

    def test_atomic_replace_used(self, store: RunStore, rec: RunRecord) -> None:
        """update() writes to a .tmp file first and swaps it into place
        via a single os.replace() call, rather than writing the target
        file directly."""
        store.create(rec)
        replaced: list[str] = []

        original_replace = __import__("os").replace

        with patch("toolserver.store.os.replace", side_effect=lambda src, dst: (
            replaced.append((str(src), str(dst))),
            original_replace(src, dst),
        )):
            store.update(_make_record(run_id=rec.run_id, state="COMPLETED"))

        assert len(replaced) == 1
        src, dst = replaced[0]
        assert src.endswith(".json.tmp")
        assert dst.endswith(f"{rec.run_id}.json")

    def test_update_without_prior_create(self, store: RunStore) -> None:
        """update() writes a new record to cache and disk even when
        create() was never called for that run_id first."""
        # update() makes no precondition check — it should still write
        rec = _make_record(run_id="new-run", state="RUNNING")
        store.update(rec)
        assert store._path(rec.run_id).exists()
        assert store._cache["new-run"].state == "RUNNING"

    def test_get_after_update_returns_new_state(self, store: RunStore, rec: RunRecord) -> None:
        """get() immediately reflects a record's updated state after
        update(), via the cache."""
        store.create(rec)
        store.update(_make_record(run_id=rec.run_id, state="COMPLETED"))
        # get() should return the updated record from cache
        result = store.get(rec.run_id)
        assert result.state == "COMPLETED"


# ===========================================================================
# try_get
# ===========================================================================

class TestTryGet:
    """RunStore.try_get() wraps get() to return None instead of raising
    KeyError for a missing run_id."""

    def test_returns_record_when_found(self, store: RunStore, rec: RunRecord) -> None:
        """try_get() returns the record when the run_id exists."""
        store.create(rec)
        result = store.try_get(rec.run_id)
        assert result is not None
        assert result.run_id == rec.run_id

    def test_returns_none_when_missing(self, store: RunStore) -> None:
        """try_get() returns None, not an exception, for a run_id that
        doesn't exist."""
        result = store.try_get("ghost-999")
        assert result is None

    def test_does_not_raise_for_missing(self, store: RunStore) -> None:
        """try_get() never raises for a missing run_id."""
        # Must never raise — the whole point of try_get
        try:
            store.try_get("does-not-exist")
        except Exception as exc:
            pytest.fail(f"try_get raised unexpectedly: {exc}")

    def test_delegates_to_get(self, store: RunStore, rec: RunRecord) -> None:
        """try_get() is implemented in terms of get(), calling it exactly
        once with the given run_id."""
        store.create(rec)
        with patch.object(store, "get", wraps=store.get) as mock_get:
            store.try_get(rec.run_id)
            mock_get.assert_called_once_with(rec.run_id)


# ===========================================================================
# Thread safety
# ===========================================================================

class TestThreadSafety:
    """Concurrent creates, updates, and reads against RunStore never
    raise, lose records, or corrupt on-disk JSON, because all cache/file
    access is serialized through the store's shared lock."""

    def test_concurrent_creates_all_succeed(self, store: RunStore) -> None:
        """20 threads calling create() concurrently for distinct run_ids
        all succeed without error, and every record ends up in the
        cache."""
        errors: List[Exception] = []

        def create_one(i: int) -> None:
            try:
                store.create(_make_record(run_id=f"thread-{i}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=create_one, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(store._cache) == 20

    def test_concurrent_updates_do_not_corrupt_file(self, store: RunStore) -> None:
        """20 threads calling update() concurrently on the same run_id
        never raise, and the file on disk is always well-formed,
        parseable JSON afterward."""
        rec = _make_record(run_id="shared-run")
        store.create(rec)
        errors: List[Exception] = []

        def update_one(i: int) -> None:
            try:
                s = ["QUEUED", "RUNNING", "COMPLETED", "FAILED"][i % 4]
                store.update(_make_record(run_id="shared-run", state=s))
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=update_one, args=(i,))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # File must be valid JSON after all concurrent writes
        raw = store._path("shared-run").read_text()
        data = json.loads(raw)
        assert data["run_id"] == "shared-run"

    def test_concurrent_get_and_create(self, store: RunStore) -> None:
        """Concurrent readers calling get() and writers calling update()
        on the same run_id never raise, regardless of interleaving."""
        rec = _make_record(run_id="race-run")
        store.create(rec)
        errors: List[Exception] = []

        def reader() -> None:
            for _ in range(50):
                try:
                    store.get("race-run")
                except Exception as exc:
                    errors.append(exc)

        def writer() -> None:
            for i in range(10):
                try:
                    store.update(_make_record(run_id="race-run", state="RUNNING"))
                except Exception as exc:
                    errors.append(exc)

        threads = (
            [threading.Thread(target=reader) for _ in range(4)]
            + [threading.Thread(target=writer) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []