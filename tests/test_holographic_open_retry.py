"""A failed store open must not be permanent for the process lifetime.

Observed 2026-11-11: memory_store.db was corrupted, the provider's first open raised, and
``_init_error`` kept every later ``fact_store`` call dead — even after the file was restored,
until the whole agent restarted. ``handle_tool_call`` now retries the open once.
"""
import sqlite3

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider


def _provider(home, **config):
    provider = HolographicMemoryProvider(config={"scope": "profile", **config})
    provider.initialize("retry-test", hermes_home=str(home), platform="cli")
    return provider


def _corrupt(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)  # header, no valid schema


def test_open_failure_is_retried_after_the_store_is_repaired(tmp_path):
    db = tmp_path / "memory_store.db"
    _corrupt(db)

    provider = _provider(tmp_path)
    assert provider._store is None, "a corrupt store must not be opened"
    first = provider.handle_tool_call("fact_store", {"action": "stats"})
    assert "unavailable" in str(first).lower()

    db.unlink()
    writer = sqlite3.connect(str(db))
    writer.close()

    second = provider.handle_tool_call("fact_store", {"action": "stats"})
    assert "unavailable" not in str(second).lower(), f"still sticky: {second}"
    assert provider._init_error == ""
    provider.shutdown()


def test_fresh_store_still_initialises_normally(tmp_path):
    provider = _provider(tmp_path)
    assert provider._store is not None
    assert provider._init_error == ""
    provider.shutdown()
