"""Event dates: a fact's own date (what it is about), separate from the date it was written.

Stored at write time from an explicit date in the text (or a caller-supplied event_date), backfilled once for facts
that predate the column, and usable as a hard bound on search (since/before). Relative words are never resolved.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider
from hermes_plugin_holographic.store import MemoryStore, extract_event_date

DIM = 512


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": DIM})
    p.initialize(session_id="s")
    yield p
    p.shutdown()


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


def _search(provider, query, **extra):
    return _call(provider, action="search", query=query, category="general", **extra)["results"]


# -- extraction ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Kanban worktree landing is automated since 2026-10-18.", "2026-10-18"),
    ("The palette swap landed on October 12, 2026.", "2026-10-12"),
    ("Retired Oct. 3, 2025 after the rewrite.", "2025-10-03"),
    ("First 2026-09-01 then 2026-11-02.", "2026-09-01"),  # leftmost explicit date wins
    ("No date in this sentence at all.", None),
    ("Impossible 2026-02-30 is not a date.", None),
    ("Done yesterday.", None),  # relative is never guessed
    ("", None),
])
def test_extract_event_date(text, expected):
    assert extract_event_date(text) == expected


# -- write path ------------------------------------------------------------------------------------------------------

def test_add_infers_the_date_and_returns_it(provider):
    _call(provider, action="add", content="The palette swap landed on October 12, 2026.")
    assert _search(provider, "palette swap")[0]["event_date"] == "2026-10-12"


def test_add_without_a_date_in_the_text_has_no_event_date(provider):
    _call(provider, action="add", content="The palette swap landed last week.")
    assert "event_date" not in _search(provider, "palette swap")[0]


def test_explicit_event_date_overrides_the_text(provider):
    _call(provider, action="add", content="Retro notes mention 2026-09-01 as the start.",
          event_date="2026-10-01")
    assert _search(provider, "retro notes")[0]["event_date"] == "2026-10-01"


def test_an_invalid_explicit_event_date_falls_back_to_the_text(provider):
    _call(provider, action="add", content="Retro notes mention 2026-09-01 as the start.",
          event_date="not-a-date")
    assert _search(provider, "retro notes")[0]["event_date"] == "2026-09-01"


def test_rewording_recomputes_the_date(store):
    fact = store.add_fact_status("Ship date is 2026-09-01.", category="general")[0]
    assert store.read("SELECT event_date FROM facts WHERE fact_id = ?", (fact,))[0][0] == "2026-09-01"
    store.update_fact(fact, content="Ship date slipped to 2026-10-05.")
    assert store.read("SELECT event_date FROM facts WHERE fact_id = ?", (fact,))[0][0] == "2026-10-05"


# -- search bounds ---------------------------------------------------------------------------------------------------

def test_since_and_before_bound_the_search(provider):
    _call(provider, action="add", content="Kanban lane count was 3 on 2026-09-10.")
    _call(provider, action="add", content="Kanban lane count is 5 on 2026-11-02.")

    dated = lambda rows: sorted(r["event_date"] for r in rows)
    assert dated(_search(provider, "kanban lane count")) == ["2026-09-10", "2026-11-02"]
    assert dated(_search(provider, "kanban lane count", since="2026-10-01")) == ["2026-11-02"]
    assert dated(_search(provider, "kanban lane count", before="2026-10-01")) == ["2026-09-10"]
    assert _search(provider, "kanban lane count", since="2026-12-01") == []


def test_undated_facts_fall_back_to_the_write_date(provider):
    _call(provider, action="add", content="Kanban lane count has no date in it.")
    assert _search(provider, "kanban lane count", since="2000-01-01")  # written today: inside the bound
    assert _search(provider, "kanban lane count", before="2000-01-01") == []


def test_retired_facts_stay_out_of_date_bounded_searches(provider):
    old = _call(provider, action="add", content="Kanban has 4 lanes as of 2026-09-01.")["fact_id"]
    _call(provider, action="add", content="Kanban has 5 lanes as of 2026-11-01.", supersedes=old)
    assert [r["event_date"] for r in _search(provider, "kanban lanes", since="2000-01-01")] == ["2026-11-01"]


# -- migration -------------------------------------------------------------------------------------------------------

def test_backfill_dates_old_rows_once(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE facts (fact_id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL UNIQUE,"
        " category TEXT DEFAULT 'general', tags TEXT DEFAULT '', trust_score REAL DEFAULT 0.5,"
        " retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0,"
        " created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
        " hrr_vector BLOB);"
        "INSERT INTO facts (content) VALUES ('The migration ran on 2026-09-14.');"
        "INSERT INTO facts (content) VALUES ('No date here.');"
    )
    conn.commit()
    conn.close()

    store = MemoryStore(str(db), hrr_dim=DIM)
    try:
        rows = {r["content"]: r["event_date"] for r in store.read("SELECT content, event_date FROM facts")}
        assert rows["The migration ran on 2026-09-14."] == "2026-09-14"
        assert rows["No date here."] is None
        assert store._meta("event_dates_backfilled") is not None
    finally:
        store.close()

    again = MemoryStore(str(db), hrr_dim=DIM)  # second open: guarded, no repeat work
    try:
        assert again.read("SELECT event_date FROM facts WHERE content = 'The migration ran on 2026-09-14.'")[0][0] == "2026-09-14"
    finally:
        again.close()
