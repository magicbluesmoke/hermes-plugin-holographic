"""Superseded facts are retired, not deleted: kept on disk, out of recall, linked to their replacement.

The chain (old -> new, relation='supersedes') is what makes a correction auditable: history shows what the store
used to assert and which fact replaced it, search/probe/list never return the retired wording, and re-asserting an
old wording makes it current again instead of silently pointing at a retired row.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider
from hermes_plugin_holographic.store import MemoryStore

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


def _found(provider, query):
    # category pinned: an unpinned query is scoped by intent, which would hide a general-category fact
    return [r["content"] for r in _call(provider, action="search", query=query, category="general").get("results", [])]


def test_old_fact_is_kept_not_deleted(provider):
    old = _call(provider, action="add", content="Realm Forge ships on Fridays.")["fact_id"]
    new = _call(provider, action="add", content="Realm Forge ships on Tuesdays.", supersedes=old)["fact_id"]

    assert new != old
    assert _call(provider, action="stats")["retired_facts"] == 1
    hits = _found(provider, "Realm Forge ships Fridays")
    assert "Realm Forge ships on Fridays." not in hits  # the retired wording is out of recall
    assert "Realm Forge ships on Tuesdays." in hits


def test_history_walks_the_chain_from_either_end(provider):
    old = _call(provider, action="add", content="The kanban board has 4 lanes.")["fact_id"]
    new = _call(provider, action="add", content="The kanban board has 5 lanes.", supersedes=old)["fact_id"]

    for start in (old, new):
        chain = _call(provider, action="history", fact_id=start)["chain"]
        assert [c["fact_id"] for c in chain] == [old, new]
        assert [c["status"] for c in chain] == ["retired", "active"]
        assert chain[0]["retired_by"] == new
        assert chain[0]["retired_at"] is not None
        assert chain[1]["retired_at"] is None


def test_list_and_export_hide_retired_rows(provider):
    old = _call(provider, action="add", content="Pygame is pinned to 2.5.0.")["fact_id"]
    _call(provider, action="add", content="Pygame is pinned to 2.6.1.", supersedes=old)

    listed = [f["content"] for f in _call(provider, action="list")["facts"]]
    assert listed == ["Pygame is pinned to 2.6.1."]
    assert "Pygame is pinned to 2.5.0." not in _call(provider, action="export")["markdown"]


def test_probe_hides_retired_rows(provider):
    old = _call(provider, action="add", content="Michael runs the 7700S GPU at 100W.",
                entities=["Michael"])["fact_id"]
    _call(provider, action="add", content="Michael runs the 7700S GPU at 120W.", supersedes=old,
          entities=["Michael"])

    contents = [f["content"] for f in _call(provider, action="probe", entity="Michael")["results"]]
    assert contents == ["Michael runs the 7700S GPU at 120W."]


def test_superseding_an_already_retired_fact_targets_the_current_one(provider):
    first = _call(provider, action="add", content="Cron runs at 3am.")["fact_id"]
    second = _call(provider, action="add", content="Cron runs at 4am.", supersedes=first)["fact_id"]
    third = _call(provider, action="add", content="Cron runs at 5am.", supersedes=first)["fact_id"]

    chain = _call(provider, action="history", fact_id=first)["chain"]
    assert [c["fact_id"] for c in chain] == [first, second, third]
    assert [c["status"] for c in chain] == ["retired", "retired", "active"]


def test_reasserting_a_retired_wording_makes_it_current_again(provider):
    old = _call(provider, action="add", content="The wiki lives on :8082.")["fact_id"]
    _call(provider, action="add", content="The wiki lives on :8090.", supersedes=old)
    again = _call(provider, action="add", content="The wiki lives on :8082.")

    assert again["fact_id"] == old  # the same row, reactivated rather than a duplicate
    assert _call(provider, action="stats")["retired_facts"] == 0
    assert again["status"] == "exists"  # no new row: the old one is current again
    hits = _found(provider, "wiki lives 8082")
    assert "The wiki lives on :8082." in hits
    # both wordings are asserted now; the caller may supersede the one that no longer holds
    assert "The wiki lives on :8090." in hits


def test_explicit_remove_still_deletes(provider):
    fact = _call(provider, action="add", content="A throwaway fact about nothing.")["fact_id"]
    assert _call(provider, action="remove", fact_id=fact)["removed"] is True
    assert _call(provider, action="stats")["facts"] == 0


def test_writing_a_retired_wording_is_not_flagged_as_a_conflict(provider):
    old = _call(provider, action="add", content="Deploys happen on Friday nights.")["fact_id"]
    _call(provider, action="add", content="Deploys happen on Thursday nights.", supersedes=old)

    result = _call(provider, action="add", content="Deploys happen on Thursday nights at 8pm.")
    assert all(c["fact_id"] != old for c in result.get("possible_conflicts", []))
    assert all(c["fact_id"] != old for c in result.get("similar", []))


def test_supersede_keeps_entities_linked_to_the_retired_side(store):
    old = store.add_fact_status("Maria prefers morning appointments.", category="general",
                                entities=["Maria"])[0]
    new = store.add_fact_status("Maria prefers afternoon appointments.", category="general",
                                entities=["Maria"])[0]
    store.retire_fact(old, superseded_by=new)

    assert store.fact_history(old)[0]["status"] == "retired"
    linked = store.read("SELECT f.fact_id FROM facts f JOIN fact_entities fe ON fe.fact_id = f.fact_id "
                        "JOIN entities e ON e.entity_id = fe.entity_id WHERE e.name = 'Maria'")
    assert {r[0] for r in linked} == {old, new}  # history stays attached to the person


def test_a_correction_stays_in_the_old_fact_category(provider):
    old = _call(provider, action="add", content="Heroic games need UseTakeFocus off.", category="tool")["fact_id"]
    new = _call(provider, action="add", content="Heroic games need UseTakeFocus off (Wine reg).", supersedes=old)

    kinds = {f["fact_id"]: f["category"] for f in _call(provider, action="list")["facts"]}
    assert kinds[new["fact_id"]] == "tool"
    assert kinds == {new["fact_id"]: "tool"}  # the retired row is not in list()


def test_migration_adds_columns_to_an_existing_store(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE facts (fact_id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL UNIQUE,"
        " category TEXT DEFAULT 'general', tags TEXT DEFAULT '', trust_score REAL DEFAULT 0.5,"
        " retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0,"
        " created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
        " hrr_vector BLOB);"
        "INSERT INTO facts (content) VALUES ('A fact from the old schema.');"
    )
    conn.commit()
    conn.close()

    s = MemoryStore(str(db), hrr_dim=DIM)  # opens: must add retired_at/retired_by before the schema index
    try:
        assert s.count_facts() == 1
        assert s.count_retired() == 0
        assert s.retire_fact(1) is True
        assert s.count_facts() == 0 and s.count_retired() == 1
    finally:
        s.close()
