"""Graph expansion bounds and scoring in retrieval (Phase 4), plus memory graph maintenance.

Regression guards for a measured failure: a single generic query token resolved to a hub entity,
the unbounded 1-hop expansion admitted 110 weakly related facts, and the flat +0.15 relevance
bonus pushed them into ranks 2-8 above genuine text hits.
"""
from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import retrieval as retrieval_mod
from hermes_plugin_holographic.retrieval import FactRetriever
from hermes_plugin_holographic.store import MemoryStore

DIM = 512


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)


def test_expansion_is_capped_by_neighbour_and_fact_limits(tmp_path):
    """A hub entity with many neighbours must not flood the candidate pool."""
    store = _store(tmp_path)
    neighbour_names = [f"graphnbr{i:02d}" for i in range(40)]
    # Bridge fact: links the hub to every neighbour (the trigger turns that into co_occurs edges).
    store.add_fact_status("hubnode bridges " + " ".join(neighbour_names),
                          category="general", entities=["hubnode", *neighbour_names])
    # One fact per neighbour, carrying no query terms.
    for i, name in enumerate(neighbour_names):
        store.add_fact_status(f"filler{i:02d} unrelated payload wording", category="general", entities=[name])

    retriever = FactRetriever(store, hrr_dim=DIM)
    retriever.search("hubnode", limit=10)
    expanded = getattr(retriever, "_graph_expanded_ids", set())

    assert len(expanded) <= retrieval_mod._GRAPH_FACT_CAP
    # 40 neighbours exist, the traversal reads at most the neighbour cap.
    assert len(expanded) <= retrieval_mod._GRAPH_NEIGHBOUR_CAP


def test_graph_discoveries_are_interleaved_not_promoted(tmp_path):
    """Graph discoveries with some textual evidence surface below text hits; zero-evidence ones are dropped."""
    store = _store(tmp_path)
    # Edge hubnode <-> satellite, created by a bridging fact.
    store.add_fact_status("hubnode bridging satellite device", category="general",
                          entities=["hubnode", "satellite"])
    strong, _ = store.add_fact_status("hubnode review gate blocks unverified commits",
                                      category="general")
    related, _ = store.add_fact_status("review gate notes about weather", category="general",
                                       entities=["satellite"])
    junk, _ = store.add_fact_status("completely different topic filler", category="general",
                                    entities=["satellite"])

    retriever = FactRetriever(store, hrr_dim=DIM)
    ranked = [f["fact_id"] for f in retriever.search("hubnode review gate", limit=10)]

    # A discovery carrying real query overlap is present (recall), below the direct hit (precision).
    assert related in ranked
    assert ranked.index(strong) < ranked.index(related)
    # A discovery with no textual evidence is never surfaced by the graph.
    assert junk not in ranked


def test_maintenance_removes_unbacked_edges_and_ignores_retired_facts(tmp_path):
    """`memory_graph_maintain.main()` exists, prunes edges with no linking fact, and scores edges on active facts."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.home() / ".hermes" / "scripts"))
    try:
        import memory_graph_maintain
    except ImportError:  # pragma: no cover - scripts dir not present in a bare checkout
        pytest.skip("memory_graph_maintain.py not available")

    db = str(tmp_path / "graph.db")
    store = MemoryStore(db, hrr_dim=DIM)
    store.add_fact_status("alphagraph links betagraph one", category="general",
                          entities=["alphagraph", "betagraph"])
    gone, _ = store.add_fact_status("gammagraph links deltagraph two", category="general",
                                    entities=["gammagraph", "deltagraph"])

    summary = memory_graph_maintain.main(db)
    assert "before" in summary and "after" in summary

    conn = sqlite3.connect(db)
    try:
        edges = conn.execute(
            "SELECT e1.name, e2.name FROM entity_relations r "
            "JOIN entities e1 ON e1.entity_id=r.src_entity_id "
            "JOIN entities e2 ON e2.entity_id=r.dst_entity_id").fetchall()
    finally:
        conn.close()
    names = {frozenset(pair) for pair in edges}
    assert frozenset({"alphagraph", "betagraph"}) in names

    # Drop the fact that backed the second edge: leave the entity links, no fact links both endpoints.
    conn = sqlite3.connect(db)
    try:
        conn.execute("DELETE FROM fact_entities WHERE fact_id = ?", (gone,))
        conn.commit()
    finally:
        conn.close()

    memory_graph_maintain.main(db)
    conn = sqlite3.connect(db)
    try:
        after = {frozenset((a, b)) for a, b in conn.execute(
            "SELECT e1.name, e2.name FROM entity_relations r "
            "JOIN entities e1 ON e1.entity_id=r.src_entity_id "
            "JOIN entities e2 ON e2.entity_id=r.dst_entity_id").fetchall()}
    finally:
        conn.close()
    assert frozenset({"gammagraph", "deltagraph"}) not in after
    assert frozenset({"alphagraph", "betagraph"}) in after


def test_wiki_pointer_fact_links_namespace_and_page_topic(tmp_path):
    """A wiki pointer fact (raw-SQL ingest, no capitalized names) is linked from its path."""
    store = _store(tmp_path)
    fact, _ = store.add_fact_status(
        "[WIKI HINT - consult when relevant] Wiki page: realm-forge/concepts/combat-systems.md | type: concept",
        category="wiki")

    names = {r["name"].lower() for r in store.read(
        "SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id=fe.entity_id WHERE fe.fact_id=?",
        (fact,))}
    assert "realm forge" in names
    assert "combat systems" in names
    # The co-occurrence trigger turns that pair into a graph edge.
    assert store.read("SELECT COUNT(*) c FROM entity_relations")[0]["c"] == 1


def test_wiki_structural_slugs_are_not_entities(tmp_path):
    """index/schema/log pages are structure, not topics: only the namespace is linked."""
    store = _store(tmp_path)
    fact, _ = store.add_fact_status(
        "[WIKI HINT - consult when relevant] Wiki page: realm-forge/SCHEMA.md | type: schema",
        category="wiki")

    names = {r["name"].lower() for r in store.read(
        "SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id=fe.entity_id WHERE fe.fact_id=?",
        (fact,))}
    assert names == {"realm forge"}


def test_hub_entities_are_not_traversed(tmp_path):
    """Hub/namespace entities are neither expanded from nor traversed into."""
    store = _store(tmp_path)
    names = [f"leaf{i:02d}" for i in range(45)]
    # One hub fact linking "hubspace" to every leaf: 45 edges, over the degree cap.
    store.add_fact_status("hubspace topics " + " ".join(names), entities=["hubspace", *names])
    for i, name in enumerate(names):
        store.add_fact_status(f"leaf fact {i:02d}", category="general", entities=[name])
    # An ordinary pair, well under the cap.
    store.add_fact_status("smallspace links tinynode", entities=["smallspace", "tinynode"])

    cap = retrieval_mod._GRAPH_MAX_ENTITY_DEGREE
    assert store.entity_degrees(store.entity_ids("hubspace"))[store.entity_ids("hubspace")[0]] > cap

    # Expanding from an ordinary leaf must not walk back into the hub.
    leaf = store.entity_ids("leaf00")
    assert store.entity_neighbors(leaf, limit=25, max_degree=cap) == set()

    # Ordinary entities still traverse normally.
    assert store.entity_neighbors(store.entity_ids("smallspace"), limit=25, max_degree=cap)

    # A query that resolves only to a hub does not expand at all.
    retriever = FactRetriever(store, hrr_dim=DIM)
    retriever.search("hubspace", limit=10)
    assert getattr(retriever, "_graph_expanded_ids", set()) == set()
