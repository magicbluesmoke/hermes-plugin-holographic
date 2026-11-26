"""Supersession ordering: a fact that names what it replaced outranks the fact it replaced.

Regression guard for a measured failure (2026-11-25): the query "is memory graph expansion enabled right now"
returned the stale "enabled" fact at rank 1 and the fact that switched it off at rank 6, because text ranking
cannot see that one claim corrects the other.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic.retrieval import _demote_superseded


def _fact(fid, content, created="2026-11-01 00:00:00"):
    return {"fact_id": fid, "content": content, "created_at": created,
            "trust_score": 0.5, "score": 1.0}


def test_replacement_takes_the_stale_fact_slot():
    stale = _fact(100, "graph expansion is enabled in both profiles", created="2026-11-20 00:00:00")
    current = _fact(200, "graph expansion is now false in both profiles (replaces #100)",
                    created="2026-11-25 00:00:00")
    other = _fact(150, "unrelated fact about kanban", created="2026-11-21 00:00:00")
    out = _demote_superseded([stale, other, current])
    assert [f["fact_id"] for f in out] == [current["fact_id"], other["fact_id"], stale["fact_id"]]


def test_an_issue_number_without_a_replacement_cue_is_ignored():
    stale = _fact(136, "an issue number that is also a fact id", created="2026-11-20 00:00:00")
    other = _fact(300, "upstream issue #136 proposes exposing fact_links", created="2026-11-25 00:00:00")
    out = _demote_superseded([stale, other])
    assert [f["fact_id"] for f in out] == [136, 300]


def test_an_older_fact_cannot_demote_a_newer_one():
    newer = _fact(400, "the current claim", created="2026-11-25 00:00:00")
    older = _fact(410, "an older note that says it replaces #400", created="2026-11-01 00:00:00")
    out = _demote_superseded([newer, older])
    assert [f["fact_id"] for f in out] == [400, 410]


def test_order_is_untouched_without_references():
    facts = [_fact(i, f"plain fact {i}") for i in (1, 2, 3)]
    assert _demote_superseded(facts) is facts
