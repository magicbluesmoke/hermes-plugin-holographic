"""Sentence starts are not entity names ("How Hermes", "Every Hermes", "Requires Hermes", "Use when").

Regression guard for a measured defect (2026-11-25): 14 of 762 entities were sentence fragments holding 16 facts,
which polluted probe/related and inflated the degree-1 count.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic.store import MemoryStore


@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "facts.db"))


def _names(store, text):
    return store._extract_entities(text)


@pytest.mark.parametrize("text", [
    "How Hermes routes MCP tools to the gateway",
    "Every Hermes aux consumer (approval, compression) points at the router",
    "Requires Hermes 0.26 or newer to expose the flag",
    "Turn Hermes into a memory server",
    "Use when the tool fails and the fallback chain is exhausted",
    "What Hermes Sends Down The Wire",
])
def test_a_capitalized_sentence_start_is_not_a_name(store, text):
    found = _names(store, text)
    assert not any(name.split()[0].lower() in {"how", "every", "requires", "turn", "use", "what"}
                   for name in found), found


def test_a_real_name_after_a_plain_start_is_still_found(store):
    found = _names(store, "Marina Pet Hospital is on Reservation Road")
    assert "Marina Pet Hospital" in found
    assert "Reservation Road" in found


def test_a_title_is_not_cut_down_to_one_word(store):
    # "The" is already a sentence starter; the run keeps its remaining capitalized words
    assert "Third Turning" in _names(store, "The Third Turning is a TTRPG setting by Chris Gonnerman")
