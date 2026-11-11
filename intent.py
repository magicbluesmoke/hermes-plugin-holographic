"""Rule-based intent classification for holographic memory retrieval.

Maps a query string to an intent label and a set of retrieval parameters
(category filter, trust floor, result limit). Used to scope retrieval so
that, e.g., "what did we decide about X" narrows to decision-bearing facts
instead of returning everything mentioning X.

Intent labels:
  decision  - queries about decisions, choices, conclusions
  causal    - queries about why/how something happened
  list      - queries asking for all/everything about a topic
  specific  - queries for a single fact (who/what/when/where)
  broad     - fallback: no scoping, current behavior
"""

from __future__ import annotations

import re
from typing import Any

# Each entry: (intent, compiled pattern list). First match wins.
_INTENT_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    ("decision", [
        re.compile(r"\b(decid|chose|chosen|conclusion|agreed|resolved|settled)\b", re.I),
        re.compile(r"what (did we|was) (decide|decided|choose|chosen)", re.I),
        re.compile(r"(decision|choice) (about|on|for|regarding)", re.I),
    ]),
    ("causal", [
        re.compile(r"\b(why|how (did|does|come|work)|because|reason|cause)\b", re.I),
        re.compile(r"what (caused|led to|made)", re.I),
    ]),
    ("list", [
        re.compile(r"\b(all|every|list|enumerate|everything|each)\b", re.I),
        re.compile(r"what (are|were) (all|the) ", re.I),
    ]),
    ("specific", [
        re.compile(r"^(who|what|when|where|which)\b", re.I),
        re.compile(r"\b(is|was|are|were) (the|a|an)\b", re.I),
    ]),
]

# Default parameters per intent.  None means "no filter" (current behavior).
_INTENT_PARAMS: dict[str, dict[str, Any]] = {
    "decision": {"category": None, "min_trust": 0.5, "limit_mult": 0.5},
    "causal":   {"category": None, "min_trust": 0.4, "limit_mult": 0.6},
    "list":     {"category": None, "min_trust": 0.3, "limit_mult": 1.5},
    "specific": {"category": None, "min_trust": 0.3, "limit_mult": 0.4},
    "broad":    {"category": None, "min_trust": None, "limit_mult": 1.0},
}

# Category hints: if the query mentions a category keyword, restrict to it.
_CATEGORY_HINTS: list[tuple[str, re.Pattern]] = [
    ("tool", re.compile(r"\b(tool|config|setting|command|script|plugin)\b", re.I)),
    ("project", re.compile(r"\b(project|realm.?forge|kanban|board|task|card)\b", re.I)),
    ("user_pref", re.compile(r"\b(prefer|preference|style|convention|rule)\b", re.I)),
]


def classify_intent(query: str) -> str:
    """Return an intent label for *query*."""
    if not query or not query.strip():
        return "broad"
    for intent, patterns in _INTENT_PATTERNS:
        if any(p.search(query) for p in patterns):
            return intent
    return "broad"


def intent_params(intent: str, query: str = "") -> dict[str, Any]:
    """Return retrieval parameters for *intent*, optionally refined by *query*.

    Returns a dict with keys: category (str|None), min_trust (float|None),
    limit_mult (float).  None means "don't override caller's value".
    """
    params = dict(_INTENT_PARAMS.get(intent, _INTENT_PARAMS["broad"]))

    # Refine category from query hints (only if no explicit category was set).
    if params.get("category") is None and query:
        for cat, pat in _CATEGORY_HINTS:
            if pat.search(query):
                params["category"] = cat
                break

    return params


def scope_search(query: str, *, category=None, min_trust=None, limit=10):
    """Compute scoped (category, min_trust, limit) from query intent.

    Returns a tuple.  If the caller explicitly provided category or min_trust
    (not None), those are preserved; only None values are filled from intent.
    """
    intent = classify_intent(query)
    params = intent_params(intent, query)

    out_category = category if category is not None else params["category"]
    out_min_trust = min_trust if min_trust is not None else params["min_trust"]
    if out_min_trust is None:
        out_min_trust = 0.3  # safe default
    out_limit = max(1, int(limit * params["limit_mult"]))
    return out_category, out_min_trust, out_limit, intent