"""Fact retrieval for the memory store: FTS5 keyword candidates reranked with Jaccard and HRR similarity, weighted by
trust (ported from KIK memory_agent.py); entity recall straight from the entity index."""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

from . import holographic as hrr
from .text import (
    CJK_NEGATORS,
    CLAIM_WORDS,
    CONTRASTIVE,
    NEGATORS,
    UNSEGMENTED,
    analyze,
    content_chars,
    content_keys,
    content_stems,
    gram_coverage,
    is_greeting,
    is_sea,
    query_grams,
    query_terms,
    run_grams,
    soft_jaccard,
    soft_overlap,
    words,
)

logger = logging.getLogger(__name__)

_FACT_COLUMNS = "fact_id, content, category, tags, trust_score, retrieval_count, helpful_count, created_at, updated_at"
_F_COLUMNS = ", ".join(f"f.{c.strip()}" for c in _FACT_COLUMNS.split(","))
_ROLE_ENTITY, _ROLE_CONTENT = hrr.ROLE_ENTITY, hrr.ROLE_CONTENT

# Graph expansion bounds (Phase 4). Measured 2026-11-11: one generic query token ("review") resolved to a hub
# entity and pulled 110 weakly related facts; with the old flat +0.15 they took ranks 2-8. Caps keep the pool
# bounded, and graph discoveries are INTERLEAVED (every Nth slot) instead of score-boosted: they add recall
# without displacing genuine text hits.
_GRAPH_NEIGHBOUR_CAP = 25    # entity neighbours traversed per query
_GRAPH_FACT_CAP = 40         # extra facts admitted from those neighbours
_GRAPH_SLOT_EVERY = 3        # one graph discovery per N result slots, starting at slot 4
_GRAPH_FIRST_SLOT = 4        # discoveries never take rank 1-3: the top text hits stay untouched
_GRAPH_MIN_REL_RATIO = 0.5   # a discovery must carry >= half the best text score to be shown at all
_GRAPH_MAX_ENTITY_DEGREE = 40  # hub entities (namespace/project nodes) are not traversed at all
_PUNCT = ".,;:!?\"'()[]{}#@<>"
# FTS5 operator characters. Inside a token they become spaces of a quoted phrase: "pve-01" -> "pve 01", the same two
# tokens unicode61 indexed, adjacent (deleting them gave "pve01", which matches nothing).
_FTS_OPERATORS = str.maketrans({c: " " for c in '"()*^:-+'})
_RE_UNSEGMENTED = UNSEGMENTED
# The commonest Indonesian function words: a store with none of them holds no Indonesian, and Latin words of a query
# of undetermined language need no word scan (Indonesian prefixes are what hides a word from FTS5).
_INDONESIAN_PROBE = '"yang" OR "dengan" OR "untuk" OR "tidak" OR "adalah" OR "sudah" OR "belum"'
# ... and of German, whose compounds hide their last word from a prefix query ("Katze" in "Familienkatze").
_GERMAN_PROBE = '"und" OR "nicht" OR "ist" OR "sind" OR "der" OR "das" OR "eine" OR "mit"'


def _shift(sim: float) -> float:
    """Cosine similarity [-1, 1] -> [0, 1]."""
    return (sim + 1.0) / 2.0


def _related_floor(dim: int) -> float:
    """Residual similarity a fact must reach to count as structurally tied to an entity in `related`. Measured at
    dim=1024 on 400 facts with one to four entities: tied facts >= 0.35, the best of 17k untied pairs 0.088; 0.2 sits
    in the gap with a wide margin. Small dimensions get a proportionally higher floor (8 sd of chance similarity)."""
    return min(0.5, max(0.2, 8.0 * math.sqrt(0.5 / max(dim, 1))))


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    def __init__(self, store: MemoryStore, temporal_decay_half_life: int = 0,  # days, 0 = disabled
                 fts_weight: float = 0.4, jaccard_weight: float = 0.3, hrr_weight: float = 0.3, hrr_dim: int = 1024):
        self.store, self.half_life, self.hrr_dim = store, temporal_decay_half_life, hrr_dim
        if hrr_weight > 0 and not hrr._HAS_NUMPY:  # redistribute weights without numpy
            fts_weight, jaccard_weight, hrr_weight = 0.6, 0.4, 0.0
        self.fts_weight, self.jaccard_weight, self.hrr_weight = fts_weight, jaccard_weight, hrr_weight
        self._analyses: dict[int, tuple[str, object]] = {}  # fact_id -> (text, Analysis), for word-level scans

    def _atom(self, word: str):
        return hrr.encode_atom(word, self.hrr_dim)

    def _phases(self, blob: bytes):
        """Decode a stored vector; None for a blob of another dimension or a damaged one (logged, row skipped)."""
        try:
            return hrr.bytes_to_phases(blob, dim=self.hrr_dim)
        except (ValueError, TypeError) as exc:
            logger.warning("holographic: skipping an unreadable HRR vector (%s)", exc)
            return None

    def search(self, query: str, category: str | None = None, min_trust: float = 0.3, limit: int = 10,
               since: str | None = None, before: str | None = None) -> list[dict]:
        """FTS5 candidates (limit*3) → Jaccard + HRR rerank → trust weighting → optional temporal decay
        0.5^(age_days / half_life). Returns fact dicts with 'score', sorted desc."""
        terms = query_terms(query or "")
        candidates = {f["fact_id"]: f for f in self._fts_candidates(query, category, min_trust, limit * 3)
                      if self._contains_a_term(terms, f)}
        for fact in self._ngram_candidates(query, category, min_trust, limit * 3):
            if fact["fact_id"] not in candidates or candidates[fact["fact_id"]].get("fts_rank", 0) < fact["fts_rank"]:
                candidates[fact["fact_id"]] = fact
        for fact in self._scan_candidates(terms, category, min_trust, limit * 3):
            candidates.setdefault(fact["fact_id"], fact)  # fills in what FTS cannot see; FTS ranks what it found
        # Graph expansion (Phase 4): traverse entity_relations from query-resolved entities to surface
        # facts linked to structurally connected entities, not just direct co-occurrence.
        # Bounded: a generic token can resolve to a hub entity with a large degree, and an unbounded expansion
        # would flood the candidate pool with weakly related facts (observed: 110 additions from one token).
        query_entity_ids: set[int] = set()
        for term in (query or "").split():
            query_entity_ids.update(self.store.entity_ids(term))
        graph_expanded_ids: set[int] = set()
        if query_entity_ids:
            # A namespace/project entity (the wiki root, "hermes", "ai agents") touches a whole corpus: expanding
            # from it yields arbitrary neighbours, not related ones. Expand only when at least one resolved entity
            # is specific enough to say something about its neighbourhood.
            degrees = self.store.entity_degrees(list(query_entity_ids))
            specific = [eid for eid in query_entity_ids
                        if degrees.get(eid, 0) <= _GRAPH_MAX_ENTITY_DEGREE]
            connected = self.store.entity_neighbors(specific, min_trust=min_trust or 0.0,
                                                   limit=_GRAPH_NEIGHBOUR_CAP,
                                                   max_degree=_GRAPH_MAX_ENTITY_DEGREE) if specific else set()
            if connected:
                rows = self.store.read(
                    "SELECT DISTINCT fact_id FROM fact_entities WHERE entity_id IN ("
                    + ",".join("?" * len(connected)) + ")",
                    list(connected),
                )
                for r in rows[:_GRAPH_FACT_CAP]:
                    nid = int(r[0])
                    if nid not in candidates:
                        rows2 = self._rows([nid], category, min_trust)
                        if rows2:
                            candidates[nid] = rows2[0]
                            graph_expanded_ids.add(nid)
        if since or before:  # a fact's own date (else the day it was written): "what did we know as of ..." bounds
            allowed = self.store.filter_by_event_date(candidates, since, before)
            candidates = {fid: fact for fid, fact in candidates.items() if fid in allowed}
        # Store for rerank boost
        self._graph_expanded_ids = graph_expanded_ids
        return self._rerank(query, list(candidates.values()), limit)

    @staticmethod
    def _contains_a_term(terms, fact: dict) -> bool:
        """A stem prefix ("walk"*) also matches other words ("walker"): keep a candidate only when it really contains
        a query term (same word, same stem, or the same identifier with word boundaries)."""
        if not terms:
            return True
        analysis = analyze(f'{fact.get("content", "")} {fact.get("tags") or ""}')
        return any(t.found_in(analysis) for t in terms)

    def _texts(self, category: str | None = None, min_trust: float | None = None) -> list[tuple[int, float, object]]:
        """(fact_id, trust, Analysis) of every fact passing the filters. The rows are read on every call (the table is
        the truth; another process may have written); a fact is analysed again only when its text changed, so the
        first scan after start analyses every fact and later ones only what changed."""
        where = ["retired_at IS NULL"] + [c for c, on in (("trust_score >= ?", min_trust is not None), ("category = ?", category)) if on]
        params = [p for p, on in ((min_trust, min_trust is not None), (category, category)) if on]
        rows = self.store.read("SELECT fact_id, content, tags, trust_score FROM facts WHERE " + " AND ".join(where)
                               + " ORDER BY fact_id", params)
        cache, out = self._analyses, []
        for row in rows:
            text = f'{row["content"]} {row["tags"] or ""}'
            hit = cache.get(row["fact_id"])
            if hit is None or hit[0] != text:
                hit = cache[row["fact_id"]] = (text, analyze(text))
            out.append((row["fact_id"], row["trust_score"], hit[1]))
        if min_trust is None and category is None and len(cache) > len(out) + 1024:  # deleted facts: only an unfiltered read sees every fact
            self._analyses = {fact_id: cache[fact_id] for fact_id, _, _ in out}
        return out

    def _scan_terms(self, terms) -> list:
        """The terms matched word by word: Arabic and Indic always; Latin ones (Indonesian, German, or a query of
        undetermined language) only when the store holds Indonesian or German facts, so an English store never pays
        for a scan."""
        scan = [t for t in terms if t.scan]
        if any(t.fts_ok for t in scan):
            try:
                needed = bool(self.store.read("SELECT 1 FROM facts_fts WHERE facts_fts MATCH ? LIMIT 1",
                                              (f"{_INDONESIAN_PROBE} OR {_GERMAN_PROBE}",)))
            except sqlite3.DatabaseError:  # a damaged index: the scan does not need it
                needed = True
            if not needed:
                scan = [t for t in scan if not t.fts_ok]
        return scan

    def _rows(self, ids: list[int], category: str | None = None, min_trust: float | None = None) -> list[dict]:
        """Full rows of *ids*, the filters applied again: the fact may have changed since its text was read."""
        if not ids:
            return []
        sql = (f"SELECT * FROM facts WHERE fact_id IN ({','.join('?' * len(ids))}) AND retired_at IS NULL"
               + (" AND trust_score >= ?" if min_trust is not None else "") + (" AND category = ?" if category else ""))
        params = [*ids, *([min_trust] if min_trust is not None else []), *([category] if category else [])]
        return [dict(r) for r in self.store.read(sql, params)]

    def _scan_candidates(self, terms, category: str | None, min_trust: float, limit: int) -> list[dict]:
        """Facts containing the query's scan terms (Arabic, Indic, Indonesian: words whose other forms the FTS5
        index cannot find), matched word by word against the text of every fact and ranked by the IDF-weighted share
        of those terms they contain ('fts_rank'). A scan, like the n-gram path: milliseconds at the size this store
        is meant for."""
        scan = self._scan_terms(terms)
        if not scan:
            return []
        texts = self._texts(category, min_trust)
        found = [(fact_id, trust, [t for t in scan if t.found_in(a)]) for fact_id, trust, a in texts]
        df = {t: sum(t in hit for _, _, hit in found) for t in scan}
        weight = {t: math.log(1.0 + len(texts) / df[t]) for t in scan if df[t]}
        total = sum(weight.values()) or 1.0
        # ties: the more trusted fact, then the newer one
        ranked = sorted(((sum(weight[t] for t in hit) / total, trust, fact_id) for fact_id, trust, hit in found if hit),
                        reverse=True)[:limit]
        rank = {fact_id: score for score, _, fact_id in ranked}
        rows = [f for f in self._rows([fact_id for _, _, fact_id in ranked], category, min_trust)
                if self._contains_a_term(scan, f)]
        for fact in rows:
            fact["fts_rank"] = rank[fact["fact_id"]]
        return rows

    def _ngram_candidates(self, query: str, category: str | None, min_trust: float, limit: int) -> list[dict]:
        """Facts sharing character n-grams with the query's unsegmented runs, ranked by the share of n-grams they
        contain ('fts_rank'). A substring scan: no extra index, so nothing to migrate or keep in sync; at the size
        this store is meant for (hundreds to thousands of facts) it takes milliseconds."""
        grams = query_grams(query)
        if not grams:
            return []
        # content and tags: the agent's keywords (kept in tags) are searched like the text
        hits = " + ".join(f"(instr(lower(content || ' ' || COALESCE(tags, '')), ?) > 0) * {weight}" for _, weight in grams)
        sql = (f"SELECT * FROM (SELECT f.*, ({hits}) AS gram_hits FROM facts f WHERE f.retired_at IS NULL AND f.trust_score >= ?"
               + (" AND f.category = ?" if category else "") + ") WHERE gram_hits > 0 ORDER BY gram_hits DESC LIMIT ?")
        rows = self.store.read(sql, [*(g for g, _ in grams), min_trust, *([category] if category else []), limit])
        total = sum(weight for _, weight in grams)
        results = []
        for row in rows:
            fact = dict(row)
            fact["fts_rank"] = fact.pop("gram_hits") / total
            results.append(fact)
        return results

    def _rerank(self, query: str, candidates: list[dict], limit: int) -> list[dict]:
        query_tokens = content_stems(query)
        # Query vector is loop-invariant; encode lazily on the first candidate that carries an HRR vector
        # so stores whose hrr_vector was never backfilled don't pay for it.
        query_vec = None  # bind(encode_text(query), ROLE_CONTENT): the shape a fact's content is stored in
        # Graph recall: a graph-expanded fact is a DISCOVERY, not a better text match. Scored (and boosted) it either
        # displaced real hits (flat +0.15) or never surfaced (pure relevance). So rank text matches on relevance and
        # interleave graph discoveries into every _GRAPH_SLOT_EVERY-th slot: the graph adds recall without ever
        # pushing a genuine text hit out of the top two positions.
        expanded = getattr(self, "_graph_expanded_ids", set())
        for fact in candidates:
            jaccard = self._jaccard_similarity(query_tokens, content_stems(f'{fact["content"]} {fact.get("tags") or ""}'))
            hrr_sim = 0.5  # neutral
            fact_vec = self._phases(fact["hrr_vector"]) if self.hrr_weight > 0 and fact.get("hrr_vector") else None
            if fact_vec is not None:
                if query_vec is None:
                    query_vec = hrr.bind(hrr.encode_text(query, self.hrr_dim), self._atom(_ROLE_CONTENT))
                hrr_sim = _shift(hrr.similarity(query_vec, fact_vec))
            relevance = (self.fts_weight * fact.get("fts_rank", 0.0)
                         + self.jaccard_weight * jaccard + self.hrr_weight * hrr_sim)
            fact["score"] = relevance * fact["trust_score"]
            if self.half_life > 0:
                fact["score"] *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))
        ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        if expanded:
            text_hits = [f for f in ranked if f["fact_id"] not in expanded]
            top_score = ranked[0]["score"] if ranked else 0.0
            # Only discoveries plausibly about the query: a co-occurrence edge alone (score ~0.05 against a 0.30
            # top hit) is not evidence. This keeps junk hub neighbours out while still surfacing related facts.
            discoveries = [f for f in ranked
                           if f["fact_id"] in expanded and f["score"] >= _GRAPH_MIN_REL_RATIO * top_score]
            merged, di = [], 0
            for fact in text_hits:
                merged.append(fact)
                slot = len(merged) + 1  # 1-based position the next appended result would take
                if (di < len(discoveries) and slot >= _GRAPH_FIRST_SLOT
                        and (slot - _GRAPH_FIRST_SLOT) % _GRAPH_SLOT_EVERY == 0):
                    merged.append(discoveries[di])
                    di += 1
            merged.extend(discoveries[di:])
            ranked = merged
        results = ranked[:limit]
        for fact in results:
            fact.pop("hrr_vector", None)  # callers expect JSON-serializable dicts
            if not fact.get("retired_at"):  # active by construction: do not carry the filter columns into the output
                fact.pop("retired_at", None)
                fact.pop("retired_by", None)
            if fact.get("event_date") is None:  # no date stated in the text: the write date is what the caller has
                fact.pop("event_date", None)
        return results

    def _linked_facts(self, entity_groups: list[list[int]], category: str | None, min_trust: float, limit: int) -> list[dict]:
        """Facts linked to at least one entity of EVERY group (each group = the ids one name resolves to)."""
        joins = " INTERSECT ".join(
            f"SELECT fact_id FROM fact_entities WHERE entity_id IN ({','.join('?' * len(g))})" for g in entity_groups)
        params = [i for g in entity_groups for i in g] + [min_trust] + ([category] if category else []) + [limit]
        sql = (f"SELECT {_F_COLUMNS} FROM facts f WHERE f.fact_id IN ({joins}) AND f.retired_at IS NULL "
               f"AND f.trust_score >= ? "
               + ("AND f.category = ? " if category else "") + "ORDER BY f.trust_score DESC, f.updated_at DESC, f.fact_id DESC LIMIT ?")
        results = [dict(r) for r in self.store.read(sql, params)]
        for fact in results:
            fact["score"] = fact["trust_score"]
        return results

    def probe(self, entity: str, category: str | None = None, limit: int = 10, min_trust: float = 0.3) -> list[dict]:
        """Every fact linked to the entity (exact, by the entity index; name or alias, any letter case), most trusted
        first. A name never linked as an entity (e.g. a one-word name) falls back to keyword search."""
        ids = self.store.entity_ids(entity)
        if not ids:
            return self.search(entity, category=category, min_trust=min_trust, limit=limit)
        return self._linked_facts([ids], category, min_trust, limit)

    def reason(self, entities: list[str], category: str | None = None, limit: int = 10, min_trust: float = 0.3) -> list[dict]:
        """Facts linked to ALL the given entities. When some name is not a known entity, keyword search that still
        requires every name (FTS5 AND of the names as phrases), never facts about just one of them."""
        names = [e for e in (entities or []) if isinstance(e, str) and e.strip()]
        if not names:
            return []
        groups = [self.store.entity_ids(name) for name in names]
        if all(groups):
            return self._linked_facts(groups, category, min_trust, limit)
        unknown = [name for name, ids in zip(names, groups) if not ids]
        known = [ids for ids in groups if ids]
        spaced = [n for n in unknown if not UNSEGMENTED.search(n)]
        glued = [n.strip().lower() for n in unknown if UNSEGMENTED.search(n)]
        required = []
        for name in spaced:  # every word of the name, in any word form ("allergy" finds "allergic")
            terms = query_terms(name)
            if not terms:
                return []
            required += terms
        if required:  # every constraint first, then the 500 most trusted (newest on ties)
            allowed = {f["fact_id"] for f in self._linked_facts(known, category, min_trust, 10_000)} if known else None
            hits = sorted(((trust, fact_id) for fact_id, trust, a in self._texts(category, min_trust)
                           if (allowed is None or fact_id in allowed) and all(t.found_in(a) for t in required)),
                          reverse=True)[:500]
            lexical = self._rows([fact_id for _, fact_id in hits], category, min_trust)
        elif known:
            lexical = self._linked_facts(known, category, min_trust, 500)
        else:
            lexical = self._ngram_candidates(" ".join(glued), category, min_trust, 500)
        lexical = [f for f in lexical if all(g in (f.get("content") or "").lower() for g in glued)]
        if known:  # known names by identity (aliases included), unknown names by their words
            allowed = {f["fact_id"] for f in self._linked_facts(known, category, min_trust, 10_000)}
            lexical = [f for f in lexical if f["fact_id"] in allowed]
        return self._rerank(" ".join(names), lexical, limit)

    def related(self, entity: str, category: str | None = None, limit: int = 10, min_trust: float = 0.3) -> list[dict]:
        """Facts structurally connected to an entity (the entity's atom unbinds to a role), above a noise floor.
        Keyword search without numpy, vectors, or any fact over the floor."""
        if not hrr._HAS_NUMPY:
            return self.search(entity, category=category, min_trust=min_trust, limit=limit)
        entity_vec = self._atom(entity.strip().lower())  # bare atom, not role-bound: ANY structural match
        roles = (self._atom(_ROLE_ENTITY), self._atom(_ROLE_CONTENT))  # loop-invariant: encode once
        floor = _related_floor(self.hrr_dim)
        scored = []
        for row in self._vector_rows(category, min_trust):
            fact = dict(row)
            fact_vec = self._phases(fact.pop("hrr_vector"))
            if fact_vec is None:
                continue
            sim = max(hrr.similarity(hrr.unbind(fact_vec, entity_vec), role) for role in roles)
            if sim >= floor:
                fact["score"] = _shift(sim) * fact["trust_score"]
                scored.append(fact)
        if not scored:
            return self.search(entity, category=category, min_trust=min_trust, limit=limit)
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:limit]

    def contradict(self, category: str | None = None, threshold: float = 0.3, limit: int = 10,
                   min_trust: float = 0.0) -> list[dict]:
        """Candidate conflicts: pairs about the same subject (shared linked entities) making a similar but not
        identical claim ("Alice lives in Paris" / "Alice lives in London", a changed number, a flipped preference).
        Score = entity overlap x word overlap of the two claims with the entity names left out. Identical claims,
        a claim that only adds detail to the other ("likes apples" / "likes apples and bananas") and different
        topics about the same subject do not qualify. ``content_similarity`` is that word overlap. Word sets, so
        word order is ignored. Works without numpy."""
        sql = ("SELECT fact_id, content, category, tags, trust_score, created_at, updated_at FROM facts "
               "WHERE retired_at IS NULL AND trust_score >= ?" + (" AND category = ?" if category else "")
               + " ORDER BY COALESCE(updated_at, created_at) DESC LIMIT 500")  # bound: the most recently changed facts
        rows = self.store.read(sql, [min_trust] + ([category] if category else []))
        if len(rows) < 2:
            return []
        ids = [r["fact_id"] for r in rows]
        names_by_fact: dict[int, set[str]] = {}
        for r in self.store.read("SELECT fe.fact_id, e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id "
                                 f"WHERE fe.fact_id IN ({','.join('?' * len(ids))})", ids):
            names_by_fact.setdefault(r["fact_id"], set()).add(r["name"].lower())
        facts = []  # (public dict, lower-cased entity names, claim)
        for row in rows:
            names = names_by_fact.get(row["fact_id"])
            if names:
                entity_words = {w for name in names for w in words(name)}
                facts.append((dict(row), names, self._claim(row["content"], entity_words)))
        # Only pairs that share an entity can qualify: group by entity instead of comparing every pair.
        by_entity: dict[str, list[int]] = {}
        for index, (_, names, _) in enumerate(facts):
            for name in names:
                by_entity.setdefault(name, []).append(index)
        pairs = sorted({(a, b) for group in by_entity.values() for i, a in enumerate(group) for b in group[i + 1:]})
        contradictions = []
        for i, j in pairs:
            (f1, ents1, claim1), (f2, ents2, claim2) = facts[i], facts[j]
            entity_overlap = len(ents1 & ents2) / len(ents1 | ents2)
            if entity_overlap < 0.3:
                continue  # different subjects
            claim_sim = self._conflicting(claim1, claim2)
            contradiction_score = entity_overlap * claim_sim
            if contradiction_score >= threshold:
                contradictions.append({
                    "fact_a": f1, "fact_b": f2,
                    "entity_overlap": round(entity_overlap, 3),
                    "content_similarity": round(claim_sim, 3),
                    "contradiction_score": round(contradiction_score, 3),
                    "shared_entities": sorted(ents1 & ents2),
                })
        return sorted(contradictions, key=lambda x: x["contradiction_score"], reverse=True)[:limit]

    def _vector_rows(self, category: str | None, min_trust: float = 0.0) -> list:
        """All facts that carry an HRR vector and pass the trust filter, optionally within a category."""
        where = "WHERE retired_at IS NULL AND hrr_vector IS NOT NULL AND trust_score >= ?" + (" AND category = ?" if category else "")
        return self.store.read(f"SELECT {_FACT_COLUMNS}, hrr_vector FROM facts {where}",
                               [min_trust] + ([category] if category else []))

    def _fts_candidates(self, query: str, category: str | None, min_trust: float, limit: int) -> list[dict]:
        """FTS5 candidates for a natural-language query; [] when nothing in it is searchable."""
        expression = self._sanitize_fts_query(query)
        return self._fts_query(expression, category, min_trust, limit) if expression else []

    def _fts_query(self, expression: str, category: str | None, min_trust: float, limit: int) -> list[dict]:
        """Raw FTS5 MATCH candidates with rank normalized to [0, 1] as 'fts_rank'."""
        category_clause = "AND f.category = ? " if category else ""
        params = [expression] + ([category] if category else []) + [min_trust, limit]
        sql = ("SELECT f.*, facts_fts.rank as fts_rank_raw FROM facts_fts JOIN facts f ON f.fact_id = facts_fts.rowid "
               f"WHERE facts_fts MATCH ? {category_clause}AND f.retired_at IS NULL AND f.trust_score >= ? ORDER BY facts_fts.rank LIMIT ?")
        try:
            results = [dict(row) for row in self.store.read(sql, params)]
        except sqlite3.DatabaseError as exc:  # a damaged index (expressions are built to be valid)
            # Logged, not raised: recall must not take the turn down; a silent [] hides a broken index.
            logger.warning("holographic: FTS5 search failed (%s: %s)", type(exc).__name__, exc)
            return []
        # FTS5 rank is negative (lower = better); normalize |rank| / max to [0, 1] (1e-6 floor avoids div by zero)
        max_rank = max([abs(f["fts_rank_raw"]) for f in results] + [1e-6])
        for fact in results:
            fact["fts_rank"] = abs(fact.pop("fts_rank_raw")) / max_rank
        return results

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Lowercase whitespace tokens with surrounding punctuation stripped (no stemming)."""
        return {c for c in (w.strip(_PUNCT) for w in text.lower().split()) if c} if text else set()

    @staticmethod
    def _fts_phrase(text: str) -> str:
        """Text as one FTS5 phrase (operator characters become spaces); "" when nothing is left."""
        words = [w.strip(_PUNCT) for w in text.lower().translate(_FTS_OPERATORS).split()]
        words = [w for w in words if w]
        return f'"{" ".join(words)}"' if words else ""

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Natural-language query -> FTS5-safe OR expression. FTS5 AND-joins a multi-word MATCH by default, which
        tanks recall on prose: informative terms only (English and Russian stopwords dropped), each as its exact
        word OR its stem as a prefix ("маши" -> ("маши" OR "маш"*)), FTS5 operator characters inside a token
        turned into a phrase ("pve-01" -> "pve 01"). Nothing left -> "" (no search, never a raw MATCH)."""
        return " OR ".join(term.fts(_FTS_OPERATORS) for term in query_terms(query or "") if term.fts_ok)

    # -- prefetch relevance --------------------------------------------------------------------------------------

    def _doc_freq(self, term) -> int:
        try:
            return int(self.store.read("SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH ?", (term.fts(_FTS_OPERATORS),))[0][0])
        except sqlite3.DatabaseError:
            return 0

    def relevant(self, query: str, candidates: list[dict], min_coverage: float = 0.3) -> list[dict]:
        """Keep the candidates that answer the query, not merely share a word with it.

        Coverage = IDF-weighted share of the query's informative terms the fact contains (exact word or same stem).
        Terms no fact contains are left out of the total (they cannot be matched by anything), and a matched
        identifier (pve-01, v0.21.6) is decisive on its own. A query with no informative term ("ok, thanks",
        "да, это так") recalls nothing. Order is preserved."""
        if is_greeting(query):  # "good morning", "доброе утро", "早上好": a courtesy, not a question about mornings
            return []
        grams = query_grams(query or "")
        gram_kept = []
        if grams and candidates:
            # Unsegmented script: weighted coverage of the query's n-grams, or (stricter, since one character says
            # less than a word) IDF-weighted coverage of its meaningful characters: in Chinese a one-character
            # word (猫, cat) or a reordered phrase (名字 / 名叫) shares characters, not bigrams.
            char_weights = self._char_weights(content_chars(query))
            char_total = sum(char_weights.values())
            for f in candidates:
                text = f'{f.get("content", "")} {f.get("tags") or ""}'.lower()
                char_cov = sum(w for c, w in char_weights.items() if c in text) / char_total if char_total else 0.0
                if gram_coverage(grams, text) >= min_coverage or char_cov >= max(0.5, min_coverage):
                    gram_kept.append(f)
        terms = [t for t in query_terms(_RE_UNSEGMENTED.sub(" ", query or "")) if t.word]
        identifiers = [t for t in terms if t.identifier]
        if identifiers:  # "hr-ovh" inside Chinese text: the character overlap must not stand in for the host
            gram_kept = [f for f in gram_kept
                         if any(t.found_in(analyze(f'{f.get("content", "")} {f.get("tags") or ""}')) for t in identifiers)]
        if not terms or not candidates:
            return gram_kept
        total_facts = max(self.store.count_facts(), 1)
        scan = self._scan_terms(terms)
        texts = self._texts() if scan else []
        scan_df = {t: sum(1 for _, _, a in texts if t.found_in(a)) for t in scan}
        weights, unknown = {}, []
        for term in terms:
            df = scan_df[term] if term in scan_df else self._doc_freq(term)
            if df:
                weights[term] = math.log(1.0 + total_facts / df)
            else:
                unknown.append(term)
        if not weights:
            return gram_kept
        # A word no fact contains still says what the message is about: it counts against coverage, weighted like the
        # most common known word, so "do submarines deploy on Jupiter?" is not "about" a fact that mentions deploy.
        total = sum(weights.values()) + len(unknown) * min(weights.values())
        kept = []
        for fact in candidates:
            analysis = analyze(f'{fact.get("content", "")} {fact.get("tags") or ""}')
            if identifiers and not any(t.found_in(analysis) for t in identifiers):
                continue  # a message naming pve-01 or v0.21.6 is about that, not about pve-010
            matched = [t for t in weights if t.found_in(analysis)]
            if any(t.identifier for t in matched) or sum(weights[t] for t in matched) / total >= min_coverage \
                    or fact in gram_kept:
                kept.append(fact)
        return kept

    def _char_weights(self, chars: list[str]) -> dict[str, float]:
        """IDF weight of each character over the store; a character no fact contains weighs like the most common
        one that is present (it still says what the message is about)."""
        if not chars:
            return {}
        total = max(self.store.count_facts(), 1)
        weights, missing = {}, []
        for c in chars[:16]:
            df = self.store.read("SELECT COUNT(*) FROM facts WHERE instr(lower(content || ' ' || COALESCE(tags, '')), ?) > 0",
                                 (c,))[0][0]
            if df:
                weights[c] = math.log(1.0 + total / df)
            else:
                missing.append(c)
        if weights:
            floor = min(weights.values())
            weights.update({c: floor for c in missing})
        return weights

    # -- write-time checks and housekeeping ----------------------------------------------------------------------

    @staticmethod
    def _claim(text: str, entity_words: set[str]) -> tuple[tuple, bool]:
        """(the claim without the entity names, negators and function words, negated?). Words compare by their keys
        (word and stems); unsegmented text (Chinese, Thai) by its characters, which is what a short claim shares with
        a contradicting one."""
        claim, negated, contrast = [], False, False
        for word in analyze(text).words:
            w = word.text
            if w in entity_words:
                continue
            if UNSEGMENTED.search(w):
                for name in entity_words:
                    if UNSEGMENTED.search(name):
                        w = w.replace(name, " ")
                if is_sea(w):  # Thai letters mean nothing alone: trigrams, and "ไม่" (not)
                    negated = negated or ("ไม่" in w and not contrast)
                    claim += [frozenset({g}) for run in UNSEGMENTED.findall(w) for g, weight in run_grams(run) if weight]
                    continue
                for ch in w:
                    if ch in CJK_NEGATORS:
                        negated = negated or not contrast
                    elif ch.isalnum():
                        claim.append(frozenset({ch}))
            elif w in CONTRASTIVE:  # "likes cats but not dogs": the negation is another clause's
                contrast = True
            elif w in NEGATORS:
                negated = negated or not contrast
            elif not word.stop or w in CLAIM_WORDS:  # "with" / "without" sugar is the claim
                claim.append(word.keys)
        return tuple(dict.fromkeys(claim)), negated

    @classmethod
    def _conflicting(cls, a: tuple[frozenset, bool], b: tuple[frozenset, bool]) -> float:
        """Claim overlap when two claims about one subject look like a conflict, else 0: the same statement with
        the opposite polarity ("likes apples" / "does not like apples": at least 60 % of the shorter claim is in the
        other, so "eats meat every day" / "does not eat meat" counts and "likes cats" / "does not like cooking" does
        not), or a similar but different statement where neither only adds detail to the other ("lives in Paris" /
        "lives in London"). Lexical candidates for a person to judge; word order (who paid whom) is not modelled."""
        (claim_a, neg_a), (claim_b, neg_b) = a, b
        a_in_b, b_in_a = soft_overlap(list(claim_a), list(claim_b))
        if neg_a != neg_b:
            shorter = min(len(claim_a), len(claim_b))
            contained = min(a_in_b, b_in_a) / shorter if shorter else 0.0
            return contained if contained >= 0.6 else 0.0
        overlap = soft_jaccard(list(claim_a), list(claim_b))
        if a_in_b == len(claim_a) or b_in_a == len(claim_b):  # identical, or one only adds detail
            return 0.0
        return overlap

    def write_check(self, fact_id: int, limit: int = 3) -> dict:
        """Facts that look like a near-duplicate of fact *fact_id* or a possible conflict with it (advisory: the
        agent decides; nothing is changed). Candidates: facts sharing an entity, plus the keyword neighbours."""
        rows = self.store.read("SELECT fact_id, content FROM facts WHERE fact_id = ?", (fact_id,))
        if not rows:
            return {"similar": [], "possible_conflicts": []}
        content = rows[0]["content"]
        own_entities = {r[0] for r in self.store.read("SELECT entity_id FROM fact_entities WHERE fact_id = ?", (fact_id,))}
        candidate_ids = {r[0] for r in self.store.read(
            "SELECT DISTINCT fact_id FROM fact_entities WHERE entity_id IN ({}) AND fact_id != ? ORDER BY fact_id DESC LIMIT 50".format(
                ",".join("?" * len(own_entities)) or "NULL"), [*own_entities, fact_id])}
        candidate_ids |= {f["fact_id"] for f in self.search(content, min_trust=0.0, limit=10)} - {fact_id}
        if not candidate_ids:
            return {"similar": [], "possible_conflicts": []}
        entity_names = {r["entity_id"]: r["name"] for r in self.store.read(
            "SELECT entity_id, name FROM entities WHERE entity_id IN (SELECT entity_id FROM fact_entities WHERE fact_id IN ({}))".format(
                ",".join("?" * (len(candidate_ids) + 1))), [fact_id, *sorted(candidate_ids)])}
        links: dict[int, set[int]] = {}
        for r in self.store.read("SELECT fact_id, entity_id FROM fact_entities WHERE fact_id IN ({})".format(
                ",".join("?" * len(candidate_ids))), sorted(candidate_ids)):
            links.setdefault(r["fact_id"], set()).add(r["entity_id"])
        own_words = {w for e in own_entities for w in words(entity_names.get(e, ""))}
        own_keys, own_claim = content_keys(content), self._claim(content, own_words)
        similar, conflicts = [], []
        for other in self.store.read("SELECT fact_id, content FROM facts WHERE retired_at IS NULL AND fact_id IN ({})".format(
                ",".join("?" * len(candidate_ids))), sorted(candidate_ids)):
            other_entities = links.get(other["fact_id"], set())
            overlap = soft_jaccard(own_keys, content_keys(other["content"]))
            entry = {"fact_id": other["fact_id"], "content": other["content"], "word_overlap": round(overlap, 2)}
            shared = own_entities & other_entities
            other_words = {w for e in other_entities for w in words(entity_names.get(e, ""))}
            other_claim = self._claim(other["content"], other_words)
            score = self._conflicting(own_claim, other_claim)
            if (shared and score >= 0.3) or (not own_entities and not other_entities and score >= 0.5):
                conflicts.append(entry)  # before duplicates: "budget is 500" / "budget is 600" is an update
            elif overlap >= 0.7 and own_claim[1] == other_claim[1]:
                similar.append(entry)
        def by_overlap(xs: list[dict]) -> list[dict]:
            return sorted(xs, key=lambda e: e["word_overlap"], reverse=True)[:limit]

        return {"similar": by_overlap(similar), "possible_conflicts": by_overlap(conflicts)}

    def review(self, limit: int = 20, stale_days: int = 60) -> list[dict]:
        """Clean-up candidates with a reason each: low trust; never recalled and older than *stale_days*; a
        near-duplicate of a more trusted fact. Advisory: nothing is removed."""
        out, seen = [], set()
        def add(fact, reason):
            if fact["fact_id"] not in seen:
                seen.add(fact["fact_id"])
                out.append({"fact_id": fact["fact_id"], "content": fact["content"], "trust_score": fact["trust_score"],
                            "retrieval_count": fact["retrieval_count"], "reason": reason})
        for fact in self.store.read(f"SELECT {_FACT_COLUMNS} FROM facts WHERE retired_at IS NULL AND trust_score < 0.3 "
                                    "ORDER BY trust_score LIMIT ?", (limit,)):
            add(dict(fact), "low trust (rated unhelpful)")
        for fact in self.store.read(
                f"SELECT {_FACT_COLUMNS} FROM facts WHERE retired_at IS NULL AND retrieval_count = 0 "
                "AND julianday('now') - julianday(created_at) > ? "
                "ORDER BY created_at LIMIT ?", (stale_days, limit)):
            add(dict(fact), f"never recalled in {stale_days}+ days")
        if len(out) >= limit:
            return out[:limit]
        # near-duplicates among the most recent 200 facts (bounded pair scan), the more trusted one kept
        facts = sorted((dict(r) for r in self.store.read(
            f"SELECT {_FACT_COLUMNS} FROM facts WHERE retired_at IS NULL ORDER BY fact_id DESC LIMIT 200")),
            key=lambda f: f["trust_score"], reverse=True)
        keyed = [(f, content_stems(f["content"])) for f in facts]
        for i, (better, a) in enumerate(keyed):
            for worse, b in keyed[i + 1:]:
                if self._jaccard_similarity(a, b) >= 0.8:
                    add(worse, f"near-duplicate of #{better['fact_id']}")
        return out[:limit]

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        return len(set_a & set_b) / len(set_a | set_b) if set_a and set_b else 0.0

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """0.5^(age_days / half_life); 1.0 if disabled, missing, unparseable, or in the future."""
        if not self.half_life or not timestamp_str:
            return 1.0
        try:
            ts = datetime.fromisoformat(timestamp_str) if isinstance(timestamp_str, str) else timestamp_str  # "Z" is parsed since 3.11
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)  # stored timestamps are naive UTC by contract
            age_days = (datetime.now(UTC) - ts).total_seconds() / 86400
            return 1.0 if age_days < 0 else math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0
