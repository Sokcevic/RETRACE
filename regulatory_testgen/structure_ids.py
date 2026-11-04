"""Canonical clause-id namespacing — the SINGLE source of truth shared by the
knowledge graph (regulatory_testgen.graph) and the chat clause index
(chat_ui.normalize). Previously the namespacing lived only in chat_ui.normalize,
so the graph ran on bare ids while the index namespaced annex clauses — and since
every annex restarts numbering at 1, annex ids collided with body ids and clauses
were silently overwritten in the graph. Centralizing it here keeps both consistent.

A UN regulation's annexes reuse the body's paragraph numbers, so a bare id like
"5" is ambiguous. We prefix annex clauses with their container ("Annex 3 / 5") using
document order (line_start) and the annex/appendix heading clauses. Parentage is then
fully derivable from the id string (parent_of).
"""

from __future__ import annotations

import re

_NUM_RE = re.compile(r"^[A-Z]?\d+(\.\d+)*$")  # optional letter prefix: ADGM "A1.2"
_APPENDIX_RE = re.compile(r"^(Annex\b.*?)\s*-\s*Appendix\b", re.IGNORECASE)
_ANNEX_RE = re.compile(r"^Annex\b", re.IGNORECASE)
_SEP = " / "  # separates an annex container from its inner numbering


def parent_of(clause_id: str) -> str | None:
    """The structural parent id, derived purely from the id string.

      "5.2.1"                -> "5.2"
      "5"                    -> None
      "Annex 3 / 5.1"        -> "Annex 3 / 5"
      "Annex 3 / 5"          -> "Annex 3"
      "Annex 3 - Appendix 1" -> "Annex 3"
      "Annex 3"              -> None
    """
    if _SEP in clause_id:
        container, tail = clause_id.rsplit(_SEP, 1)
        if "." in tail and _NUM_RE.match(tail):
            return f"{container}{_SEP}{tail.rsplit('.', 1)[0]}"
        return container
    m = _APPENDIX_RE.match(clause_id)
    if m:
        return m.group(1).strip()
    if _ANNEX_RE.match(clause_id):
        return None
    if _NUM_RE.match(clause_id) and "." in clause_id:
        return clause_id.rsplit(".", 1)[0]
    return None


def assign_structure(clauses: list[dict]) -> list[dict]:
    """Namespace annex clauses by document order so their numbering no longer
    collides with the body. Returns new dicts (inputs untouched); numeric clauses
    inside an annex/appendix get their clause_id prefixed with the container id.
    Body clauses (before the first annex heading) keep their bare ids.
    """
    ordered = sorted(clauses, key=lambda c: c.get("line_start", 0))
    container: str | None = None
    out: list[dict] = []
    for c in ordered:
        cid = (c.get("clause_id") or "").strip()
        if not cid or c.get("is_pseudo_clause"):
            out.append(c)
            continue
        if _NUM_RE.match(cid):
            if container:
                c = {**c, "clause_id": f"{container}{_SEP}{cid}"}
            out.append(c)
            continue
        if _APPENDIX_RE.match(cid):
            container = cid
            out.append(c)
        elif _ANNEX_RE.match(cid):
            container = cid
            out.append(c)
        else:
            if container:
                c = {**c, "clause_id": f"{container}{_SEP}{cid}"}
            out.append(c)
    return out


def namespace_clauses(clauses: list):
    """Return copies of the given Clause objects with annex-namespaced clause_ids,
    using the exact assign_structure logic (via each clause's uid). Idempotent: if
    the clauses are already namespaced (any id contains the separator), returns them
    unchanged. Clauses without a uid keep their bare id.
    """
    if any(_SEP in (getattr(c, "clause_id", "") or "") for c in clauses):
        return clauses
    raw = [
        {
            "clause_id": getattr(c, "clause_id", "") or "",
            "line_start": getattr(c, "line_start", 0) or 0,
            "is_pseudo_clause": getattr(c, "is_pseudo_clause", False),
            "uid": getattr(c, "uid", "") or "",
        }
        for c in clauses
    ]
    uid2ns = {r["uid"]: r["clause_id"] for r in assign_structure(raw) if r.get("uid")}
    out = []
    for c in clauses:
        uid = getattr(c, "uid", "") or ""
        ns = uid2ns.get(uid, getattr(c, "clause_id", ""))
        out.append(c.model_copy(update={"clause_id": ns}) if hasattr(c, "model_copy") else c)
    return out
