"""
OmniBioAI toolserver.tools.enrichr_pathway.

Purpose:
    Defines _validate, _row_to_item, _normalize_enrichr_payload and _sort_and_top for toolserver.tools.enrichr_pathway.

Author:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import httpx


# ----------------------------
# Validation
# ----------------------------
def _validate(inputs: Dict[str, Any], resources: Dict[str, Any]) -> Dict[str, Any]:
    errors = []

    genes = inputs.get("genes")
    if not isinstance(genes, list) or not genes or not all(isinstance(g, str) and g.strip() for g in genes):
        errors.append({"field": "genes", "message": "genes must be a non-empty list of strings"})

    libraries = inputs.get("libraries")
    if libraries is not None:
        if not isinstance(libraries, list) or not libraries or not all(
            isinstance(x, str) and x.strip() for x in libraries
        ):
            errors.append({"field": "libraries", "message": "libraries must be a non-empty list of strings"})

    # Optional knobs (safe defaults)
    top_n = inputs.get("top_n")
    if top_n is not None:
        try:
            n = int(top_n)
            if n < 1 or n > 500:
                errors.append({"field": "top_n", "message": "top_n must be between 1 and 500"})
        except Exception:
            errors.append({"field": "top_n", "message": "top_n must be an integer"})

    sort_by = inputs.get("sort_by")
    if sort_by is not None:
        if str(sort_by) not in ("adj_p_value", "p_value", "combined_score"):
            errors.append(
                {
                    "field": "sort_by", 
                    "message": "sort_by must be one of: adj_p_value, p_value, combined_score"
                }
            )

    return {"ok": len(errors) == 0, "errors": errors, "warnings": []}


# ----------------------------
# Enrichr parsing helpers
# ----------------------------
# Enrichr standard row:
# [rank, term, p_value, z_score, combined_score, overlap_genes, adj_p_value, old_p_value, old_adj_p_value]
def _row_to_item(row: List[Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(row, list) or len(row) < 7:
        return None

    def _f(x):
        try:
            return float(x)
        except Exception:
            return None

    def _i(x):
        try:
            return int(x)
        except Exception:
            return None

    item = {
        "rank": _i(row[0]),
        "term": row[1],
        "p_value": _f(row[2]),
        "z_score": _f(row[3]),
        "combined_score": _f(row[4]),
        "overlap_genes": row[5] if isinstance(row[5], list) else [],
        "adj_p_value": _f(row[6]),
        "old_p_value": _f(row[7]) if len(row) > 7 else None,
        "old_adj_p_value": _f(row[8]) if len(row) > 8 else None,
    }

    if not isinstance(item["term"], str) or not item["term"].strip():
        return None
    return item


def _normalize_enrichr_payload(payload: Dict[str, Any], library: str) -> Dict[str, Any]:
    """
    Enrichr returns JSON shaped like:
      {"<library>": [[...row...], [...]]}

    Normalize to:
      {"library": "<library>", "columns": [...], "items": [...], "n_terms": N}
    """
    rows = (payload or {}).get(library, [])
    items: List[Dict[str, Any]] = []
    for r in rows:
        it = _row_to_item(r)
        if it is not None:
            items.append(it)

    return {
        "library": library,
        "columns": [
            "rank",
            "term",
            "p_value",
            "z_score",
            "combined_score",
            "overlap_genes",
            "adj_p_value",
            "old_p_value",
            "old_adj_p_value",
        ],
        "items": items,
        "n_terms": len(items),
    }


def _sort_and_top(items: List[Dict[str, Any]], sort_by: str, top_n: int) -> List[Dict[str, Any]]:
    # Stable, defensive sorting (None-safe)
    def key_adj(x):
        return (
            x.get("adj_p_value") if x.get("adj_p_value") is not None else 1e9,
            x.get("p_value") if x.get("p_value") is not None else 1e9,
            -(x.get("combined_score") if x.get("combined_score") is not None else -1e9),
        )

    def key_p(x):
        return (
            x.get("p_value") if x.get("p_value") is not None else 1e9,
            x.get("adj_p_value") if x.get("adj_p_value") is not None else 1e9,
            -(x.get("combined_score") if x.get("combined_score") is not None else -1e9),
        )

    def key_combined(x):
        return (
            -(x.get("combined_score") if x.get("combined_score") is not None else -1e9),
            x.get("adj_p_value") if x.get("adj_p_value") is not None else 1e9,
            x.get("p_value") if x.get("p_value") is not None else 1e9,
        )

    if sort_by == "combined_score":
        items_sorted = sorted(items, key=key_combined)
    elif sort_by == "p_value":
        items_sorted = sorted(items, key=key_p)
    else:
        items_sorted = sorted(items, key=key_adj)

    return items_sorted[: max(1, top_n)]


# ----------------------------
# Tool execution
# ----------------------------
def _run(
    inputs: Dict[str, Any],
    resources: Dict[str, Any],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    base_url = str(inputs.get("_enrichr_base_url", "https://maayanlab.cloud/Enrichr")).rstrip("/")
    timeout_s = float(inputs.get("_timeout_s", 30.0))

    genes: List[str] = [g.strip() for g in inputs["genes"] if isinstance(g, str) and g.strip()]
    libraries: List[str] = inputs.get("libraries") or ["WikiPathways_2024_Human", "Reactome_2022"]
    description = str(inputs.get("description", "toolserver enrichr run"))

    # Output shaping knobs
    top_n = int(inputs.get("top_n", 25))
    sort_by = str(inputs.get("sort_by", "adj_p_value"))  # adj_p_value | p_value | combined_score
    return_mode = str(inputs.get("return_mode", "top"))  # "top" or "all"

    # Enrichr expects newline-separated genes; trailing newline improves compatibility.
    gene_blob = "\n".join(genes) + "\n"

    headers = {
        "User-Agent": "omnibioai-toolserver/0.1",
        "Accept": "application/json",
    }

    # 1) addList (multipart/form-data; required by Enrichr in practice)
    log("Enrichr: addList")
    with httpx.Client(timeout=timeout_s, headers=headers) as c:
        r = c.post(
            f"{base_url}/addList",
            files={
                "list": (None, gene_blob),
                "description": (None, description),
            },
        )
        if r.status_code != 200:
            raise RuntimeError(f"Enrichr addList failed: {r.status_code} body={r.text[:1000]}")
        data = r.json()

    user_list_id = str(data.get("userListId") or data.get("userListID") or "")
    if not user_list_id:
        raise RuntimeError(f"Enrichr addList did not return userListId: {data}")

    log(f"Enrichr: userListId={user_list_id}")

    # 2) enrich per library + normalize
    results_by_lib: Dict[str, Any] = {}

    with httpx.Client(timeout=timeout_s, headers=headers) as c:
        for lib in libraries:
            log(f"Enrichr: enrich {lib}")
            rr = c.get(f"{base_url}/enrich", params={"userListId": user_list_id, "backgroundType": lib})
            if rr.status_code != 200:
                raise RuntimeError(f"Enrichr enrich failed for {lib}: {rr.status_code} body={rr.text[:1000]}")

            raw = rr.json()
            norm = _normalize_enrichr_payload(raw, lib)

            if return_mode == "all":
                # keep all terms, just normalized
                results_by_lib[lib] = norm
            else:
                # top-N view
                top_items = _sort_and_top(norm["items"], sort_by=sort_by, top_n=top_n)
                results_by_lib[lib] = {
                    "library": lib,
                    "columns": norm["columns"],
                    "sort_by": sort_by,
                    "top_n": top_n,
                    "n_terms": norm["n_terms"],
                    "items": top_items,
                }

    return {
        "ok": True,
        "userListId": user_list_id,
        "results": results_by_lib,
        "meta": {
            "sort_by": sort_by,
            "top_n": top_n,
            "return_mode": return_mode,
            "libraries": libraries,
            "n_genes": len(genes),
        },
    }
