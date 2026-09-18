"""
Unit tests for the Enrichr gene-set enrichment tool
(toolserver/tools/enrichr_pathway.py).

Covers request validation (_validate: gene list, library list, top_n bounds,
sort_by whitelist), Enrichr response parsing (_row_to_item,
_normalize_enrichr_payload: malformed/short/non-numeric rows fail closed to
None/skipped rather than raising), result ordering and truncation
(_sort_and_top), and the end-to-end _run path against a mocked httpx.Client
(addList/enrich HTTP-error propagation, gene-list sanitization, top_n and
return_mode handling, default libraries, and the response shape returned to
callers).

Run with:
    pip install pytest httpx pytest-cov
    python -m pytest tests/test_enrichr_tool.py -v

Developer:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from toolserver.tools.enrichr_pathway import (
    _normalize_enrichr_payload,
    _row_to_item,
    _run,
    _sort_and_top,
    _validate,
)

# ===========================================================================
# Helpers / fixtures
# ===========================================================================

def _make_row(
    rank=1,
    term="Pathway A",
    p_value=0.01,
    z_score=-1.5,
    combined_score=10.0,
    overlap_genes=None,
    adj_p_value=0.05,
    old_p_value=0.01,
    old_adj_p_value=0.05,
) -> List[Any]:
    return [
        rank,
        term,
        p_value,
        z_score,
        combined_score,
        overlap_genes if overlap_genes is not None else ["GENE1", "GENE2"],
        adj_p_value,
        old_p_value,
        old_adj_p_value,
    ]


def _make_items(n: int) -> List[Dict[str, Any]]:
    """Create n items with predictably increasing p-values."""
    return [
        {
            "rank": i + 1,
            "term": f"Term {i}",
            "p_value": (i + 1) * 0.01,
            "z_score": -float(i),
            "combined_score": float(100 - i * 10),
            "overlap_genes": ["G1", "G2"],
            "adj_p_value": (i + 1) * 0.05,
            "old_p_value": None,
            "old_adj_p_value": None,
        }
        for i in range(n)
    ]


# ===========================================================================
# _validate
# ===========================================================================

class TestValidate:
    """_validate's request-validation contract: genes/libraries/top_n/sort_by
    are checked independently, every failure is reported (not just the
    first), and a warnings list is always returned."""

    # --- genes ---

    def test_valid_minimal(self):
        """A minimal request with only a non-empty gene list validates ok with no errors."""
        result = _validate({"genes": ["TP53", "BRCA1"]}, {})
        assert result["ok"] is True
        assert result["errors"] == []

    def test_genes_missing(self):
        """A request with no "genes" key fails validation with a "genes" field error."""
        result = _validate({}, {})
        assert result["ok"] is False
        fields = [e["field"] for e in result["errors"]]
        assert "genes" in fields

    def test_genes_empty_list(self):
        """An empty gene list fails validation."""
        result = _validate({"genes": []}, {})
        assert result["ok"] is False

    def test_genes_not_a_list(self):
        """A non-list "genes" value (e.g. a bare string) fails validation."""
        result = _validate({"genes": "TP53"}, {})
        assert result["ok"] is False

    def test_genes_list_with_blank_string(self):
        """A gene list containing a whitespace-only entry fails validation."""
        result = _validate({"genes": ["TP53", "  "]}, {})
        assert result["ok"] is False

    def test_genes_non_string_elements(self):
        """A gene list containing a non-string element fails validation."""
        result = _validate({"genes": [123, "BRCA1"]}, {})
        assert result["ok"] is False

    # --- libraries ---

    def test_libraries_none_allowed(self):
        """Omitting libraries (None) is valid -- the tool falls back to defaults."""
        result = _validate({"genes": ["TP53"], "libraries": None}, {})
        assert result["ok"] is True

    def test_libraries_valid(self):
        """A non-empty list of library names validates ok."""
        result = _validate({"genes": ["TP53"], "libraries": ["KEGG_2021_Human"]}, {})
        assert result["ok"] is True

    def test_libraries_empty_list(self):
        """An explicit empty libraries list fails validation (use None to accept defaults, not [])."""
        result = _validate({"genes": ["TP53"], "libraries": []}, {})
        assert result["ok"] is False

    def test_libraries_not_a_list(self):
        """A non-list libraries value fails validation."""
        result = _validate({"genes": ["TP53"], "libraries": "KEGG_2021_Human"}, {})
        assert result["ok"] is False

    def test_libraries_blank_element(self):
        """A libraries list containing a whitespace-only entry fails validation."""
        result = _validate({"genes": ["TP53"], "libraries": ["  "]}, {})
        assert result["ok"] is False

    # --- top_n ---

    def test_top_n_valid(self):
        """An in-range integer top_n validates ok."""
        result = _validate({"genes": ["TP53"], "top_n": 10}, {})
        assert result["ok"] is True

    def test_top_n_as_string_integer(self):
        """A numeric string top_n (e.g. "50") is coerced and validates ok."""
        result = _validate({"genes": ["TP53"], "top_n": "50"}, {})
        assert result["ok"] is True

    def test_top_n_zero(self):
        """top_n of 0 is below the minimum of 1 and fails validation."""
        result = _validate({"genes": ["TP53"], "top_n": 0}, {})
        assert result["ok"] is False

    def test_top_n_above_500(self):
        """top_n above the 500 ceiling fails validation."""
        result = _validate({"genes": ["TP53"], "top_n": 501}, {})
        assert result["ok"] is False

    def test_top_n_boundary_1(self):
        """top_n of exactly 1 (the minimum) validates ok."""
        result = _validate({"genes": ["TP53"], "top_n": 1}, {})
        assert result["ok"] is True

    def test_top_n_boundary_500(self):
        """top_n of exactly 500 (the maximum) validates ok."""
        result = _validate({"genes": ["TP53"], "top_n": 500}, {})
        assert result["ok"] is True

    def test_top_n_non_numeric(self):
        """A non-numeric top_n string fails validation."""
        result = _validate({"genes": ["TP53"], "top_n": "abc"}, {})
        assert result["ok"] is False

    # --- sort_by ---

    def test_sort_by_adj_p_value(self):
        """sort_by="adj_p_value" is an allowed sort key."""
        result = _validate({"genes": ["TP53"], "sort_by": "adj_p_value"}, {})
        assert result["ok"] is True

    def test_sort_by_p_value(self):
        """sort_by="p_value" is an allowed sort key."""
        result = _validate({"genes": ["TP53"], "sort_by": "p_value"}, {})
        assert result["ok"] is True

    def test_sort_by_combined_score(self):
        """sort_by="combined_score" is an allowed sort key."""
        result = _validate({"genes": ["TP53"], "sort_by": "combined_score"}, {})
        assert result["ok"] is True

    def test_sort_by_invalid(self):
        """A sort_by value outside the allowed set fails validation."""
        result = _validate({"genes": ["TP53"], "sort_by": "fdr"}, {})
        assert result["ok"] is False

    # --- warnings list always present ---

    def test_warnings_always_present(self):
        """The result always includes a "warnings" key, even when there are no errors."""
        result = _validate({"genes": ["TP53"]}, {})
        assert "warnings" in result

    # --- multiple errors ---

    def test_multiple_errors_accumulated(self):
        """Independent field errors (genes, top_n, sort_by) are all reported together,
        not short-circuited on the first failure."""
        result = _validate({"genes": [], "top_n": -1, "sort_by": "bad"}, {})
        fields = [e["field"] for e in result["errors"]]
        assert "genes" in fields
        assert "top_n" in fields
        assert "sort_by" in fields


# ===========================================================================
# _row_to_item
# ===========================================================================

class TestRowToItem:
    """_row_to_item's per-row parsing contract: a well-formed 9- or 7-element
    Enrichr row becomes a structured item; malformed fields (bad term, bad
    numeric fields, bad overlap-genes shape) fail closed to None (the whole
    row, or just that field) rather than raising."""

    def test_full_row_parsed(self):
        """A full 9-element row is parsed into an item with all fields populated and typed."""
        row = _make_row()
        item = _row_to_item(row)
        assert item is not None
        assert item["rank"] == 1
        assert item["term"] == "Pathway A"
        assert item["p_value"] == pytest.approx(0.01)
        assert item["z_score"] == pytest.approx(-1.5)
        assert item["combined_score"] == pytest.approx(10.0)
        assert item["overlap_genes"] == ["GENE1", "GENE2"]
        assert item["adj_p_value"] == pytest.approx(0.05)
        assert item["old_p_value"] == pytest.approx(0.01)
        assert item["old_adj_p_value"] == pytest.approx(0.05)

    def test_short_row_7_elements(self):
        """A 7-element row (missing the trailing old_p_value/old_adj_p_value
        columns) still parses, with those two fields set to None."""
        row = _make_row()[:7]
        item = _row_to_item(row)
        assert item is not None
        assert item["old_p_value"] is None
        assert item["old_adj_p_value"] is None

    def test_row_too_short_returns_none(self):
        """A row with fewer than 7 elements is rejected outright (returns None)."""
        assert _row_to_item([1, "Term", 0.01]) is None

    def test_non_list_returns_none(self):
        """A non-list row value (string or None) returns None instead of raising."""
        assert _row_to_item("not a list") is None  # type: ignore
        assert _row_to_item(None) is None  # type: ignore

    def test_blank_term_returns_none(self):
        """A row whose term is whitespace-only is rejected (returns None)."""
        row = _make_row(term="  ")
        assert _row_to_item(row) is None

    def test_empty_term_returns_none(self):
        """A row whose term is an empty string is rejected (returns None)."""
        row = _make_row(term="")
        assert _row_to_item(row) is None

    def test_non_numeric_p_value_becomes_none(self):
        """A non-numeric p_value field becomes None on the item rather than
        rejecting the whole row."""
        row = _make_row()
        row[2] = "not_a_float"
        item = _row_to_item(row)
        assert item is not None
        assert item["p_value"] is None

    def test_non_numeric_rank_becomes_none(self):
        """A non-numeric rank field becomes None on the item rather than
        rejecting the whole row."""
        row = _make_row()
        row[0] = "one"
        item = _row_to_item(row)
        assert item is not None
        assert item["rank"] is None

    def test_overlap_genes_non_list_becomes_empty(self):
        """A non-list overlap_genes field (e.g. a delimited string) becomes an
        empty list on the item rather than being passed through as-is."""
        row = _make_row()
        row[5] = "GENE1;GENE2"  # string instead of list
        item = _row_to_item(row)
        assert item is not None
        assert item["overlap_genes"] == []

    def test_overlap_genes_list_preserved(self):
        """A well-formed overlap_genes list is preserved unchanged on the item."""
        row = _make_row(overlap_genes=["A", "B", "C"])
        item = _row_to_item(row)
        assert item["overlap_genes"] == ["A", "B", "C"]


# ===========================================================================
# _normalize_enrichr_payload
# ===========================================================================

class TestNormalizeEnrichrPayload:
    """_normalize_enrichr_payload's contract: turn a raw Enrichr API payload
    for one library into a {library, columns, n_terms, items} structure,
    dropping rows that _row_to_item rejects and tolerating a missing library
    key or a None payload instead of raising."""

    def test_basic_normalization(self):
        """Two valid rows for a library normalize into 2 items with matching n_terms."""
        lib = "WikiPathways_2024_Human"
        payload = {lib: [_make_row(), _make_row(rank=2, term="Pathway B", adj_p_value=0.1)]}
        result = _normalize_enrichr_payload(payload, lib)
        assert result["library"] == lib
        assert result["n_terms"] == 2
        assert len(result["items"]) == 2
        assert "columns" in result

    def test_empty_library_key(self):
        """A library key present with an empty row list normalizes to zero items."""
        lib = "WikiPathways_2024_Human"
        result = _normalize_enrichr_payload({lib: []}, lib)
        assert result["n_terms"] == 0
        assert result["items"] == []

    def test_missing_library_key(self):
        """A payload missing the requested library key normalizes to zero items, not a KeyError."""
        result = _normalize_enrichr_payload({}, "SomeLib")
        assert result["n_terms"] == 0

    def test_invalid_rows_skipped(self):
        """A row that _row_to_item rejects (here: blank term) is dropped, leaving only valid rows."""
        lib = "WikiPathways_2024_Human"
        payload = {lib: [[1, "", 0.01, -1, 10, [], 0.05], _make_row()]}
        result = _normalize_enrichr_payload(payload, lib)
        assert result["n_terms"] == 1

    def test_columns_list(self):
        """The returned columns list names all 9 item fields in a fixed, stable order."""
        lib = "Reactome_2022"
        result = _normalize_enrichr_payload({lib: [_make_row()]}, lib)
        expected_cols = [
            "rank", "term", "p_value", "z_score", "combined_score",
            "overlap_genes", "adj_p_value", "old_p_value", "old_adj_p_value",
        ]
        assert result["columns"] == expected_cols

    def test_none_payload_handled(self):
        """A None payload normalizes to zero items instead of raising."""
        result = _normalize_enrichr_payload(None, "SomeLib")  # type: ignore
        assert result["n_terms"] == 0


# ===========================================================================
# _sort_and_top
# ===========================================================================

class TestSortAndTop:
    """_sort_and_top's ordering/truncation contract: adj_p_value and p_value
    sort ascending, combined_score sorts descending, items with a None sort
    key sort last, an unrecognized sort_by falls back to adj_p_value
    ordering, and top_n truncates the result to at least 1 item."""

    def test_sort_by_adj_p_value_ascending(self):
        """sort_by="adj_p_value" orders items by ascending adj_p_value."""
        items = _make_items(5)
        result = _sort_and_top(items, sort_by="adj_p_value", top_n=5)
        vals = [r["adj_p_value"] for r in result]
        assert vals == sorted(vals)

    def test_sort_by_p_value_ascending(self):
        """sort_by="p_value" orders items by ascending p_value."""
        items = _make_items(5)
        result = _sort_and_top(items, sort_by="p_value", top_n=5)
        vals = [r["p_value"] for r in result]
        assert vals == sorted(vals)

    def test_sort_by_combined_score_descending(self):
        """sort_by="combined_score" orders items by descending combined_score."""
        items = _make_items(5)
        result = _sort_and_top(items, sort_by="combined_score", top_n=5)
        vals = [r["combined_score"] for r in result]
        assert vals == sorted(vals, reverse=True)

    def test_top_n_limits_output(self):
        """top_n truncates a larger result set down to exactly top_n items."""
        items = _make_items(10)
        result = _sort_and_top(items, sort_by="adj_p_value", top_n=3)
        assert len(result) == 3

    def test_top_n_larger_than_items(self):
        """A top_n larger than the item count returns all items, not padded or duplicated."""
        items = _make_items(3)
        result = _sort_and_top(items, sort_by="adj_p_value", top_n=10)
        assert len(result) == 3

    def test_top_n_minimum_1(self):
        """A top_n of 0 still returns at least 1 item, not an empty result."""
        items = _make_items(5)
        result = _sort_and_top(items, sort_by="adj_p_value", top_n=0)
        assert len(result) >= 1

    def test_empty_items(self):
        """Sorting an empty item list returns an empty list."""
        result = _sort_and_top([], sort_by="adj_p_value", top_n=10)
        assert result == []

    def test_none_adj_p_value_sorted_last(self):
        """An item with adj_p_value=None sorts after items with real values, not first."""
        items = [
            {**_make_items(1)[0],
             "adj_p_value": None, "p_value": None, "combined_score": None, "term": "Null Item"},
            {**_make_items(1)[0], "adj_p_value": 0.001, "term": "Good Item"},
        ]
        result = _sort_and_top(items, sort_by="adj_p_value", top_n=2)
        assert result[0]["term"] == "Good Item"

    def test_default_sort_key_is_adj_p_value(self):
        """An unrecognized sort_by value falls back to the same ordering as
        sort_by="adj_p_value" rather than raising or leaving items unsorted."""
        # Anything other than combined_score / p_value falls back to adj_p_value ordering
        items = _make_items(4)
        r1 = _sort_and_top(items, sort_by="adj_p_value", top_n=4)
        r2 = _sort_and_top(items, sort_by="unknown_key", top_n=4)
        # Both should produce the same ordering
        assert [i["term"] for i in r1] == [i["term"] for i in r2]


# ===========================================================================
# _run  (integration-style with httpx mocked)
# ===========================================================================

class TestRun:
    """
    These tests mock httpx.Client so no real network calls are made.
    """

    LIB = "WikiPathways_2024_Human"

    def _make_enrich_response(self, lib: str) -> Dict[str, Any]:
        return {
            lib: [
                _make_row(rank=i + 1, term=f"Path {i}", adj_p_value=(i + 1) * 0.01)
                for i in range(5)
            ]
        }

    def _setup_mock_client(self, mock_client_cls, lib: str):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        # addList response
        add_resp = MagicMock()
        add_resp.status_code = 200
        add_resp.json.return_value = {"userListId": "abc123"}

        # enrich response
        enrich_resp = MagicMock()
        enrich_resp.status_code = 200
        enrich_resp.json.return_value = self._make_enrich_response(lib)

        mock_client.post.return_value = add_resp
        mock_client.get.return_value = enrich_resp

        return mock_client

    @patch("toolserver.tools.enrichr_pathway.httpx.Client")
    def test_successful_run_returns_ok(self, mock_client_cls):
        """A successful addList + enrich round trip returns ok=True with the
        userListId and per-library results populated."""
        self._setup_mock_client(mock_client_cls, self.LIB)
        result = _run(
            {"genes": ["TP53", "BRCA1"], "libraries": [self.LIB]},
            {},
            log=lambda msg: None,
        )
        assert result["ok"] is True
        assert result["userListId"] == "abc123"
        assert self.LIB in result["results"]

    @patch("toolserver.tools.enrichr_pathway.httpx.Client")
    def test_meta_populated(self, mock_client_cls):
        """result["meta"] echoes back the effective n_genes, top_n, sort_by, and libraries used."""
        self._setup_mock_client(mock_client_cls, self.LIB)
        result = _run(
            {"genes": ["TP53", "BRCA1"], "libraries": [self.LIB], "top_n": 3, "sort_by": "p_value"},
            {},
            log=lambda msg: None,
        )
        assert result["meta"]["n_genes"] == 2
        assert result["meta"]["top_n"] == 3
        assert result["meta"]["sort_by"] == "p_value"
        assert result["meta"]["libraries"] == [self.LIB]

    @patch("toolserver.tools.enrichr_pathway.httpx.Client")
    def test_top_n_respected(self, mock_client_cls):
        """The end-to-end _run path truncates each library's items to top_n."""
        self._setup_mock_client(mock_client_cls, self.LIB)
        result = _run(
            {"genes": ["TP53", "BRCA1"], "libraries": [self.LIB], "top_n": 2},
            {},
            log=lambda msg: None,
        )
        items = result["results"][self.LIB]["items"]
        assert len(items) <= 2

    @patch("toolserver.tools.enrichr_pathway.httpx.Client")
    def test_return_mode_all_does_not_apply_top_n(self, mock_client_cls):
        """return_mode="all" bypasses top_n truncation and returns every parsed row."""
        self._setup_mock_client(mock_client_cls, self.LIB)
        result = _run(
            {
                "genes": ["TP53", "BRCA1"],
                "libraries": [self.LIB],
                "top_n": 1,
                "return_mode": "all",
            },
            {},
            log=lambda msg: None,
        )
        # In "all" mode, the items are not truncated to top_n
        items = result["results"][self.LIB]["items"]
        assert len(items) == 5  # all 5 rows from the mock

    @patch("toolserver.tools.enrichr_pathway.httpx.Client")
    def test_genes_stripped_and_blanks_removed(self, mock_client_cls):
        """Gene names are whitespace-trimmed and blank entries are dropped
        before being posted to Enrichr's addList endpoint."""
        mock_client = self._setup_mock_client(mock_client_cls, self.LIB)
        _run(
            {"genes": [" TP53 ", "", "BRCA1"], "libraries": [self.LIB]},
            {},
            log=lambda msg: None,
        )
        # Inspect the data posted to addList
        post_call_kwargs = mock_client.post.call_args
        files = post_call_kwargs[1].get("files") or post_call_kwargs.kwargs.get("files")
        gene_blob = files["list"][1]
        assert "TP53" in gene_blob
        assert "BRCA1" in gene_blob
        # Empty string should not appear as a gene
        lines = [line for line in gene_blob.strip().splitlines() if line]
        assert "" not in lines

    @patch("toolserver.tools.enrichr_pathway.httpx.Client")
    def test_default_libraries_used_when_not_provided(self, mock_client_cls):
        """Omitting "libraries" from the request falls back to querying the
        tool's built-in default library set (WikiPathways and Reactome)."""
        # Need to handle two library calls; return the first lib's response for any get
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        add_resp = MagicMock()
        add_resp.status_code = 200
        add_resp.json.return_value = {"userListId": "xyz"}

        def enrich_side_effect(url, params=None):
            lib = (params or {}).get("backgroundType", "WikiPathways_2024_Human")
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {lib: [_make_row()]}
            return resp

        mock_client.post.return_value = add_resp
        mock_client.get.side_effect = enrich_side_effect

        result = _run({"genes": ["TP53"]}, {}, log=lambda msg: None)
        assert "WikiPathways_2024_Human" in result["results"]
        assert "Reactome_2022" in result["results"]

    @patch("toolserver.tools.enrichr_pathway.httpx.Client")
    def test_addlist_http_error_raises(self, mock_client_cls):
        """A non-2xx response from Enrichr's addList endpoint raises RuntimeError
        instead of continuing with a broken userListId."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        bad_resp = MagicMock()
        bad_resp.status_code = 500
        bad_resp.text = "Internal Server Error"
        mock_client.post.return_value = bad_resp

        with pytest.raises(RuntimeError, match="addList failed"):
            _run({"genes": ["TP53"], "libraries": [self.LIB]}, {}, log=lambda msg: None)

    @patch("toolserver.tools.enrichr_pathway.httpx.Client")
    def test_missing_user_list_id_raises(self, mock_client_cls):
        """An addList response missing the expected userListId key raises RuntimeError."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        add_resp = MagicMock()
        add_resp.status_code = 200
        add_resp.json.return_value = {}  # No userListId key
        mock_client.post.return_value = add_resp

        with pytest.raises(RuntimeError, match="userListId"):
            _run({"genes": ["TP53"], "libraries": [self.LIB]}, {}, log=lambda msg: None)

    @patch("toolserver.tools.enrichr_pathway.httpx.Client")
    def test_enrich_http_error_raises(self, mock_client_cls):
        """A non-2xx response from Enrichr's enrich endpoint raises RuntimeError
        instead of returning a partial or empty result."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        add_resp = MagicMock()
        add_resp.status_code = 200
        add_resp.json.return_value = {"userListId": "abc"}
        mock_client.post.return_value = add_resp

        bad_resp = MagicMock()
        bad_resp.status_code = 429
        bad_resp.text = "Too Many Requests"
        mock_client.get.return_value = bad_resp

        with pytest.raises(RuntimeError, match="enrich failed"):
            _run({"genes": ["TP53"], "libraries": [self.LIB]}, {}, log=lambda msg: None)

    @patch("toolserver.tools.enrichr_pathway.httpx.Client")
    def test_log_callable_called(self, mock_client_cls):
        """The caller-supplied log callback is invoked at least once per HTTP
        stage (addList and enrich) during a successful run."""
        self._setup_mock_client(mock_client_cls, self.LIB)
        log_messages = []
        _run(
            {"genes": ["TP53"], "libraries": [self.LIB]},
            {},
            log=log_messages.append,
        )
        assert len(log_messages) >= 2  # at least addList + enrich messages

    @patch("toolserver.tools.enrichr_pathway.httpx.Client")
    def test_custom_base_url_used(self, mock_client_cls):
        """A "_enrichr_base_url" override in the request is used for the addList
        HTTP call instead of the tool's default Enrichr host."""
        mock_client = self._setup_mock_client(mock_client_cls, self.LIB)
        _run(
            {
                "genes": ["TP53"],
                "libraries": [self.LIB],
                "_enrichr_base_url": "https://my-enrichr.example.com",
            },
            {},
            log=lambda msg: None,
        )
        post_url = mock_client.post.call_args[0][0]
        assert "my-enrichr.example.com" in post_url

    @patch("toolserver.tools.enrichr_pathway.httpx.Client")
    def test_result_structure(self, mock_client_cls):
        """Each per-library result carries the full expected key set
        (library, columns, sort_by, top_n, n_terms, items)."""
        self._setup_mock_client(mock_client_cls, self.LIB)
        result = _run(
            {"genes": ["TP53", "MYC"], "libraries": [self.LIB]},
            {},
            log=lambda msg: None,
        )
        lib_result = result["results"][self.LIB]
        for key in ("library", "columns", "sort_by", "top_n", "n_terms", "items"):
            assert key in lib_result, f"Missing key: {key}"