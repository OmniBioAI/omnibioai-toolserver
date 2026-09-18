"""Tests for toolserver/tools/david_annotation.py, the DAVID functional
annotation tool adapter.

Covers input validation (_validate), the raw SOAP request/response
mechanics against the DAVID web service (_soap), chart-record XML parsing
with and without XML namespaces (_parse_chart_records), and the end-to-end
tool run that authenticates, submits a gene list, and retrieves formatted
enrichment results (_run).

Developer:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from toolserver.tools.david_annotation import _parse_chart_records, _run, _soap, _validate


# ═════════════════════════════════════════════════════════════════════════════
# _validate
# ═════════════════════════════════════════════════════════════════════════════

class TestValidate:
    """_validate's required-field checks for the DAVID annotation tool's inputs."""

    def test_valid_inputs_pass(self):
        """A request with both email and gene_ids passes with no errors or warnings."""
        result = _validate({"email": "user@example.com", "gene_ids": "1,2,3"}, {})
        assert result["ok"] is True
        assert result["errors"] == []
        assert result["warnings"] == []

    def test_missing_email_fails(self):
        """Omitting email fails validation and reports an error on the email field."""
        result = _validate({"gene_ids": "1,2,3"}, {})
        assert result["ok"] is False
        assert any(e["field"] == "email" for e in result["errors"])

    def test_empty_email_fails(self):
        """An empty-string email fails validation."""
        result = _validate({"email": "", "gene_ids": "1,2,3"}, {})
        assert result["ok"] is False

    def test_missing_gene_ids_fails(self):
        """Omitting gene_ids fails validation and reports an error on the gene_ids field."""
        result = _validate({"email": "user@example.com"}, {})
        assert result["ok"] is False
        assert any(e["field"] == "gene_ids" for e in result["errors"])

    def test_empty_gene_ids_fails(self):
        """An empty gene_ids list fails validation."""
        result = _validate({"email": "user@example.com", "gene_ids": []}, {})
        assert result["ok"] is False

    def test_missing_both_gives_two_errors(self):
        """Omitting both required fields reports exactly one error per field."""
        result = _validate({}, {})
        assert result["ok"] is False
        assert len(result["errors"]) == 2

    def test_error_message_references_field(self):
        """Each validation error names the specific field it applies to."""
        result = _validate({}, {})
        fields = {e["field"] for e in result["errors"]}
        assert fields == {"email", "gene_ids"}


# ═════════════════════════════════════════════════════════════════════════════
# _soap
# ═════════════════════════════════════════════════════════════════════════════

class TestSoap:
    """_soap's construction of DAVID SOAP requests and handling of responses."""

    def _make_client(self, response_text="<ok/>", raise_error=False):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = response_text
        if raise_error:
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=mock_resp
            )
        mock_client.post.return_value = mock_resp
        return mock_client

    def test_posts_to_david_ws_url(self):
        """The request is posted to the DAVID web service URL."""
        client = self._make_client()
        _soap(client, "authenticate", "<body/>")
        url = client.post.call_args[0][0]
        assert "DAVIDWebService" in url

    def test_sets_content_type_header(self):
        """The request declares a text/xml Content-Type header."""
        client = self._make_client()
        _soap(client, "action", "<body/>")
        headers = client.post.call_args[1]["headers"]
        assert headers["Content-Type"] == "text/xml"

    def test_sets_soap_action_header(self):
        """The SOAPAction header carries the requested action name."""
        client = self._make_client()
        _soap(client, "myAction", "<body/>")
        headers = client.post.call_args[1]["headers"]
        assert "myAction" in headers["SOAPAction"]

    def test_includes_body_in_envelope(self):
        """The caller-supplied SOAP body is embedded in the request envelope."""
        client = self._make_client()
        _soap(client, "action", "<custom>data</custom>")
        content = client.post.call_args[1]["content"]
        assert "<custom>data</custom>" in content

    def test_returns_response_text(self):
        """_soap returns the raw response text unchanged."""
        client = self._make_client("expected_response_text")
        result = _soap(client, "action", "<body/>")
        assert result == "expected_response_text"

    def test_raise_for_status_called(self):
        """_soap checks the HTTP response status before returning."""
        client = self._make_client()
        _soap(client, "action", "<body/>")
        client.post.return_value.raise_for_status.assert_called_once()

    def test_propagates_http_error(self):
        """An HTTP error status from DAVID propagates out of _soap unhandled."""
        client = self._make_client(raise_error=True)
        with pytest.raises(httpx.HTTPStatusError):
            _soap(client, "action", "<body/>")


# ═════════════════════════════════════════════════════════════════════════════
# _parse_chart_records
# ═════════════════════════════════════════════════════════════════════════════

class TestParseChartRecords:
    """_parse_chart_records's extraction of <return> records from DAVID chart XML."""

    def test_bare_return_element(self):
        """A <return> element with no XML namespace is parsed into a record."""
        xml = "<root><return><termName>GO:0001</termName></return></root>"
        records = _parse_chart_records(xml)
        assert len(records) == 1
        assert records[0]["termName"] == "GO:0001"

    def test_namespaced_return_element(self):
        """A namespaced <ns:return> element is parsed the same as a bare one."""
        xml = (
            '<root xmlns:ns="http://service.session.sample">'
            "<ns:return><ns:termName>GO:0002</ns:termName></ns:return>"
            "</root>"
        )
        records = _parse_chart_records(xml)
        assert len(records) == 1
        assert records[0]["termName"] == "GO:0002"

    def test_mixed_namespaced_and_bare_children(self):
        """A <return> element's namespaced and bare child tags are both captured."""
        xml = (
            '<root xmlns:ns="http://example.com">'
            "<ns:return><ns:cat>BP</ns:cat><term>GO:003</term></ns:return>"
            "</root>"
        )
        records = _parse_chart_records(xml)
        assert records[0]["cat"] == "BP"
        assert records[0]["term"] == "GO:003"

    def test_multiple_records(self):
        """Multiple <return> elements each produce their own record, in document order."""
        xml = (
            "<root>"
            "<return><t>A</t></return>"
            "<return><t>B</t></return>"
            "</root>"
        )
        records = _parse_chart_records(xml)
        assert len(records) == 2
        assert records[0]["t"] == "A"
        assert records[1]["t"] == "B"

    def test_invalid_xml_returns_empty(self):
        """Malformed XML yields an empty record list rather than raising."""
        records = _parse_chart_records("<<<invalid xml>>>")
        assert records == []

    def test_empty_string_returns_empty(self):
        """An empty input string yields an empty record list rather than raising."""
        records = _parse_chart_records("")
        assert records == []

    def test_empty_return_element_skipped(self):
        """A <return> element with no child fields is skipped, not returned as an empty record."""
        xml = "<root><return></return><return><t>valid</t></return></root>"
        records = _parse_chart_records(xml)
        assert len(records) == 1
        assert records[0]["t"] == "valid"

    def test_no_return_elements_returns_empty(self):
        """XML with no <return> elements at all yields an empty record list."""
        xml = (
            "<soapenv:Envelope "
            "xmlns:soapenv='http://schemas.xmlsoap.org/soap/envelope/'>"
            "<soapenv:Body/>"
            "</soapenv:Envelope>"
        )
        records = _parse_chart_records(xml)
        assert records == []

    def test_all_fields_extracted(self):
        """Every child field of a <return> element is extracted into the record."""
        xml = (
            "<root><return>"
            "<categoryName>GOTERM_BP</categoryName>"
            "<termName>GO:0006915</termName>"
            "<geneIds>CASP3,CASP9</geneIds>"
            "<listHits>10</listHits>"
            "<foldEnrichment>3.5</foldEnrichment>"
            "<ease>0.001</ease>"
            "<fisher>0.0001</fisher>"
            "<benjamini>0.002</benjamini>"
            "<bonferroni>0.003</bonferroni>"
            "</return></root>"
        )
        records = _parse_chart_records(xml)
        assert records[0]["categoryName"] == "GOTERM_BP"
        assert records[0]["benjamini"] == "0.002"


# ═════════════════════════════════════════════════════════════════════════════
# _run
# ═════════════════════════════════════════════════════════════════════════════

class TestRun:
    """_run's end-to-end DAVID annotation flow: authenticate, submit, fetch chart, format."""

    def _soap_iter(self, responses):
        it = iter(responses)
        return lambda *args, **kwargs: next(it)

    def test_successful_run_returns_ok(self):
        """A full successful run (auth, addList, setCategories, getChartReport) reports ok."""
        responses = ["true", "<ok/>", "<ok/>", "<ok/>"]
        with patch("toolserver.tools.david_annotation._soap", side_effect=self._soap_iter(responses)):
            result = _run({"email": "test@lab.com", "gene_ids": "1,2,3"}, {}, MagicMock())
        assert result["ok"] is True
        assert result["n_terms"] == 0
        assert result["results"] == []

    def test_run_meta_reflects_inputs(self):
        """The run's meta block reflects the caller-supplied email and id_type."""
        responses = ["true", "<ok/>", "<ok/>", "<ok/>"]
        with patch("toolserver.tools.david_annotation._soap", side_effect=self._soap_iter(responses)):
            result = _run(
                {"email": "user@lab.com", "gene_ids": "100,200", "id_type": "GENE_SYMBOL"},
                {}, MagicMock()
            )
        assert result["meta"]["email"] == "user@lab.com"
        assert result["meta"]["id_type"] == "GENE_SYMBOL"

    def test_custom_params_reflected_in_meta(self):
        """Custom threshold, count, and categories parameters are reflected in the run's meta block."""
        responses = ["true", "<ok/>", "<ok/>", "<ok/>"]
        with patch("toolserver.tools.david_annotation._soap", side_effect=self._soap_iter(responses)):
            result = _run(
                {
                    "email": "test@test.com",
                    "gene_ids": "1",
                    "threshold": 0.05,
                    "count": 5,
                    "id_type": "GENE_SYMBOL",
                    "categories": "KEGG_PATHWAY",
                    "list_name": "mylist",
                    "_timeout_s": 30.0,
                },
                {}, MagicMock()
            )
        assert result["meta"]["threshold"] == 0.05
        assert result["meta"]["count"] == 5
        assert result["meta"]["categories"] == "KEGG_PATHWAY"

    def test_auth_failure_raises_runtime_error(self):
        """A DAVID authentication failure raises RuntimeError instead of proceeding."""
        responses = ["<auth>FALSE</auth>"]
        with patch("toolserver.tools.david_annotation._soap", side_effect=self._soap_iter(responses)):
            with pytest.raises(RuntimeError, match="authentication failed"):
                _run({"email": "bad@test.com", "gene_ids": "1"}, {}, MagicMock())

    def test_addlist_fault_raises_runtime_error(self):
        """A SOAP Fault from the addList step raises RuntimeError instead of proceeding."""
        responses = ["true", "<Fault>bad gene list</Fault>"]
        with patch("toolserver.tools.david_annotation._soap", side_effect=self._soap_iter(responses)):
            with pytest.raises(RuntimeError, match="addList failed"):
                _run({"email": "test@test.com", "gene_ids": "bad"}, {}, MagicMock())

    def test_chart_records_are_parsed_and_formatted(self):
        """A single chart record from DAVID is parsed and reformatted into the run's result fields."""
        chart_xml = (
            "<root><return>"
            "<categoryName>GOTERM_BP_DIRECT</categoryName>"
            "<termName>GO:0006915~apoptosis</termName>"
            "<ease>0.001</ease>"
            "<foldEnrichment>3.5</foldEnrichment>"
            "<geneIds>CASP3,CASP9</geneIds>"
            "<listHits>10</listHits>"
            "<fisher>0.0001</fisher>"
            "<benjamini>0.002</benjamini>"
            "<bonferroni>0.003</bonferroni>"
            "</return></root>"
        )
        responses = ["true", "<ok/>", "<ok/>", chart_xml]
        with patch("toolserver.tools.david_annotation._soap", side_effect=self._soap_iter(responses)):
            result = _run({"email": "test@test.com", "gene_ids": "1,2,3"}, {}, MagicMock())

        assert result["n_terms"] == 1
        r = result["results"][0]
        assert r["category"] == "GOTERM_BP_DIRECT"
        assert r["term"] == "GO:0006915~apoptosis"
        assert r["genes"] == "CASP3,CASP9"
        assert r["p_value"] == "0.001"
        assert r["fold_enrichment"] == "3.5"
        assert r["fisher"] == "0.0001"
        assert r["benjamini"] == "0.002"
        assert r["bonferroni"] == "0.003"

    def test_multiple_chart_records(self):
        """Multiple chart records from DAVID are each parsed into their own result entry, in order."""
        chart_xml = (
            "<root>"
            "<return><categoryName>GO_BP</categoryName><termName>T1</termName></return>"
            "<return><categoryName>KEGG</categoryName><termName>T2</termName></return>"
            "</root>"
        )
        responses = ["true", "<ok/>", "<ok/>", chart_xml]
        with patch("toolserver.tools.david_annotation._soap", side_effect=self._soap_iter(responses)):
            result = _run({"email": "test@test.com", "gene_ids": "1"}, {}, MagicMock())
        assert result["n_terms"] == 2
        assert result["results"][0]["term"] == "T1"
        assert result["results"][1]["category"] == "KEGG"

    def test_log_called_during_run(self):
        """The caller-supplied log callback is invoked at each step of the run."""
        responses = ["true", "<ok/>", "<ok/>", "<ok/>"]
        log = MagicMock()
        with patch("toolserver.tools.david_annotation._soap", side_effect=self._soap_iter(responses)):
            _run({"email": "test@test.com", "gene_ids": "1"}, {}, log)
        assert log.call_count >= 4

    def test_default_parameters_applied(self):
        """Omitted optional parameters fall back to DAVID's documented defaults."""
        responses = ["true", "<ok/>", "<ok/>", "<ok/>"]
        with patch("toolserver.tools.david_annotation._soap", side_effect=self._soap_iter(responses)):
            result = _run({"email": "test@test.com", "gene_ids": "1,2"}, {}, MagicMock())
        meta = result["meta"]
        assert meta["id_type"] == "ENTREZ_GENE_ID"
        assert meta["threshold"] == 0.1
        assert meta["count"] == 2
