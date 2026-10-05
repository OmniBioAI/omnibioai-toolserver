"""
OmniBioAI toolserver.tools.david_annotation.

Purpose:
    Defines _validate, _soap, _parse_chart_records and _run for toolserver.tools.david_annotation.

Author:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations
import re
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Optional
import httpx


DAVID_WS = "https://davidbioinformatics.nih.gov/webservice/services/DAVIDWebService"
NS = "http://service.session.sample"
DAVID_NS = "http://DAVID/xsd"
XSD_NS = "http://service.session.sample/xsd"


def _validate(inputs: Dict[str, Any], resources: Dict[str, Any]) -> Dict[str, Any]:
    errors = []
    if not inputs.get("email"):
        errors.append({"field": "email", "message": "email is required (registered DAVID email)"})
    gene_ids = inputs.get("gene_ids")
    if not gene_ids:
        errors.append({"field": "gene_ids", "message": "gene_ids is required"})
    return {"ok": len(errors) == 0, "errors": errors, "warnings": []}


def _soap(client: httpx.Client, action: str, body: str) -> str:
    envelope = f'''<?xml version="1.0" encoding="UTF-8"?>
<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/"
                  xmlns:ser="http://service.session.sample">
  <soapenv:Body>{body}</soapenv:Body>
</soapenv:Envelope>'''
    resp = client.post(
        DAVID_WS,
        content=envelope,
        headers={"Content-Type": "text/xml", "SOAPAction": f"urn:{action}"},
    )
    resp.raise_for_status()
    return resp.text


def _parse_chart_records(xml_text: str) -> List[Dict[str, Any]]:
    """Parse DAVID getChartReport SOAP response into list of dicts."""
    results = []
    try:
        root = ET.fromstring(xml_text)
        # Find all return elements
        for elem in root.iter():
            if elem.tag.endswith("}return") or elem.tag == "return":
                record: Dict[str, Any] = {}
                for child in elem:
                    tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    record[tag] = child.text
                if record:
                    results.append(record)
    except Exception as e:
        pass
    return results


def _run(
    inputs: Dict[str, Any],
    resources: Dict[str, Any],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    email      = str(inputs["email"]).strip()
    gene_ids   = str(inputs["gene_ids"]).strip()
    id_type    = str(inputs.get("id_type", "ENTREZ_GENE_ID")).strip()
    categories = str(inputs.get("categories", "GOTERM_BP_DIRECT,KEGG_PATHWAY")).strip()
    threshold  = float(inputs.get("threshold", 0.1))
    count      = int(inputs.get("count", 2))
    list_name  = str(inputs.get("list_name", "omni_gene_list")).strip()
    timeout    = float(inputs.get("_timeout_s", 60.0))

    # Use httpx with cookie jar to maintain DAVID session
    with httpx.Client(timeout=timeout, cookies={}) as client:

        # 1. Authenticate
        log("DAVID: authenticating...")
        r = _soap(client, "authenticate",
            f"<ser:authenticate><ser:abbr>{email}</ser:abbr></ser:authenticate>")
        if "true" not in r.lower():
            raise RuntimeError(f"DAVID authentication failed for {email}. Is email registered?")
        log("DAVID: authenticated ✅")

        # 2. addList
        log(f"DAVID: adding gene list ({id_type})...")
        r = _soap(client, "addList", f"""<ser:addList>
  <ser:inputIds>{gene_ids}</ser:inputIds>
  <ser:idType>{id_type}</ser:idType>
  <ser:listName>{list_name}</ser:listName>
  <ser:listType>0</ser:listType>
</ser:addList>""")
        if "Fault" in r:
            raise RuntimeError(f"DAVID addList failed: {r[:300]}")
        log("DAVID: gene list added ✅")

        # 3. setCategories
        log(f"DAVID: setting categories: {categories}")
        _soap(client, "setCategories",
            f"<ser:setCategories><ser:categories>{categories}</ser:categories></ser:setCategories>")

        # 4. getChartReport
        log("DAVID: getting chart report...")
        r = _soap(client, "getChartReport", f"""<ser:getChartReport>
  <ser:threshold>{threshold}</ser:threshold>
  <ser:count>{count}</ser:count>
</ser:getChartReport>""")

        records = _parse_chart_records(r)
        log(f"DAVID: got {len(records)} enriched terms ✅")

    # Format results
    results = []
    for rec in records:
        results.append({
            "category":        rec.get("categoryName"),
            "term":            rec.get("termName"),
            "genes":           rec.get("geneIds"),
            "list_hits":       rec.get("listHits"),
            "fold_enrichment": rec.get("foldEnrichment"),
            "p_value":         rec.get("ease"),
            "fisher":          rec.get("fisher"),
            "benjamini":       rec.get("benjamini"),
            "bonferroni":      rec.get("bonferroni"),
        })

    return {
        "ok":        True,
        "n_terms":   len(results),
        "results":   results,
        "meta": {
            "email":      email,
            "id_type":    id_type,
            "categories": categories,
            "threshold":  threshold,
            "count":      count,
        },
    }
