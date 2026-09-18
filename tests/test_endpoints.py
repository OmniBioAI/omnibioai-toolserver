"""
Tests for ToolServer's core HTTP endpoints: health, capabilities,
validation, and the create-run-to-completion happy path.

Developer:
    Manish Kumar <manish@omnibioai.org>
"""


def test_health(client):
    """Return 200 with ok=True and a service identifier."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "service" in body


def test_capabilities(client):
    """List registered engines and tools, including enrichr_pathway."""
    r = client.get("/capabilities")
    assert r.status_code == 200
    caps = r.json()
    assert "engines" in caps
    assert "tools" in caps
    tool_ids = {t["tool_id"] for t in caps["tools"]}
    assert "enrichr_pathway" in tool_ids


def test_validate_unknown_tool(client):
    """Report UNKNOWN_TOOL as a validation error, not an HTTP failure, for an unregistered tool_id."""
    r = client.post("/validate", json={"tool_id": "nope", "inputs": {}, "resources": {}})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["errors"][0]["code"] == "UNKNOWN_TOOL"


def test_validate_enrichr_pathway_requires_genes(client):
    """Flag the missing required "genes" field when validating enrichr_pathway inputs."""
    r = client.post("/validate", json={"tool_id": "enrichr_pathway", "inputs": {}, "resources": {}})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert any(e["field"] == "genes" for e in body["errors"])


def test_create_run_validation_fails(client):
    """Reject run creation with 400 VALIDATION_FAILED when required inputs are missing."""
    r = client.post("/runs", json={"tool_id": "enrichr_pathway", "inputs": {}, "resources": {}})
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "VALIDATION_FAILED"


def test_create_run_and_poll_to_complete(client):
    """Drive a valid run from creation through COMPLETED state, with logs and results populated along the way."""
    req = {
        "tool_id": "enrichr_pathway",
        "inputs": {"genes": ["TP53", "BRCA1"], "top_n": 10},
        "resources": {},
    }
    r = client.post("/runs", json=req)
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    assert run_id.startswith("ts_")

    # Poll until COMPLETED (short loop; fake run sleeps 0.01s)
    state = None
    for _ in range(50):
        rr = client.get(f"/runs/{run_id}")
        assert rr.status_code == 200
        state = rr.json()["state"]
        if state == "COMPLETED":
            break
    assert state == "COMPLETED"

    logs = client.get(f"/runs/{run_id}/logs").json()["logs"]
    assert "Starting tool: enrichr_pathway" in logs
    assert "FAKE RUN called" in logs
    assert "Completed" in logs

    res = client.get(f"/runs/{run_id}/results")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert "results" in body
