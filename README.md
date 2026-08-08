# omnibioai-toolserver

A standalone **HTTP ToolServer** for the **OmniBioAI** ecosystem.

This service implements the REST contract expected by
`omnibioai-tes (Tool Execution Service, legacy package name: `omnibioai-tes`) `HttpToolServerAdapter`, enabling **secure, validated, and reproducible execution of REST-backed bioinformatics tools** (e.g. Enrichr, annotation services, external APIs).

It is designed to run **independently** and be registered as a remote execution server in OmniBioAI TES.

---

## Implemented API Contract

The ToolServer exposes the following endpoints:

* `GET  /capabilities`
  Advertise supported tools, engines, resources, and runtime policies.

* `POST /validate`
  Validate tool inputs and resource requests.

* `POST /runs`
  Submit a tool execution request.

* `GET  /runs/{id}`
  Retrieve run state (`QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`).

* `GET  /runs/{id}/logs`
  Retrieve execution logs.

* `GET  /runs/{id}/results`
  Retrieve structured tool results once the run is completed.

* `GET  /health`
  Service health check — returns `{"ok": true, "service": "omnibioai-toolserver"}`

This contract matches the expectations of
**`omnibioai-tes` → `HttpToolServerAdapter`**.

---

## Current Capabilities

* Engine: `http_toolserver`
* Tools:

  * `enrichr_pathway` — Pathway enrichment via Enrichr (REST, multipart-safe)
  * `david_annotation` — Functional annotation via DAVID (REST)
* Execution model:

  * Stateless REST calls
  * Structured validation
  * Run lifecycle tracking
* Designed for:

  * OmniBioAI agents
  * TES-controlled execution
  * Future multi-tool expansion (OMIM, GO, UniProt, etc.)

---

## Running

### Via OmniBioAI Studio (recommended)

```bash
cd ~/Desktop/machine/omnibioai-studio
docker compose up -d toolserver
```

Access: `http://localhost:9090`
Via nginx: `http://localhost/_svc/toolserver`

### Standalone (development)

```bash
pip install -r requirements.txt
uvicorn toolserver_app:create_app --factory \
  --host 0.0.0.0 --port 9090 --reload
```

### Verify

```bash
curl http://localhost:9090/health
# {"ok": true, "service": "omnibioai-toolserver"}

curl http://localhost:9090/capabilities | python -m json.tool
```

---

## Integration with OmniBioAI TES

Register this service as a server in `omnibioai-tes`:

```yaml
- server_id: enrichment_remote
  display_name: Enrichment ToolServer
  adapter_type: http_toolserver
  config:
    base_url: "http://127.0.0.1:9090"
```

Then submit runs through TES:

```bash
POST /api/runs/submit
```

The ToolServer is **never called directly by the browser or LLM**—all execution is mediated by TES.

---

## Adding a New REST-Backed Tool

1. Create a new handler:

```
toolserver/tools/<new_tool>.py
```

Implement:

```python
def _validate(inputs, resources) -> {
  "ok": bool,
  "errors": [],
  "warnings": []
}

def _run(inputs, resources, log) -> Dict[str, Any]
```

2. Register the tool in:

```
toolserver/tools/__init__.py
```

```python
registry.register(ToolHandler(...))
```

3. Restart the server.

The tool will be **automatically advertised** via `/capabilities`.

No changes are required in TES beyond refreshing server capabilities.

---

## Testing

```bash
cd ~/Desktop/machine/omnibioai-toolserver
pytest tests/ -v --cov=.

# 97% coverage (verified 2026-08-07; every toolserver/ module itself is
# 100%, setup.py is the only file dragging the --cov=. total down)
```

---

## Design Principles

* **LLMs never execute tools directly**
* **All execution is validated and audited**
* **Strict separation** between:

  * intent (agents / UI)
  * orchestration (TES)
  * execution (ToolServer)
* REST-first, container-friendly, and infrastructure-agnostic

---

## Related Services

| Service | Role |
|---------|------|
| `omnibioai-tes` | Primary consumer — routes HTTP tool requests to ToolServer |
| `omnibioai-api-gateway` | JWT enforcement on all ToolServer requests |
| `omnibioai-control-center` | Health monitoring (toolserver:9090) |
| `omnibioai-studio` | Manages ToolServer container lifecycle |

---

## Status

| Feature | Status |
|---------|--------|
| Enrichr pathway enrichment | ✓ Stable |
| TES HttpToolServerAdapter integration | ✓ Stable |
| Health endpoint | ✓ Stable |
| REST tool lifecycle (submit/poll/results) | ✓ Stable |
| Test coverage | ✓ 97% (100% within `toolserver/` itself) |
| Docker Compose deployment | ✓ Stable |

