# bgts-case

Network operations Root Cause Analysis (RCA) agent built on LangGraph. Given a
Turkish-language network incident description, the agent produces an
evidence-grounded RCA by combining:

- **Hybrid retrieval** over a Turkish network-ops knowledge base (10 KB PDFs:
  BGP, Firewall, VPN, DNS/DHCP, SD-WAN, STP, IPAM, NOC escalation, Wireless,
  Incident archive) indexed in Qdrant.
- **An ITSM ticket MCP server** exposing read-only tools over a Postgres
  tickets table (search, aggregate, related-tickets, full detail).
- **A primary chat model** (Fireworks DeepSeek-V4-Pro by default, or a local
  OpenAI-compatible endpoint via Docker Model Runner) with a Fireworks
  fallback model wired through `ModelFallbackMiddleware`.
- **A Ragas + MLflow eval pipeline** that scores the agent's final answer on
  `ResponseRelevancy` and `FactualCorrectness` and logs per-case and sliced
  metrics.

## Architecture

```
                ┌─────────────────────────────┐
                │   LangGraph agent (graph)   │
                │  src/bgts_case/agent/main   │
                └──────┬──────────────┬───────┘
                       │              │
            tools.py   │              │   MCP (streamable HTTP)
        retrieve_kb()  │              │
                       ▼              ▼
              ┌─────────────┐   ┌──────────────────┐
              │   Qdrant    │   │ FastMCP server   │
              │ (hybrid:    │   │ src/.../mcp_server│
              │  dense+BM25)│   │ get_ticket /     │
              └─────┬───────┘   │ search_tickets / │
                    │           │ aggregate /      │
        embed_query │           │ related_tickets  │
       (Fireworks)  │           └────────┬─────────┘
                    │                    │ SQLAlchemy
                    │                    ▼
                    │            ┌──────────────┐
                    │            │  Postgres    │
                    │            │  (tickets)   │
                    │            └──────────────┘
                    │
        Indexed by RAG pipeline
        (orchestrator → chunk → index)
```

## Project structure

```
.
├── alembic/                       Schema migrations for the tickets table
├── docker/postgres-init/          One-shot init SQL (e.g. CREATE DATABASE mlflow)
├── docker-compose.yaml            postgres + mlflow + qdrant
├── langgraph.json                 LangGraph dev-server entry (graphs.agent)
├── pyproject.toml                 Console scripts: seed-tickets,
│                                  run-ticket-mcp-server, run-rag-pipeline, run-eval
├── static/
│   ├── knowledge_base/            10 KB PDFs (Turkish) consumed by the RAG pipeline
│   ├── itsm_tickets.json          Seed data for the tickets table
│   └── eval_cases.json            20-case Ragas eval dataset (Ragas 0.2 schema)
└── src/bgts_case/
    ├── secret.py                  pydantic-settings; reads .env
    ├── eval.py                    Ragas + MLflow eval driver (`run-eval`)
    ├── agent/
    │   ├── main.py                Builds the LangGraph agent (`graph`)
    │   ├── interfaces.py          make_chat_model / make_embeddings /
    │   │                          make_qdrant_client / make_openai_client
    │   ├── retriever.py           Qdrant hybrid search (dense + BM25, RRF-fused)
    │   ├── tools.py               `retrieve_knowledge_base` tool (RAG)
    │   ├── alerts.py              Mock Slack alert helper
    │   ├── utils.py
    │   └── rag_pipeline/
    │       ├── orchestrator.py    `run-rag-pipeline` entry point
    │       ├── step1_chunk.py     PDF → Documents (pdfplumber + table extraction)
    │       ├── step2_qdrant_index.py  Documents → Qdrant hybrid collection
    │       └── patterns.py        Regex/heuristic helpers for chunking
    └── mcp_server/
        ├── server.py              FastMCP tools (`run-ticket-mcp-server`)
        ├── crud.py                SQLAlchemy queries
        ├── models.py              ORM model + enums + DTOs
        ├── session.py             Engine / SessionLocal
        └── seed.py                `seed-tickets` (loads itsm_tickets.json)
```

## Configuration

All configuration is loaded by `src/bgts_case/secret.py` (pydantic-settings)
from a `.env` file at the repo root. Required / notable settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `fireworks_api_key` | *(required)* | Fireworks key for primary/fallback/eval/embeddings |
| `qdrant_url` | `http://localhost:6333` | Qdrant endpoint |
| `qdrant_api_key` | *(none)* | Optional, for managed Qdrant |
| `database_url` | `postgresql+psycopg://postgres:postgres@localhost:5432/postgres` | Tickets DB |
| `mcp_server_url` | `http://localhost:8765/mcp/` | URL the agent uses to reach the MCP server |
| `mlflow_tracking_uri` | `http://localhost:5000` | MLflow tracking server |
| `run_as_a_local_model` | `True` | If true, route the **primary** chat model to a local OpenAI-compatible endpoint |
| `local_model_base_url` | `http://localhost:12434/engines/v1` | Local model endpoint (Docker Model Runner) |
| `local_model_name` | `hf.co/mlx-community/Qwen2.5-7B-Instruct-4bit` | Local model id |

Fallback chat, eval LLM, and embeddings always use Fireworks regardless of
`run_as_a_local_model` — embeddings in particular must stay on the same model
that indexed the corpus.

## Quick start

```bash
# 1. Install deps (Python 3.14 via uv)
make install

# 2. Bring up Postgres + MLflow + Qdrant
make up

# 3. Apply migrations & seed tickets
make migrate
make seed

# 4. Index the knowledge base into Qdrant (one-shot)
make rag-pipeline

# 5. Start the ITSM MCP server (separate terminal — agent depends on it)
make mcp-server

# 6. Run the LangGraph dev UI
make dev
```

Console scripts (also runnable via `uv run`):

- `seed-tickets` — load `static/itsm_tickets.json` into Postgres.
- `run-ticket-mcp-server` — start the FastMCP server over HTTP at `mcp_server_url`.
- `run-rag-pipeline` — chunk + index every PDF under `static/knowledge_base` into Qdrant.
- `run-eval` — run the Ragas + MLflow eval over `static/eval_cases.json`.

## MCP tools (ITSM tickets)

Exposed by `src/bgts_case/mcp_server/server.py` (FastMCP, streamable HTTP):

- `get_ticket(ticket_id)` — full detail for a single ticket.
- `get_tickets(ticket_ids)` — batch full detail (max 20).
- `search_tickets(...)` — filtered list with truncated summaries; supports
  category / priority / status / sub_category / opened_after-before /
  affected_system / assigned_team / Postgres full-text `text_query` /
  `error_message_contains` ILIKE over the `error_messages` JSONB array /
  `has_resolution` / ordering.
- `aggregate_tickets(group_by, ...)` — group counts, avg resolution hours,
  still-open counts, most-recent ticket per bucket.
- `get_related_tickets(ticket_id, ...)` — four channels: explicit_links,
  same_affected_system, same_error_pattern, same_category_recent.
  Overlap is intentionally not deduplicated (overlap = stronger signal).
- `list_enum_values()` — valid enum values for filter parameters.

## RAG pipeline

`make rag-pipeline` runs `orchestrator.run_pipeline` over every PDF in
`static/knowledge_base`:

1. **Chunk** (`step1_chunk.py`) — pdfplumber-based extraction with separate
   handling for prose (`text_chunk`) and tables (`table_chunk`); preserves
   `doc_title`, `page_number`, `source_pdf_name`.
2. **Index** (`step2_qdrant_index.py`) — upserts into the `bgts_kb` Qdrant
   collection with both a dense vector (Fireworks `qwen3-embedding-8b`, 1024d)
   and a BM25 sparse vector (`Qdrant/bm25`).

Retrieval (`retriever.py`) issues two prefetches (dense + sparse) and fuses
them server-side with RRF. Optional metadata filters (`source_pdf_name`,
`chunk_type`, `page_number`, `doc_title`) are ANDed and applied to both
prefetches so the candidate pools stay aligned.

## Eval

```bash
make eval
```

Logs to the experiment `network-ops-agent-eval` on the configured MLflow
tracking server. For each run:

- **Params**: dataset name, schema version, total cases, primary/eval/embedding
  model ids.
- **Per-case**: `ragas_per_case.json` table with `case_id`, metric scores,
  `difficulty`, `question_type`, `source_kbs`, `related_tickets`.
- **Slices**: mean metrics by `difficulty` and `question_type`, plus flat
  per-slice metrics so they show up as columns in the Compare Runs
  parallel-coordinates view.
- **Traces**: `mlflow.langchain.autolog()` is enabled so each agent `ainvoke`
  is captured as an MLflow Trace for debugging poor-scoring cases.

Eval cases are run sequentially (not `asyncio.gather`) to stay under Fireworks
rate limits, and the Ragas `evaluate()` call runs AFTER the async loop closes
to avoid nesting conflicts with `nest_asyncio` (which LangGraph pulls in
transitively).
