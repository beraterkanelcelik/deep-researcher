# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Chat web application with a Django/LangGraph backend and a React/TypeScript frontend. Features include multi-thread conversations, tool calling (calculator, current time, web search, structured planning), RAG (Retrieval-Augmented Generation) over uploaded documents using pgvector, and a deep research agent with human-in-the-loop (HITL) interrupts, parallel sub-agents via LangGraph `Send()`, and structured output via Pydantic models.

## Development Commands

### Full Stack (Docker Compose)
```bash
docker compose up --build        # Start all services (frontend, backend, postgres, redis)
docker compose down              # Stop all services
docker compose logs -f backend   # Tail backend logs
```

### Backend (standalone, from `backend/`)
```bash
uv sync                                              # Install dependencies
uv run python manage.py migrate                      # Run migrations
uv run uvicorn config.asgi:application --host 0.0.0.0 --port 8000 --reload  # Start dev server
uv run python manage.py makemigrations               # Generate new migrations
```

### Frontend (standalone, from `frontend/`)
```bash
pnpm install     # Install dependencies
pnpm dev         # Start Vite dev server on :5173
pnpm build       # TypeScript check + production build
```

### Testing

**Backend** — pytest + pytest-django + pytest-asyncio. Install test deps with `uv sync --extra test`.

```bash
cd backend
uv run pytest tests/unit/ -v                      # Unit tests (schemas, tools, routing, nodes, stream helpers, serializers, views, RAG chunking)
uv run pytest tests/unit/ -m "not integration" -v  # Unit tests that don't need API keys
uv run pytest tests/integration/ -v                # Integration tests (requires running Postgres + OpenAI/Tavily keys)
uv run pytest tests/evals/ -v --tb=short           # Agent evals (requires OpenAI + Tavily keys, slower)
uv run pytest -v                                   # Full suite

# Two-phase HITL evals (record once, grade cheaply)
uv run pytest tests/evals/test_deep_research_hitl_eval.py --record -v -s          # Phase 1: run real graph, save eval cases (~80s)
uv run pytest tests/evals/test_deep_research_hitl_eval.py -v -s                   # Phase 2: load saved cases, re-grade offline (~5s)
uv run pytest tests/evals/test_deep_research_hitl_eval.py --record -k hitl-01 -v  # Record single task only
```

**Frontend** — Vitest + Testing Library + jsdom.

```bash
cd frontend
pnpm test          # Run all tests once
pnpm test:watch    # Watch mode
```

**Test markers** (backend):
- `@pytest.mark.unit` — Fast, no external services needed (except some need `django_db`)
- `@pytest.mark.integration` — Requires running Postgres + real API keys (OpenAI, Tavily)
- `@pytest.mark.eval` — Agent evaluations with LLM-as-judge grading

**Eval harness** (`tests/evals/`): Based on Anthropic's "Demystifying Evals" article. Uses `EvalTask`/`Trial`/`EvalResult` abstractions with code-based graders (tool selection, content matching, HITL payload structure) and model-based graders (gpt-4.1-mini as judge for response quality, groundedness, coverage). Reports `pass@k` and `pass^k` metrics. Datasets in `tests/evals/datasets/*.json`.

**Two-phase HITL evals** (`tests/evals/eval_case.py`): Decouples expensive graph execution from cheap grading iteration for deep research HITL tests. Phase 1 (`pytest --record`) runs the full LangGraph with real LLM + Tavily calls, saves structured eval cases as JSON in `recorded_cases/`. Phase 2 (default) loads saved cases, reconstructs `HITLTrial` objects, applies code + model graders, and computes pass@k/pass^k — no graph execution needed (~5s vs ~80s). Edit graders in `graders.py` and re-run without `--record` for instant feedback. Uses `run_or_load_eval()` as unified entry point. Eval case JSON follows Anthropic structure: `meta` / `input` / `expected` / `actual` / `grades`. `GenericFakeChatModel` from `langchain_core` can be used to simulate LLM responses for fully offline recording.

## Architecture

### Backend (`backend/`)

Django 5.1 project using `uv` as the package manager. Config is in `config/` (settings, urls, asgi/wsgi). Two Django apps:

**`chat/`** — Core AI conversation engine:
- `schemas.py` — Pydantic v2 models used throughout: `TaskItem`/`TaskPlan` (structured planning), `SearchResult`/`ResearchFinding`/`ResearchReport` (research data), `HITLOption`/`HITLPayload` (typed interrupt payloads with `hitl_type` field: `checkbox`|`yes_no`|`select`|`text`|`review`|`confirm`), `ExplorerInstruction`/`InstructionList` (per-explorer search instructions).
- `graph.py` — Main LangGraph `StateGraph(AgentState)` with eight nodes: `retrieve`, `agent`, `tools`, `prepare_research`, `deep_research` (compiled subgraph added as native node), `process_research_result`, `save_confirm`, `save_to_db`. `AgentState` extends `MessagesState` with `research_reports` (accumulating list via `operator.add`), `current_plan`, `pending_save`, `research_request`, plus shared subgraph keys `topic`, `depth`, `report`. `build_graph()` takes no parameters — parent checkpointer auto-propagates to the subgraph node. Uses `AsyncPostgresSaver` for conversation checkpointing.
- `nodes.py` — Node implementations: `retrieve_node` (RAG context injection), `agent_node` (calls LLM with `ALL_TOOLS`), `should_continue` (4-way routing: `tools`|`deep_research`|`save_confirm`|`end` — detects sentinel tool calls by name), `prepare_research_node` (extracts `topic`/`depth` from agent's `deep_research` tool call), `process_research_result_node` (formats subgraph `report` output as `ToolMessage`, appends to `research_reports`), `should_run_research` (routes to subgraph if topic set, else back to agent on error), `save_confirm_node` (HITL interrupt #3 for save confirmation), `save_to_db_node` (persists report), `should_continue_after_save_confirm`. Model is configurable via `MODEL_CONFIG` dict (default: `gpt-4.1-mini`, also supports `gpt-5-nano` with variable reasoning effort).
- `tools.py` — LangChain `@tool` functions: `get_current_time`, `calculator`, `tavily_search` (Tavily web search), `create_plan` (calls `llm.with_structured_output(TaskPlan)`), `deep_research` (sentinel — routes to subgraph), `save_report` (sentinel — routes to save flow). Exports `SIMPLE_TOOLS` (executed by ToolNode), `ALL_TOOLS` (bound to LLM including sentinels).
- `research_graph.py` — Deep research subgraph with `TypedDict` states: `DeepResearchState` (full internal state), `ExplorerState` (per-explorer), `ResearchInput` (input schema: `topic`, `depth`), `ResearchOutput` (output schema: `report`). Uses `StateGraph(DeepResearchState, input=ResearchInput, output=ResearchOutput)` for clean parent/subgraph interface. Five nodes: `clarify_node` (LLM suggests sub-topics, HITL interrupt #1 with checkbox selection), `orchestrate_node` (`llm.with_structured_output(InstructionList)` generates N search instructions), `explorer_node` (runs `TavilySearchResults`, parallelized via `Send()`), `synthesize_node` (`llm.with_structured_output(ResearchReport)`), `review_node` (HITL interrupt #2 with approve/edit/redo). Conditional redo loop from review back to orchestrate. Exported via `build_research_subgraph()`.
- `views_stream.py` — Async SSE streaming. Shared `_stream_graph()` async generator handles both `stream_run` (new messages) and `resume_run` (HITL resume via `Command(resume=value)`). `GRAPH_NODES` set includes all 8 main graph nodes (`retrieve`, `agent`, `tools`, `prepare_research`, `deep_research`, `process_research_result`, `save_confirm`, `save_to_db`); anything not in this set is treated as a subgraph node. Emits SSE events: `messages/partial`, `messages/complete`, `metadata`, `node/status` (with `subgraph: true` flag for subgraph nodes), `interrupt` (HITL payload), `error`, `end`. After stream, checks `aget_state()` for interrupts and persists `pending_save` reports to Django `ResearchReport` model.
- `views.py` — REST endpoints for thread CRUD and thread state. `thread_state` fetches interrupt info from graph checkpointer via `aget_state` and returns it in the `tasks` array.
- `models.py` — Django models: `Thread`, `Message`, `ResearchReport` (uuid, thread FK, title, summary, key_findings JSONField, sources JSONField, tags JSONField, methodology, created_at).
- `rag/` — RAG pipeline, fully wired end-to-end:
  - `embeddings.py` — `generate_embedding(text)` and `generate_embeddings(texts)` using OpenAI `text-embedding-3-small` (1536 dims).
  - `ingest.py` — `ingest_document(file, filename)` orchestrates: `extract_text()` (PDF/DOCX/TXT dispatch) → `chunk_text()` (tiktoken cl100k_base, 500 tokens/50 overlap) → `generate_embeddings()` → `Embedding.objects.bulk_create()`. Returns `(Document, chunk_count)`.
  - `retriever.py` — `retrieve_documents(query, top_k=3)` generates query embedding, then `Embedding.objects.annotate(distance=CosineDistance(...)).order_by("distance")[:top_k]`. Returns list of `{content, filename, distance, document_id}`.
  - **RAG wiring**: Upload (`POST /api/documents/upload`) → `ingest_document()` stores chunks + vectors in `Embedding` table (pgvector `VectorField(1536)`). At query time, graph runs `START → retrieve_node → agent`. `retrieve_node` calls `get_rag_context()` → `retrieve_documents()` → injects top-3 results as `SystemMessage` with `[Source: filename]` markers prepended to the system prompt. Migration `0001_initial` creates `CREATE EXTENSION IF NOT EXISTS vector`.

**`documents/`** — Document upload and management:
- Multipart file upload at `POST /api/documents/upload` triggers the full RAG ingest pipeline.
- Models: `Document` (stores full text), `Embedding` (stores chunks with 1536-dim `VectorField`).
- Supported file types: PDF (pypdf), DOCX (python-docx), and text files (.txt, .md, .csv, .json, .py, .js, .ts, .html, .css). Max upload size: 10MB.

### Frontend (`frontend/`)

React 19 + TypeScript (strict mode) + Vite 6 + Tailwind CSS + pnpm. Uses `@assistant-ui/react` with `@assistant-ui/react-langgraph` for the chat runtime.

- `src/components/MyAssistant.tsx` — Wires `useLangGraphRuntime` to the backend API (create, load, stream). Connects the node status store and interrupt store to the SSE stream via `onNodeStatus` and `onInterrupt` callbacks. Provides `handleResume` callback that calls `resumeRun` to continue past HITL interrupts.
- `src/components/assistant-ui/thread.tsx` — Chat UI built with `assistant-ui` primitives (ThreadPrimitive, MessagePrimitive, ComposerPrimitive). Renders `GraphNodeIndicator` as a fixed right-side tab. Accepts `onResume` prop. When `activeInterrupt` is set, renders `HITLWidget` after messages and disables the Composer (user must respond to HITL first).
- `src/components/HITLWidgets.tsx` — HITL widget components matching dark theme (zinc/violet/emerald): `HITLWidget` (dispatcher by `hitl_type`), `CheckboxWidget` (multi-select with submit), `YesNoWidget` (two-button), `SelectWidget` (radio selection), `TextWidget` (textarea with submit), `ReviewWidget` (report preview + approve/edit/redo), `ConfirmWidget` (report preview + save/cancel), `ReportPreview` (shared sub-component for rendering ResearchReport data with findings, sources, tags).
- `src/components/GraphNodeIndicator.tsx` — Fixed right-side vertical tab showing real-time LangGraph pipeline status. Shows core nodes (Retrieve, Agent, Tools) always; shows additional nodes (Research, Confirm, Save) only when active/completed. When `isSubgraphRunning`, renders a "Deep Research" section below with subgraph nodes (Clarify -> Plan -> Explore -> Synthesize -> Review). Tracks run counts for agent and explorer nodes.
- `src/lib/chatApi.ts` — All backend API calls. Shared `parseSSEStream()` helper handles SSE parsing for both `sendMessage` and `resumeRun`. Side-channel events (`node/status` with subgraph flag, `interrupt`) dispatched via callbacks, not yielded to runtime. `resumeRun` POSTs to `/api/threads/<id>/runs/resume` with `resume_value`.
- `src/lib/store.ts` — Zustand stores: `useAppStore` for sidebar/thread state and model selection; `useNodeStatusStore` for real-time graph node status tracking. `NodeName` includes 8 main graph nodes (including `prepare_research` and `process_research_result`), `SubgraphNodeName` includes 5 deep research nodes. Store tracks `subgraphNodes`, `isSubgraphRunning`, `activeInterrupt: HITLPayload | null`. `handleNodeEvent` routes to main or subgraph node arrays based on `isSubgraph` flag. TypeScript types for `HITLPayload`, `HITLOption`, `ResearchReport`, `ResearchFinding` mirror backend Pydantic schemas.
- Path alias: `@` maps to `src/`.
- Vite proxies `/api` requests to `http://localhost:8000` in dev.
- Dark theme throughout; uses Lucide for icons.

### Graph Architecture

```
MAIN GRAPH (AgentState)
  START -> retrieve -> agent -> [routing via should_continue]
                                 |-> tools (simple: calculator, time, tavily, plan) -> agent
                                 |-> prepare_research -> [should_run_research]
                                 |     |-> deep_research (SUBGRAPH, native node) -> process_research_result -> agent
                                 |     |     |-> clarify (HITL #1: checkbox topic selection)
                                 |     |     |-> orchestrate (structured output -> N instructions)
                                 |     |     |-> explorer x N (parallel Tavily via Send())
                                 |     |     |-> synthesize (structured output -> ResearchReport)
                                 |     |     |-> review (HITL #2: approve/edit/redo)
                                 |     |-> agent (on error, skip subgraph)
                                 |-> save_confirm (HITL #3: confirm save) -> save_to_db -> agent
                                 |-> END
```

### Data Flow

1. Frontend sends user message via SSE POST to `/api/threads/<id>/runs/stream`
2. Backend runs LangGraph: retrieve (RAG) -> agent (LLM) -> routing based on tool calls
3. Simple tools: executed by `ToolNode`, results fed back to agent
4. Deep research: sentinel `deep_research` tool call detected by `should_continue` -> `prepare_research` (extracts args) -> compiled subgraph as native node (interrupts propagate to parent) -> `process_research_result` (formats ToolMessage) -> agent. 2 HITL interrupts in subgraph (clarify topics, review report), 1 in main graph (confirm save)
5. Streaming tokens sent as SSE events (`messages/partial` for chunks, `messages/complete` for final)
6. `node/status` SSE events emitted for both main graph and subgraph nodes (subgraph events include `subgraph: true` flag) -> parsed in `chatApi.ts` -> `onNodeStatus` callback updates `useNodeStatusStore` -> `GraphNodeIndicator` renders pipeline status
7. `interrupt` SSE events emitted when graph pauses at HITL points -> `onInterrupt` callback sets `activeInterrupt` in store -> `HITLWidget` renders appropriate UI widget -> user responds -> `handleResume` calls `resumeRun` with `Command(resume=value)` -> graph continues
8. Frontend `useLangGraphRuntime` consumes the SSE stream via async generators in `chatApi.ts` (side-channel events consumed by callbacks, not yielded)
9. LangGraph conversation state checkpointed to PostgreSQL via `AsyncPostgresSaver`
10. Research reports accumulated in `AgentState.research_reports` (multiple per conversation) and persisted to `ResearchReport` Django model on save confirmation

### Infrastructure

- **PostgreSQL** (pgvector/pgvector:pg16) — Django models + LangGraph checkpointer + pgvector embeddings
- **Redis** (redis:7-alpine) — Configured for channels/caching (REDIS_URL in settings)
- **LangSmith** — Tracing enabled via env vars (project: "travel-app")

### Key Environment Variables

Backend reads from `config/settings.py` with these defaults:
- `DATABASE_URL` (default: `postgresql://chat:chat@localhost:5432/chatdb`)
- `REDIS_URL` (default: `redis://localhost:6379/0`)
- `OPENAI_API_KEY` — required for LLM and embeddings
- `TAVILY_API_KEY` — required for web search (tavily_search tool and deep research explorers)
- `DJANGO_DEBUG`, `DJANGO_SECRET_KEY`, `ALLOWED_HOSTS`, `CORS_ALLOWED_ORIGINS`

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Health check |
| GET/POST | `/api/threads/` | List/create threads |
| GET/DELETE | `/api/threads/<id>/` | Get/delete thread |
| GET | `/api/threads/<id>/state` | Thread state with interrupt info |
| POST | `/api/threads/<id>/runs/stream` | SSE streaming chat |
| POST | `/api/threads/<id>/runs/resume` | SSE resume after HITL interrupt |
| GET/POST | `/api/documents/upload` | List/upload documents |
| DELETE | `/api/documents/<id>/` | Delete document |

### Test Structure

```
backend/tests/
  conftest.py                          # Shared fixtures: sample_thread, sample_messages, async_client, llm, compiled_graph, sample_document
  unit/
    test_schemas.py                    # Pydantic model validation (TaskItem, TaskPlan, SearchResult, ResearchReport, HITLPayload, etc.)
    test_tools.py                      # calculator, get_current_time, tavily_search, create_plan, sentinel tools, exports
    test_routing.py                    # should_continue (6 routes), should_run_research, should_continue_review, should_continue_after_save_confirm
    test_nodes.py                      # retrieve_node (empty DB + with docs/embeddings), agent_node, prepare_research_node, process_research_result_node, save_to_db_node
    test_stream_helpers.py             # parse_input_messages, format_ai_chunk, format_tool_message, GRAPH_NODES constant
    test_serializers.py                # ThreadSerializer, MessageSerializer, ThreadDetailSerializer
    test_views.py                      # REST endpoints: thread CRUD, thread state
    test_rag_chunking.py               # chunk_text (basic, overlap, small, boundary), extract_text_from_txt, extract_text dispatcher
    test_rag_embeddings.py             # generate_embedding dimension/determinism, generate_embeddings batch, cosine similarity
    test_rag_retriever.py              # retrieve_documents: empty DB, with results, relevance order, top_k
    test_rag_ingest.py                 # Full ingest pipeline (text file, small file)
  integration/
    test_graph_execution.py            # Full graph runs with real LLM (simple conversation, calculator, time, search, plan, multi-turn, state)
    test_sse_streaming.py              # SSE event format: metadata, end, node/status, messages partial→complete
    test_hitl_flow.py                  # Interrupt + resume cycles (save_confirm, clarify, graph build verification)
    test_rag_pipeline.py               # Upload → chunk → embed → retrieve, record creation, RAG-improved answers
    test_research_subgraph.py          # Subgraph build, nodes, orchestrate, explorer, synthesize
  evals/
    conftest.py                        # Eval fixtures: eval_graph, dataset loaders, --record flag + record_mode fixture
    harness.py                         # EvalTask, Trial, EvalResult, run_eval_task() + HITL variants (HITLEvalTask, HITLTrial, HITLEvalResult, run_hitl_eval_task)
    eval_case.py                       # Two-phase eval: save_eval_case, record_eval_cases, load_eval_cases_for_task, reconstruct_trial, grade_recorded_cases, run_or_load_eval
    transcript.py                      # Trial transcript + eval report persistence (save_trial_transcript, save_eval_report)
    graders.py                         # Code-based (tool_selection, content_contains, no_hallucinated_tools, hitl_payload_structure, report_structure, hitl_flow_completeness, hitl_payload_all_valid, research_report_present) + model-based (response_quality, groundedness, coverage, tone, deep_research_quality)
    metrics.py                         # pass@k (unbiased estimator), pass^k (consistency)
    datasets/
      tool_selection.json              # 15 tasks: math, time, search, planning, no-tool
      conversation_quality.json        # 10 tasks with rubrics
      research_quality.json            # 8 topics with rubrics
      rag_accuracy.json                # 8 tasks with inline documents + expected answers
      deep_research_hitl.json          # 6 HITL tasks: happy path, select-all, redo, edit, cancel, quick depth
    recorded_cases/                    # Saved eval case JSONs (gitignored, persisted locally)
    transcripts/                       # Trial transcripts + eval reports (gitignored)
    test_tool_selection_eval.py        # Agent selects correct tools (k=3, code grader)
    test_conversation_eval.py          # Response quality + tone (model grader)
    test_research_eval.py              # Report structure, groundedness, coverage
    test_rag_eval.py                   # RAG retrieval accuracy + answer quality
    test_hitl_eval.py                  # HITL payload correctness at all 3 interrupt points (uses mini StateGraphs with MemorySaver)
    test_deep_research_hitl_eval.py    # Two-phase HITL eval: full interrupt/resume cycles (--record to save, default loads + grades offline)
    test_end_to_end_eval.py            # Full user journeys (greeting→math→search, RAG upload→query, plan, multi-tool)

frontend/src/__tests__/
  setup.ts                             # Vitest setup (jest-dom matchers)
  lib/
    store.test.ts                      # useAppStore (sidebar, thread, model) + useNodeStatusStore (node events, subgraph, interrupt, reset)
    chatApi.test.ts                    # API functions (health, createThread, listThreads, deleteThread) + SSE parsing (metadata, partial, node/status callback, interrupt callback, subgraph flag)
  components/
    HITLWidgets.test.tsx               # All 6 widget types: checkbox (select/deselect/submit/disabled), yes_no, select, text, review (approve/redo/edit), confirm (save/cancel)
    GraphNodeIndicator.test.tsx        # Node rendering, active/completed styling, subgraph section visibility, run counts, opacity
    App.test.tsx                       # Loading state, health check, model selector, MyAssistant render
```
