# AGENTS.md

## Project Overview
- Name: YouTube Channel RAG Pipeline
- Purpose: Index YouTube channel transcripts into ChromaDB and query them via CLI or MCP tools.
- Core flow: discover channel videos -> fetch transcripts -> clean text -> chunk/embed -> upsert into vector store.

## Repository Layout (Current)
- `main.py`: CLI entry point (`index`, `query`, `status`, `channels`, `scheduler`, `mcp`)
- `pipeline.py`: LangGraph orchestration
- `youtube_fetcher.py`: Channel parsing, video discovery (scrapetube), transcript fetch
- `document_cleaner.py`: Transcript cleanup and metadata shaping
- `vector_store.py`: ChromaDB + embedding + retrieval logic
- `state_manager.py`: Persistent channel/video tracking (`data/pipeline_state.json`)
- `scheduler.py`: APScheduler-based periodic indexing
- `mcp_server.py`: MCP tool server exposing query/index/list/stats/delete
- `config.py`: Central config + data/log directory creation

## Runtime Behavior
- Data is persisted under:
  - `data/chroma_db`
  - `data/pipeline_state.json`
  - `logs/pipeline.log`
- Indexing is incremental:
  - Existing video IDs are tracked in state.
  - Re-indexing a channel processes only new videos.
- Vector store writes are idempotent (`upsert` with deterministic chunk IDs).

## Prerequisites
- Python 3.10+ recommended.
- Install dependencies from `requirements.txt`.
- Internet access required for:
  - YouTube video discovery/transcript fetch
  - First-time embedding model download (`all-MiniLM-L6-v2`)

## Standard Commands
- Install:
  - `python -m pip install -r requirements.txt`
- Index channel:
  - `python main.py index "https://www.youtube.com/@marketsbyzerodha" --limit 1`
- Query indexed data:
  - `python main.py query "@marketsbyzerodha" "What is SIP?" -n 3`
- Check state:
  - `python main.py status`
  - `python main.py channels`
- Run MCP server:
  - `python main.py mcp`

## What To Verify After Changes
- CLI help/status runs without import errors.
- Index command completes and reports non-zero chunk ingestion for at least one video.
- Query returns at least one result for indexed channel.
- `data/` and `logs/` files are created inside this repository.

## Known Implementation Notes
- The codebase currently uses a flat module layout (root-level Python files).
- Keep imports consistent with this layout (`from module import ...`) unless intentionally migrating to a package layout.
- Transcript availability varies by video; missing transcripts are expected and logged as non-fatal errors.
