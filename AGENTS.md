# AGENTS.md

## Project Overview
- **Name:** YouTube Channel RAG Pipeline + Persona Agent
- **Purpose:** Index YouTube channel transcripts into ChromaDB, then query them via CLI or power a conversational AI agent that talks like the creator.
- **Core flow:** discover videos → fetch transcripts → clean text → chunk/embed → upsert into ChromaDB → build creator persona → serve via MCP or chat agent.

## Repository Layout
- `main.py` — CLI entry point (`index`, `query`, `status`, `channels`, `scheduler`, `mcp`)
- `pipeline.py` — LangGraph orchestration of the three pipeline nodes
- `youtube_fetcher.py` — Channel URL parsing, video discovery (scrapetube), transcript fetch
- `document_cleaner.py` — Transcript cleanup and metadata shaping
- `vector_store.py` — ChromaDB + embedding (`all-MiniLM-L6-v2`) + retrieval
- `state_manager.py` — Persistent channel/video tracking (`data/pipeline_state.json`)
- `persona_builder.py` — Samples transcripts → calls `gpt-4o-mini` → saves `data/personas/{handle}.json`
- `channel_agent.py` — `ChannelAgent`: conversational agent with persona system prompt + RAG tool loop
- `scheduler.py` — APScheduler-based periodic re-indexing
- `mcp_server.py` — FastMCP server with 7 tools (query, index, list, stats, delete, get_channel_agent, chat_with_channel_agent)
- `config.py` — Central config + directory bootstrapping

## Runtime Data (git-ignored)
- `data/chroma_db/` — ChromaDB vector index
- `data/pipeline_state.json` — tracked channel/video IDs
- `data/personas/{handle}.json` — creator persona profiles
- `data/sessions/{handle}.json` — per-channel conversation history
- `logs/pipeline.log` — application logs

## Prerequisites
- Python 3.11+
- `OPENAI_API_KEY` in `.env` (required for persona build and agent chat)
- Internet access for YouTube fetch and first-time embedding model download

## Standard Commands
```bash
python main.py index "https://www.youtube.com/@channel" --limit 5
python main.py query "@channel" "your question" -n 3
python main.py status
python main.py channels
python main.py mcp
```

## Verification After Changes
1. `python main.py status` runs without import errors
2. `index` completes with non-zero chunk count
3. `query` returns at least one result for an indexed channel
4. `data/personas/{handle}.json` is created after indexing (requires `OPENAI_API_KEY`)

## Implementation Notes
- Flat module layout — all `.py` files at repo root. Imports are `from module import ...`.
- Incremental indexing: already-processed video IDs are filtered before fetching. ChromaDB upserts with deterministic chunk IDs (`{video_id}_chunk_{i:04d}`) make re-runs idempotent.
- Transcript availability varies by video; missing transcripts are non-fatal.
