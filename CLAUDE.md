# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` to override defaults (all optional).

## Common Commands

```bash
# Index a channel (--limit for testing)
python main.py index "https://www.youtube.com/@channel" --limit 5

# Query indexed transcripts
python main.py query "@channel" "your question here" -n 5

# Check state
python main.py status
python main.py channels

# Start MCP server (for Claude Desktop/Code integration)
python main.py mcp

# Start auto-update scheduler
python main.py scheduler --interval 60
```

There is no test suite. After changes, verify:
1. `python main.py status` runs without import errors
2. Index completes with non-zero chunk count
3. Query returns at least one result for an indexed channel

## Architecture

**Core Flow:** `discover_videos → clean_documents → ingest_to_vectorstore`

Orchestrated as a **LangGraph StateGraph** in [pipeline.py](pipeline.py). The graph state (`PipelineGraphState` TypedDict) flows through three nodes with conditional edges that skip downstream nodes if there's no new data.

**Flat module layout** — all Python files are at the repo root (no `src/` package). Imports are `from module import ...`.

### Key Files

| File | Role |
|------|------|
| [main.py](main.py) | argparse CLI entry point for all subcommands |
| [pipeline.py](pipeline.py) | LangGraph graph definition; call `run_pipeline(channel_input, limit)` |
| [youtube_fetcher.py](youtube_fetcher.py) | Parses channel URLs/@handles, fetches video list (scrapetube) and transcripts |
| [document_cleaner.py](document_cleaner.py) | Removes filler words/brackets, reconstructs paragraphs from auto-captions |
| [vector_store.py](vector_store.py) | ChromaDB CRUD; chunks text with `RecursiveCharacterTextSplitter`, upserts with deterministic IDs |
| [state_manager.py](state_manager.py) | JSON state at `data/pipeline_state.json` tracking indexed video IDs per channel |
| [mcp_server.py](mcp_server.py) | FastMCP server exposing 5 tools: query, index, list, stats, delete |
| [config.py](config.py) | `@dataclass` config with env-var overrides; creates `data/` and `logs/` dirs on import |

### Persistence

- `data/chroma_db/` — ChromaDB SQLite + HNSW index (per-channel collections)
- `data/pipeline_state.json` — tracked channels and their indexed video IDs
- `logs/pipeline.log` — application logs

### Incremental Indexing

`state_manager.get_indexed_video_ids(channel)` returns already-processed video IDs. The discover node filters these out so only new videos are fetched, cleaned, and embedded. ChromaDB writes use `upsert` with chunk IDs of the form `{video_id}_chunk_{i:04d}`, making re-runs fully idempotent.

### Embeddings

Uses `all-MiniLM-L6-v2` via `SentenceTransformerEmbeddingFunction` — no external API needed. Downloads on first run.

### MCP Integration

Configure in Claude Desktop's `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "youtube_rag": {
      "command": "python",
      "args": ["main.py", "mcp"],
      "cwd": "/absolute/path/to/YoutubeRAG"
    }
  }
}
```
