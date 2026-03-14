# YouTube Channel RAG Pipeline

A **LangGraph-based** system that indexes YouTube channel transcripts into a ChromaDB vector store, with automatic updates for new videos and an **MCP server** for querying directly from Claude.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Pipeline                           │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   DISCOVER    │───▶│    CLEAN     │───▶│  INGEST TO RAG   │  │
│  │              │    │              │    │                  │  │
│  │ • scrapetube │    │ • Filler     │    │ • Text splitting │  │
│  │ • transcript │    │   removal    │    │ • Embeddings     │  │
│  │   api        │    │ • Paragraph  │    │ • ChromaDB       │  │
│  │ • Dedup      │    │   rebuild    │    │   upsert         │  │
│  └──────┬───────┘    │ • Normalize  │    └────────┬─────────┘  │
│         │            └──────────────┘             │            │
│    (no new? → END)                          (no docs? → END)   │
└─────────────────────────────────────────────────────────────────┘
         │                                          │
         ▼                                          ▼
┌─────────────────┐                     ┌────────────────────┐
│  State Manager  │                     │     ChromaDB       │
│  (JSON on disk) │                     │  (Persistent Store)│
└─────────────────┘                     └────────┬───────────┘
                                                 │
                            ┌────────────────────┤
                            ▼                    ▼
                    ┌──────────────┐    ┌──────────────────┐
                    │  CLI Query   │    │   MCP Server     │
                    │              │    │  (Claude Claude  │
                    │  main.py     │    │   Desktop/Code)  │
                    │  query ...   │    │                  │
                    └──────────────┘    └──────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    APScheduler (Auto-Updates)                    │
│  Periodically runs the pipeline for all tracked channels        │
└─────────────────────────────────────────────────────────────────┘
```

## Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Orchestration | **LangGraph** `StateGraph` | Pipeline flow control with conditional edges |
| Video Discovery | **scrapetube** | Scrape channel video list without API key |
| Transcripts | **youtube-transcript-api** | Fetch auto/manual captions |
| Cleaning | Custom (docling-inspired) | Filler removal, paragraph reconstruction |
| Embeddings | **sentence-transformers** | Local `all-MiniLM-L6-v2` embeddings |
| Vector Store | **ChromaDB** | Persistent vector DB with cosine similarity |
| Auto-Updates | **APScheduler** | Periodic new video checks |
| Claude Integration | **MCP Server** (FastMCP) | Query RAG directly from Claude |

## Setup

### 1. Clone & Install

```bash
cd youtube-rag-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add OPENAI_API_KEY if you want persona generation and chat
# Edit other values only if you need custom paths, models, or intervals
```

You can index and query without an API key. `OPENAI_API_KEY` is required for persona generation and chat features.

## How To Run

Run everything from the repository root after activating `.venv`.

### CLI

```bash
# Show available commands
python main.py --help

# Check the project is wired correctly
python main.py status

# Index a channel
python main.py index "https://www.youtube.com/@marketsbyzerodha" --limit 5

# Query an indexed channel
python main.py query "@marketsbyzerodha" "What is SIP and how does it work?" -n 3

# List tracked channels
python main.py channels
```

### Chainlit Chat App

```bash
chainlit run app.py
```

Then open the local URL shown by Chainlit in your terminal. The app lets you pick an indexed channel and chat with its persona. If no channels are indexed yet, index one first with `python main.py index ...`.

### MCP Server

```bash
python main.py mcp
```

Start this when you want to connect the project to an MCP client such as Claude Desktop.

## Usage

### Index a YouTube Channel

```bash
# Index all videos from a channel
python main.py index "https://www.youtube.com/@marketsbyzerodha"

# Index with a limit (good for testing)
python main.py index "@marketsbyzerodha" --limit 5
```

### Query the Knowledge Base

```bash
python main.py query "@marketsbyzerodha" "What is SIP and how does it work?"
python main.py query "@marketsbyzerodha" "explain mutual fund categories"
```

### Start Auto-Updates

```bash
# Check all tracked channels every 60 minutes (default)
python main.py scheduler

# Custom interval
python main.py scheduler --interval 30
```

### View Status

```bash
python main.py status     # Pipeline overview
python main.py channels   # List tracked channels
python main.py --help     # Full CLI help
```

## Claude Integration (MCP)

### Option A: Claude Desktop

1. Edit `claude_desktop_config.json` (Claude Desktop settings):

```json
{
  "mcpServers": {
    "youtube_rag": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/full/path/to/youtube-rag-pipeline"
    }
  }
}
```

2. Restart Claude Desktop. You'll see these tools available:
   - `youtube_rag_query` - Search indexed transcripts
   - `youtube_rag_index_channel` - Index a new channel
   - `youtube_rag_list_channels` - List tracked channels
   - `youtube_rag_channel_stats` - Get channel statistics
   - `youtube_rag_delete_channel` - Remove a channel

### Option B: Claude Code (CLI)

```bash
# Add the MCP server to Claude Code
claude mcp add youtube_rag -- python -m src.mcp_server
```

### Example Claude Prompts

Once connected, you can ask Claude:

> "Search the Zerodha channel for videos about tax harvesting"
> "What does Varsity say about options Greeks?"
> "Index the channel @waaborz and then search for budgeting tips"
> "List all channels I've indexed so far"

## LangGraph Pipeline Details

The pipeline uses `StateGraph` with three nodes and conditional edges:

```python
# Simplified view of the graph
START → discover_videos
            │
            ├── (new videos found) → clean_documents
            │                            │
            │                            ├── (docs cleaned) → ingest_to_vectorstore → END
            │                            │
            │                            └── (no docs) → END
            │
            └── (no new videos) → END
```

**State** flows through the graph as a `TypedDict`:

```python
class PipelineGraphState(TypedDict):
    channel_input: str
    limit: Optional[int]
    video_infos: list[VideoInfo]
    new_video_count: int
    cleaned_docs: list[CleanedDocument]
    clean_count: int
    ingestion_stats: dict
    status: str
    errors: list[str]
```

## Project Structure

```
youtube-rag-pipeline/
├── main.py                        # CLI entry point
├── requirements.txt
├── .env.example                   # Config template
├── claude_desktop_config.example.json
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration management
│   ├── youtube_fetcher.py         # Video discovery + transcript fetching
│   ├── document_cleaner.py        # Transcript cleaning (docling-inspired)
│   ├── vector_store.py            # ChromaDB management
│   ├── state_manager.py           # Pipeline state persistence
│   ├── pipeline.py                # LangGraph pipeline (core)
│   ├── scheduler.py               # APScheduler auto-updates
│   └── mcp_server.py              # MCP server for Claude
├── data/
│   ├── chroma_db/                 # ChromaDB persistent storage
│   └── pipeline_state.json        # Pipeline state file
└── logs/
    └── pipeline.log
```

## How Auto-Updates Work

1. **Scheduler** runs every N minutes (configurable)
2. For each tracked channel, it calls `run_pipeline()`
3. The LangGraph `discover` node uses `scrapetube` to get latest videos
4. It compares against `pipeline_state.json` to find new ones only
5. New transcripts are fetched, cleaned, chunked, and embedded
6. ChromaDB `upsert` ensures idempotency (safe to re-run)

## Customization

### Add a New Embedding Model

Edit `.env`:
```
EMBEDDING_MODEL=all-mpnet-base-v2
```

### Tune Chunking

```
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```

### Support More Languages

In `src/youtube_fetcher.py`, modify the `languages` parameter:
```python
fetch_transcript(video_id, languages=["en", "hi", "ta", "te", "mr"])
```
