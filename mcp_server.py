"""
MCP Server for YouTube RAG Pipeline.

Exposes the RAG vector store as MCP tools so Claude can query
YouTube channel transcripts directly.

Run with:
    python -m src.mcp_server

Configure in Claude Desktop claude_desktop_config.json:
    {
      "mcpServers": {
        "youtube_rag": {
          "command": "python",
          "args": ["-m", "src.mcp_server"],
          "cwd": "/path/to/youtube-rag-pipeline"
        }
      }
    }
"""

import json
import logging
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from vector_store import VectorStoreManager
from state_manager import PipelineState
from pipeline import run_pipeline
from config import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Initialize MCP Server
# ─────────────────────────────────────────────

mcp = FastMCP("youtube_rag_mcp")


# ─────────────────────────────────────────────
# Input Models
# ─────────────────────────────────────────────

class QueryInput(BaseModel):
    """Input for querying the YouTube RAG knowledge base."""

    channel: str = Field(
        ...,
        description="YouTube channel URL, @handle, or name (e.g., 'https://www.youtube.com/@marketsbyzerodha' or '@marketsbyzerodha')",
    )
    query: str = Field(
        ...,
        description="Search query to find relevant transcript content",
    )
    n_results: int = Field(
        default=5,
        description="Number of results to return (1-20)",
        ge=1,
        le=20,
    )


class ChannelInput(BaseModel):
    """Input for channel management operations."""

    channel: str = Field(
        ...,
        description="YouTube channel URL, @handle, or name",
    )


class IndexChannelInput(BaseModel):
    """Input for indexing a new channel."""

    channel: str = Field(
        ...,
        description="YouTube channel URL, @handle, or name to index",
    )
    limit: Optional[int] = Field(
        default=None,
        description="Max number of videos to index (None = all videos)",
        ge=1,
    )


# ─────────────────────────────────────────────
# MCP Tools
# ─────────────────────────────────────────────

@mcp.tool(
    name="youtube_rag_query",
    annotations={
        "title": "Query YouTube Channel Knowledge Base",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def youtube_rag_query(params: QueryInput) -> str:
    """
    Search through indexed YouTube channel transcripts to find relevant content.

    This tool queries the RAG vector store containing transcripts from YouTube
    channels. It returns the most relevant transcript chunks with metadata
    including video title, URL, and relevance score.

    Use this to answer questions about content discussed in YouTube videos
    from tracked channels.

    Args:
        params (QueryInput): Contains channel identifier, search query, and result count.

    Returns:
        str: JSON-formatted search results with relevant transcript excerpts and metadata.
    """
    try:
        store = VectorStoreManager()
        results = store.query(
            channel_name=params.channel,
            query_text=params.query,
            n_results=params.n_results,
        )

        if not results["documents"]:
            return json.dumps({
                "status": "no_results",
                "message": f"No relevant content found for query: '{params.query}'. "
                f"Make sure the channel '{params.channel}' has been indexed.",
            })

        # Format results for Claude
        formatted = []
        for doc, meta, dist in zip(
            results["documents"],
            results["metadatas"],
            results["distances"],
        ):
            formatted.append({
                "content": doc,
                "video_title": meta.get("title", "Unknown"),
                "video_url": meta.get("video_url", ""),
                "video_id": meta.get("video_id", ""),
                "relevance_score": round(1 - dist, 4),  # Convert distance to similarity
                "chunk_index": meta.get("chunk_index", 0),
            })

        return json.dumps({
            "status": "success",
            "query": params.query,
            "channel": params.channel,
            "total_results": len(formatted),
            "results": formatted,
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Query failed: {str(e)}",
        })


@mcp.tool(
    name="youtube_rag_index_channel",
    annotations={
        "title": "Index a YouTube Channel",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def youtube_rag_index_channel(params: IndexChannelInput) -> str:
    """
    Index a YouTube channel by fetching all video transcripts and storing them
    in the RAG knowledge base. This is idempotent - re-running will only
    process new videos.

    Args:
        params (IndexChannelInput): Contains channel URL/handle and optional video limit.

    Returns:
        str: JSON summary of indexing results including videos processed and chunks created.
    """
    try:
        result = run_pipeline(
            channel_input=params.channel,
            limit=params.limit,
        )

        return json.dumps({
            "status": result.get("status", "unknown"),
            "channel": params.channel,
            "new_videos_found": result.get("new_video_count", 0),
            "documents_cleaned": result.get("clean_count", 0),
            "ingestion_stats": result.get("ingestion_stats", {}),
            "errors": result.get("errors", [])[:10],  # Cap error list
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Indexing failed: {str(e)}",
        })


@mcp.tool(
    name="youtube_rag_list_channels",
    annotations={
        "title": "List Tracked YouTube Channels",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def youtube_rag_list_channels() -> str:
    """
    List all YouTube channels currently tracked and indexed in the RAG system.

    Returns channel names, video counts, and last update times.

    Returns:
        str: JSON list of tracked channels with statistics.
    """
    try:
        state = PipelineState()
        summary = state.get_summary()

        store = VectorStoreManager()
        collections = store.list_collections()

        return json.dumps({
            "status": "success",
            "pipeline_summary": summary,
            "vector_store_collections": collections,
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to list channels: {str(e)}",
        })


@mcp.tool(
    name="youtube_rag_channel_stats",
    annotations={
        "title": "Get Channel Index Statistics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def youtube_rag_channel_stats(params: ChannelInput) -> str:
    """
    Get detailed statistics about an indexed YouTube channel.

    Shows total videos indexed, chunk count, and list of video IDs.

    Args:
        params (ChannelInput): Contains the channel identifier.

    Returns:
        str: JSON with detailed channel index statistics.
    """
    try:
        store = VectorStoreManager()
        stats = store.get_stats(params.channel)

        state = PipelineState()
        channel_info = state.get_channel_info(params.channel)

        return json.dumps({
            "status": "success",
            "vector_store_stats": stats,
            "pipeline_state": channel_info,
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to get stats: {str(e)}",
        })


@mcp.tool(
    name="youtube_rag_delete_channel",
    annotations={
        "title": "Delete Channel from Index",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def youtube_rag_delete_channel(params: ChannelInput) -> str:
    """
    Remove a YouTube channel and all its indexed data from the RAG system.

    WARNING: This permanently deletes all indexed transcripts for the channel.

    Args:
        params (ChannelInput): Contains the channel identifier to delete.

    Returns:
        str: JSON confirmation of deletion.
    """
    try:
        store = VectorStoreManager()
        deleted = store.delete_collection(params.channel)

        state = PipelineState()
        state.remove_channel(params.channel)

        return json.dumps({
            "status": "deleted" if deleted else "not_found",
            "channel": params.channel,
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Deletion failed: {str(e)}",
        })


class GetAgentInput(BaseModel):
    """Input for getting a channel agent."""

    channel: str = Field(
        ...,
        description="YouTube channel URL or @handle (must already be indexed)",
    )


class ChatInput(BaseModel):
    """Input for chatting with a channel agent."""

    session_id: str = Field(
        ...,
        description="session_id returned by get_channel_agent",
    )
    message: str = Field(
        ...,
        description="Your message to the channel agent",
    )


@mcp.tool(
    name="get_channel_agent",
    annotations={
        "title": "Get a YouTube Channel Agent",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_channel_agent(params: GetAgentInput) -> str:
    """
    Get a conversational agent for an indexed YouTube channel.

    The agent talks like the creator and answers questions based on their
    video content. Returns a session_id to use with chat_with_channel_agent.

    The channel must already be indexed (use youtube_rag_index_channel first).

    Args:
        params (GetAgentInput): Contains the channel URL or @handle.

    Returns:
        str: JSON with session_id, creator info, and usage instructions.
    """
    try:
        state = PipelineState()
        channel_info = state.get_channel_info(params.channel)

        if not channel_info:
            return json.dumps({
                "status": "not_indexed",
                "message": (
                    f"Channel '{params.channel}' is not indexed yet. "
                    "Use youtube_rag_index_channel to index it first."
                ),
            })

        from channel_agent import ChannelAgent
        from persona_builder import load_persona

        agent = ChannelAgent(params.channel)
        agent.ensure_session_file()

        persona = load_persona(params.channel)

        return json.dumps({
            "status": "ready",
            "session_id": agent.session_id,
            "channel": params.channel,
            "display_name": persona.get("display_name", params.channel) if persona else params.channel,
            "topics": persona.get("topics", []) if persona else [],
            "tone": persona.get("tone", "unknown") if persona else "unknown",
            "videos_indexed": channel_info.get("total_videos_indexed", 0),
            "session_messages": len(agent.messages),
            "persona_available": persona is not None,
            "instructions": (
                f"Agent is ready. Use chat_with_channel_agent with "
                f"session_id='{agent.session_id}' to start chatting."
            ),
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to get agent: {str(e)}",
        })


@mcp.tool(
    name="chat_with_channel_agent",
    annotations={
        "title": "Chat with a YouTube Channel Agent",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def chat_with_channel_agent(params: ChatInput) -> str:
    """
    Send a message to a YouTube channel agent and get a response in the creator's voice.

    The agent searches the channel's indexed video transcripts to answer your
    question, responding as the creator would.

    Call get_channel_agent first to obtain a session_id.

    Args:
        params (ChatInput): Contains session_id and the message to send.

    Returns:
        str: The agent's response in the creator's voice.
    """
    try:
        # Resolve session_id → channel_input via the session file
        session_path = Path(config.sessions_dir) / f"{params.session_id}.json"
        if not session_path.exists():
            return json.dumps({
                "status": "session_not_found",
                "message": (
                    f"No session found for session_id='{params.session_id}'. "
                    "Call get_channel_agent first."
                ),
            })

        session_data = json.loads(session_path.read_text())
        channel_input = session_data.get("channel_input", params.session_id)

        from channel_agent import ChannelAgent

        agent = ChannelAgent(channel_input)
        response = agent.chat(params.message)
        return response

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Chat failed: {str(e)}",
        })


# ─────────────────────────────────────────────
# Server Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
