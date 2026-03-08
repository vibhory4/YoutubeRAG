"""
LangGraph-based YouTube RAG Pipeline.

This is the core orchestration layer that connects:
  1. YouTube video discovery (scrapetube)
  2. Transcript fetching (youtube-transcript-api)
  3. Document cleaning (docling-inspired cleaner)
  4. Vector store ingestion (ChromaDB)

The graph supports both full indexing and incremental updates.
"""

import logging
from typing import Optional
from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END

from youtube_fetcher import get_video_infos, VideoInfo
from document_cleaner import process_transcript, CleanedDocument
from vector_store import VectorStoreManager
from state_manager import PipelineState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Graph State Definition
# ─────────────────────────────────────────────

class PipelineGraphState(TypedDict):
    """State that flows through the LangGraph pipeline."""

    # Input
    channel_input: str
    limit: Optional[int]

    # After discovery
    video_infos: list[VideoInfo]
    new_video_count: int

    # After cleaning
    cleaned_docs: list[CleanedDocument]
    clean_count: int

    # After ingestion
    ingestion_stats: dict

    # Overall
    status: str
    errors: list[str]


# ─────────────────────────────────────────────
# Graph Nodes
# ─────────────────────────────────────────────

def discover_videos(state: PipelineGraphState) -> dict:
    """
    Node 1: Discover new videos from the YouTube channel.

    Uses scrapetube to list all videos, filters out already-indexed ones.
    """
    channel_input = state["channel_input"]
    limit = state.get("limit")

    logger.info(f"[DISCOVER] Starting video discovery for: {channel_input}")

    try:
        # Get already-indexed video IDs from persistent state
        pipeline_state = PipelineState()
        existing_ids = pipeline_state.get_indexed_video_ids(channel_input)

        # Fetch and filter videos, get transcripts
        video_infos = get_video_infos(
            channel_input=channel_input,
            limit=limit,
            existing_video_ids=existing_ids,
        )

        # Split into those with and without transcripts
        with_transcript = [v for v in video_infos if v.transcript]
        without_transcript = [v for v in video_infos if not v.transcript]

        errors = []
        if without_transcript:
            for v in without_transcript:
                errors.append(f"No transcript: {v.title} ({v.video_id})")

        logger.info(
            f"[DISCOVER] Found {len(video_infos)} new videos, "
            f"{len(with_transcript)} with transcripts"
        )

        return {
            "video_infos": with_transcript,
            "new_video_count": len(with_transcript),
            "errors": state.get("errors", []) + errors,
            "status": "discovered",
        }

    except Exception as e:
        logger.error(f"[DISCOVER] Error: {e}")
        return {
            "video_infos": [],
            "new_video_count": 0,
            "errors": state.get("errors", []) + [f"Discovery error: {str(e)}"],
            "status": "discovery_failed",
        }


def clean_documents(state: PipelineGraphState) -> dict:
    """
    Node 2: Clean and normalize all fetched transcripts.

    Applies docling-inspired text processing.
    """
    video_infos = state.get("video_infos", [])

    if not video_infos:
        logger.info("[CLEAN] No videos to clean, skipping")
        return {
            "cleaned_docs": [],
            "clean_count": 0,
            "status": "no_docs_to_clean",
        }

    logger.info(f"[CLEAN] Processing {len(video_infos)} transcripts")

    cleaned_docs = []
    errors = []

    for info in video_infos:
        try:
            doc = process_transcript(
                video_id=info.video_id,
                title=info.title,
                channel_name=info.channel_name,
                raw_transcript=info.transcript,
                published_text=info.published_text,
                view_count=info.view_count,
            )

            if doc.word_count > 10:  # Minimum viable content
                cleaned_docs.append(doc)
            else:
                errors.append(
                    f"Too short after cleaning: {info.title} ({info.video_id})"
                )

        except Exception as e:
            errors.append(f"Clean error for {info.video_id}: {str(e)}")
            logger.error(f"[CLEAN] Error processing {info.video_id}: {e}")

    logger.info(f"[CLEAN] Cleaned {len(cleaned_docs)} documents successfully")

    return {
        "cleaned_docs": cleaned_docs,
        "clean_count": len(cleaned_docs),
        "errors": state.get("errors", []) + errors,
        "status": "cleaned",
    }


def ingest_to_vectorstore(state: PipelineGraphState) -> dict:
    """
    Node 3: Chunk and ingest cleaned documents into ChromaDB.
    """
    cleaned_docs = state.get("cleaned_docs", [])
    channel_input = state["channel_input"]

    if not cleaned_docs:
        logger.info("[INGEST] No documents to ingest, skipping")
        return {
            "ingestion_stats": {"documents_processed": 0, "total_chunks": 0},
            "status": "no_docs_to_ingest",
        }

    logger.info(f"[INGEST] Ingesting {len(cleaned_docs)} documents into ChromaDB")

    try:
        store = VectorStoreManager()
        stats = store.add_documents(channel_input, cleaned_docs)

        # Update persistent state with newly indexed video IDs
        pipeline_state = PipelineState()
        pipeline_state.add_channel(channel_input)
        indexed_ids = [doc.video_id for doc in cleaned_docs]
        failed_ids = []  # Could track per-doc failures
        pipeline_state.mark_videos_indexed(channel_input, indexed_ids, failed_ids)

        logger.info(
            f"[INGEST] Done: {stats['documents_processed']} docs, "
            f"{stats['total_chunks']} chunks"
        )

        return {
            "ingestion_stats": stats,
            "status": "ingested",
        }

    except Exception as e:
        logger.error(f"[INGEST] Error: {e}")
        return {
            "ingestion_stats": {"error": str(e)},
            "errors": state.get("errors", []) + [f"Ingestion error: {str(e)}"],
            "status": "ingestion_failed",
        }


def should_continue_to_clean(state: PipelineGraphState) -> str:
    """Conditional edge: skip cleaning if no new videos found."""
    if state.get("new_video_count", 0) == 0:
        return "end"
    return "clean"


def should_continue_to_ingest(state: PipelineGraphState) -> str:
    """Conditional edge: skip ingestion if no cleaned docs."""
    if state.get("clean_count", 0) == 0:
        return "end"
    return "ingest"


# ─────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────

def build_pipeline_graph() -> StateGraph:
    """
    Build the LangGraph pipeline:

        START → discover_videos → clean_documents → ingest_to_vectorstore → END
                      ↓ (no new)        ↓ (no docs)
                     END               END
    """
    graph = StateGraph(PipelineGraphState)

    # Add nodes
    graph.add_node("discover", discover_videos)
    graph.add_node("clean", clean_documents)
    graph.add_node("ingest", ingest_to_vectorstore)

    # Add edges
    graph.add_edge(START, "discover")

    graph.add_conditional_edges(
        "discover",
        should_continue_to_clean,
        {"clean": "clean", "end": END},
    )

    graph.add_conditional_edges(
        "clean",
        should_continue_to_ingest,
        {"ingest": "ingest", "end": END},
    )

    graph.add_edge("ingest", END)

    return graph


def compile_pipeline():
    """Build and compile the pipeline graph."""
    graph = build_pipeline_graph()
    return graph.compile()


# ─────────────────────────────────────────────
# Convenience runner
# ─────────────────────────────────────────────

def run_pipeline(
    channel_input: str, limit: Optional[int] = None
) -> PipelineGraphState:
    """
    Run the full pipeline for a YouTube channel.

    Args:
        channel_input: YouTube channel URL, @handle, or channel ID
        limit: Optional limit on number of videos to process

    Returns:
        Final pipeline state with all results
    """
    app = compile_pipeline()

    initial_state: PipelineGraphState = {
        "channel_input": channel_input,
        "limit": limit,
        "video_infos": [],
        "new_video_count": 0,
        "cleaned_docs": [],
        "clean_count": 0,
        "ingestion_stats": {},
        "status": "starting",
        "errors": [],
    }

    logger.info(f"Running pipeline for: {channel_input}")
    result = app.invoke(initial_state)
    logger.info(f"Pipeline complete. Status: {result.get('status')}")

    return result
