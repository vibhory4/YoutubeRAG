#!/usr/bin/env python3
"""
YouTube RAG Pipeline - CLI Entry Point

Usage:
    # Index a channel (first time - gets all videos)
    python main.py index "https://www.youtube.com/@marketsbyzerodha"

    # Index with limit
    python main.py index "https://www.youtube.com/@marketsbyzerodha" --limit 10

    # Query the knowledge base
    python main.py query "@marketsbyzerodha" "What is SIP and how does it work?"

    # Start auto-update scheduler
    python main.py scheduler

    # Start MCP server (for Claude integration)
    python main.py mcp

    # View pipeline status
    python main.py status

    # List all tracked channels
    python main.py channels
"""

import argparse
import json
import logging
import sys

from src.config import config


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.log_file),
        ],
    )


def cmd_index(args):
    """Index a YouTube channel."""
    from src.pipeline import run_pipeline

    print(f"Indexing channel: {args.channel}")
    if args.limit:
        print(f"  Limit: {args.limit} videos")

    result = run_pipeline(channel_input=args.channel, limit=args.limit)

    print("\n" + "=" * 50)
    print("PIPELINE RESULTS")
    print("=" * 50)
    print(f"  Status:        {result.get('status')}")
    print(f"  New videos:    {result.get('new_video_count', 0)}")
    print(f"  Cleaned docs:  {result.get('clean_count', 0)}")

    stats = result.get("ingestion_stats", {})
    print(f"  Chunks added:  {stats.get('total_chunks', 0)}")

    errors = result.get("errors", [])
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for err in errors[:10]:
            print(f"    - {err}")


def cmd_query(args):
    """Query the knowledge base."""
    from src.vector_store import VectorStoreManager

    store = VectorStoreManager()
    results = store.query(
        channel_name=args.channel,
        query_text=args.query,
        n_results=args.n,
    )

    if not results["documents"]:
        print("No results found. Is the channel indexed?")
        return

    print(f"\nQuery: {args.query}")
    print(f"Channel: {args.channel}")
    print(f"Results: {results['total_results']}")
    print("=" * 60)

    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"], results["metadatas"], results["distances"])
    ):
        similarity = round(1 - dist, 4)
        print(f"\n--- Result {i + 1} (score: {similarity}) ---")
        print(f"Video: {meta.get('title', 'Unknown')}")
        print(f"URL:   {meta.get('video_url', '')}")
        print(f"\n{doc[:500]}{'...' if len(doc) > 500 else ''}")


def cmd_scheduler(args):
    """Start the auto-update scheduler."""
    from src.scheduler import start_scheduler

    interval = args.interval or config.update_interval_minutes
    start_scheduler(interval_minutes=interval)


def cmd_mcp(args):
    """Start the MCP server."""
    from src.mcp_server import mcp

    print("Starting YouTube RAG MCP Server...")
    print("Configure in Claude Desktop's claude_desktop_config.json")
    mcp.run()


def cmd_status(args):
    """Show pipeline status."""
    from src.state_manager import PipelineState
    from src.vector_store import VectorStoreManager

    state = PipelineState()
    summary = state.get_summary()

    print("\n" + "=" * 50)
    print("YOUTUBE RAG PIPELINE STATUS")
    print("=" * 50)
    print(f"  Channels tracked:    {summary['total_channels']}")
    print(f"  Videos indexed:      {summary['total_videos_indexed']}")
    print(f"  Last updated:        {summary.get('last_updated', 'Never')}")

    if summary["channels"]:
        print("\n  Channels:")
        for ch, info in summary["channels"].items():
            print(f"    {ch}")
            print(f"      Videos: {info['videos_indexed']}")
            print(f"      Last check: {info.get('last_checked', 'Never')}")

    # Vector store stats
    try:
        store = VectorStoreManager()
        collections = store.list_collections()
        if collections:
            print("\n  Vector Store Collections:")
            for col in collections:
                print(f"    {col['name']}: {col['document_count']} chunks")
    except Exception as e:
        print(f"\n  Vector store: Error - {e}")


def cmd_channels(args):
    """List all tracked channels."""
    from src.state_manager import PipelineState

    state = PipelineState()
    channels = state.get_tracked_channels()

    if not channels:
        print("No channels tracked yet. Use 'index' to add one.")
        return

    print(f"\nTracked channels ({len(channels)}):")
    for ch in channels:
        info = state.get_channel_info(ch)
        vids = len(info.get("indexed_video_ids", []))
        print(f"  {ch} ({vids} videos)")


def main():
    parser = argparse.ArgumentParser(
        description="YouTube RAG Pipeline - Index YouTube channels and query with RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # index
    p_index = subparsers.add_parser("index", help="Index a YouTube channel")
    p_index.add_argument("channel", help="YouTube channel URL or @handle")
    p_index.add_argument("--limit", type=int, help="Max videos to process")
    p_index.set_defaults(func=cmd_index)

    # query
    p_query = subparsers.add_parser("query", help="Query the knowledge base")
    p_query.add_argument("channel", help="Channel to query")
    p_query.add_argument("query", help="Search query")
    p_query.add_argument("-n", type=int, default=5, help="Number of results")
    p_query.set_defaults(func=cmd_query)

    # scheduler
    p_sched = subparsers.add_parser("scheduler", help="Start auto-update scheduler")
    p_sched.add_argument("--interval", type=int, help="Check interval in minutes")
    p_sched.set_defaults(func=cmd_scheduler)

    # mcp
    p_mcp = subparsers.add_parser("mcp", help="Start MCP server for Claude")
    p_mcp.set_defaults(func=cmd_mcp)

    # status
    p_status = subparsers.add_parser("status", help="Show pipeline status")
    p_status.set_defaults(func=cmd_status)

    # channels
    p_channels = subparsers.add_parser("channels", help="List tracked channels")
    p_channels.set_defaults(func=cmd_channels)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    setup_logging()
    args.func(args)


if __name__ == "__main__":
    main()
