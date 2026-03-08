"""
Automatic update scheduler for the YouTube RAG Pipeline.

Uses APScheduler to periodically check tracked channels for new videos
and run the ingestion pipeline for any new content.
"""

import logging
import signal
import sys
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import config
from pipeline import run_pipeline
from state_manager import PipelineState

logger = logging.getLogger(__name__)


def check_all_channels():
    """Check all tracked channels for new videos and ingest them."""
    state = PipelineState()
    channels = state.get_tracked_channels()

    if not channels:
        logger.info("[SCHEDULER] No channels tracked. Add a channel first.")
        return

    logger.info(f"[SCHEDULER] Checking {len(channels)} channels at {datetime.now()}")

    for channel in channels:
        try:
            logger.info(f"[SCHEDULER] Processing: {channel}")
            result = run_pipeline(channel)

            new_count = result.get("new_video_count", 0)
            chunks = result.get("ingestion_stats", {}).get("total_chunks", 0)
            errors = result.get("errors", [])

            logger.info(
                f"[SCHEDULER] {channel}: "
                f"{new_count} new videos, {chunks} new chunks, "
                f"{len(errors)} errors"
            )

            if errors:
                for err in errors[:5]:  # Log first 5 errors
                    logger.warning(f"  → {err}")

        except Exception as e:
            logger.error(f"[SCHEDULER] Failed to process {channel}: {e}")


def start_scheduler(interval_minutes: int = None):
    """
    Start the background scheduler.

    Args:
        interval_minutes: Override for update interval (defaults to config)
    """
    interval = interval_minutes or config.update_interval_minutes

    scheduler = BlockingScheduler()

    scheduler.add_job(
        check_all_channels,
        trigger=IntervalTrigger(minutes=interval),
        id="youtube_rag_update",
        name="Check YouTube channels for new videos",
        replace_existing=True,
        next_run_time=datetime.now(),  # Run immediately on start
    )

    # Graceful shutdown
    def shutdown(signum, frame):
        logger.info("[SCHEDULER] Shutting down...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info(
        f"[SCHEDULER] Starting scheduler. "
        f"Checking every {interval} minutes."
    )
    print(f"YouTube RAG Scheduler started. Checking every {interval} minutes.")
    print("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("[SCHEDULER] Stopped.")
