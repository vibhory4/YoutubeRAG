"""Pipeline state persistence - tracks indexed channels and videos."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import config

logger = logging.getLogger(__name__)


class PipelineState:
    """Persists pipeline state across runs (channels tracked, videos indexed, etc.)."""

    def __init__(self, state_file: str = None):
        self.state_file = Path(state_file or config.state_file)
        self.state = self._load()

    def _load(self) -> dict:
        """Load state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return {"channels": {}, "last_updated": None}

    def _save(self):
        """Persist state to disk."""
        self.state["last_updated"] = datetime.now().isoformat()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def add_channel(self, channel_input: str):
        """Register a channel for tracking."""
        if channel_input not in self.state["channels"]:
            self.state["channels"][channel_input] = {
                "added_at": datetime.now().isoformat(),
                "last_checked": None,
                "indexed_video_ids": [],
                "total_videos_found": 0,
                "total_videos_indexed": 0,
                "total_videos_failed": 0,
            }
            self._save()
            logger.info(f"Added channel: {channel_input}")

    def remove_channel(self, channel_input: str):
        """Remove a channel from tracking."""
        if channel_input in self.state["channels"]:
            del self.state["channels"][channel_input]
            self._save()
            logger.info(f"Removed channel: {channel_input}")

    def get_tracked_channels(self) -> list[str]:
        """Get all tracked channel inputs."""
        return list(self.state["channels"].keys())

    def get_indexed_video_ids(self, channel_input: str) -> set[str]:
        """Get set of already-indexed video IDs for a channel."""
        ch = self.state["channels"].get(channel_input, {})
        return set(ch.get("indexed_video_ids", []))

    def mark_videos_indexed(
        self, channel_input: str, video_ids: list[str], failed_ids: list[str] = None
    ):
        """Record newly indexed video IDs."""
        if channel_input not in self.state["channels"]:
            self.add_channel(channel_input)

        ch = self.state["channels"][channel_input]
        existing = set(ch["indexed_video_ids"])
        existing.update(video_ids)
        ch["indexed_video_ids"] = list(existing)
        ch["total_videos_indexed"] = len(existing)
        ch["last_checked"] = datetime.now().isoformat()

        if failed_ids:
            ch["total_videos_failed"] = ch.get("total_videos_failed", 0) + len(failed_ids)

        self._save()

    def get_channel_info(self, channel_input: str) -> Optional[dict]:
        """Get tracking info for a specific channel."""
        return self.state["channels"].get(channel_input)

    def get_summary(self) -> dict:
        """Get overall pipeline summary."""
        total_videos = sum(
            len(ch.get("indexed_video_ids", []))
            for ch in self.state["channels"].values()
        )
        return {
            "total_channels": len(self.state["channels"]),
            "total_videos_indexed": total_videos,
            "channels": {
                k: {
                    "videos_indexed": len(v.get("indexed_video_ids", [])),
                    "last_checked": v.get("last_checked"),
                }
                for k, v in self.state["channels"].items()
            },
            "last_updated": self.state.get("last_updated"),
        }
