"""YouTube channel video discovery and transcript fetching."""

import re
import logging
from dataclasses import dataclass
from typing import Optional

import scrapetube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Represents a YouTube video with its metadata and transcript."""

    video_id: str
    title: str
    channel_name: str
    channel_url: str
    transcript: Optional[str] = None
    duration_seconds: Optional[int] = None
    published_text: Optional[str] = None
    view_count: Optional[str] = None
    error: Optional[str] = None


def extract_channel_identifier(channel_input: str) -> dict:
    """
    Parse a YouTube channel URL or name into usable identifiers.

    Supports formats:
      - https://www.youtube.com/@handle
      - https://www.youtube.com/channel/UC...
      - https://www.youtube.com/c/ChannelName
      - @handle
      - Channel ID directly
    """
    channel_input = channel_input.strip().rstrip("/")

    # Handle @username URL
    match = re.search(r"youtube\.com/@([^/\s?]+)", channel_input)
    if match:
        return {"channel_url": f"https://www.youtube.com/@{match.group(1)}"}

    # Handle /channel/UC... URL
    match = re.search(r"youtube\.com/channel/(UC[^/\s?]+)", channel_input)
    if match:
        return {"channel_id": match.group(1)}

    # Handle /c/Name URL
    match = re.search(r"youtube\.com/c/([^/\s?]+)", channel_input)
    if match:
        return {"channel_url": f"https://www.youtube.com/c/{match.group(1)}"}

    # Handle bare @handle
    if channel_input.startswith("@"):
        return {"channel_url": f"https://www.youtube.com/{channel_input}"}

    # Handle bare channel ID
    if channel_input.startswith("UC") and len(channel_input) == 24:
        return {"channel_id": channel_input}

    # Fallback: treat as URL
    return {"channel_url": channel_input}


def fetch_channel_videos(
    channel_input: str, limit: Optional[int] = None
) -> list[dict]:
    """
    Fetch all video metadata from a YouTube channel.

    Returns list of dicts with videoId, title, etc.
    """
    identifier = extract_channel_identifier(channel_input)
    logger.info(f"Fetching videos for channel: {identifier}")

    try:
        videos = scrapetube.get_channel(
            **identifier,
            limit=limit,
            sort_by="newest",
        )
        video_list = []
        for v in videos:
            video_list.append(
                {
                    "videoId": v["videoId"],
                    "title": v.get("title", {}).get("runs", [{}])[0].get("text", "Unknown"),
                    "publishedTimeText": v.get("publishedTimeText", {}).get("simpleText", ""),
                    "viewCountText": v.get("viewCountText", {}).get("simpleText", ""),
                    "lengthText": v.get("lengthText", {}).get("simpleText", ""),
                }
            )
        logger.info(f"Found {len(video_list)} videos")
        return video_list
    except Exception as e:
        logger.error(f"Error fetching channel videos: {e}")
        raise


def fetch_transcript(video_id: str, languages: list[str] = None) -> Optional[str]:
    """
    Fetch and format the transcript for a single YouTube video.

    Tries manual transcripts first, falls back to auto-generated.
    Returns cleaned plain text or None if unavailable.
    """
    if languages is None:
        languages = ["en", "hi", "en-IN"]

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=languages)

        # Format as plain text
        formatter = TextFormatter()
        text = formatter.format_transcript(transcript)

        # Basic cleanup
        text = re.sub(r"\[.*?\]", "", text)  # Remove [Music], [Applause], etc.
        text = re.sub(r"\n{3,}", "\n\n", text)  # Collapse excessive newlines
        text = text.strip()

        return text if text else None

    except Exception as e:
        logger.warning(f"Could not fetch transcript for {video_id}: {e}")
        return None


def get_video_infos(
    channel_input: str,
    limit: Optional[int] = None,
    existing_video_ids: set[str] = None,
) -> list[VideoInfo]:
    """
    Full pipeline: discover videos → fetch transcripts → return VideoInfo objects.

    If existing_video_ids is provided, only new videos are processed.
    """
    if existing_video_ids is None:
        existing_video_ids = set()

    raw_videos = fetch_channel_videos(channel_input, limit=limit)

    # Filter to only new videos
    new_videos = [v for v in raw_videos if v["videoId"] not in existing_video_ids]
    logger.info(
        f"{len(new_videos)} new videos to process (out of {len(raw_videos)} total)"
    )

    results = []
    for v in new_videos:
        vid = v["videoId"]
        logger.info(f"Processing: {v['title']} ({vid})")

        transcript = fetch_transcript(vid)

        info = VideoInfo(
            video_id=vid,
            title=v["title"],
            channel_name=channel_input,
            channel_url=f"https://www.youtube.com/watch?v={vid}",
            transcript=transcript,
            published_text=v.get("publishedTimeText", ""),
            view_count=v.get("viewCountText", ""),
            error=None if transcript else "Transcript unavailable",
        )
        results.append(info)

    return results
