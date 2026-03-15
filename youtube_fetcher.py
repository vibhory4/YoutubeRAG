"""YouTube channel video discovery and transcript fetching."""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import yt_dlp
from google import genai
from google.genai import types

from config import config

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


def _normalize_channel_url(channel_input: str) -> str:
    """Normalize a channel input to a yt-dlp compatible URL."""
    channel_input = channel_input.strip().rstrip("/")

    # Bare @handle
    if channel_input.startswith("@"):
        return f"https://www.youtube.com/{channel_input}/videos"

    # Already a full YouTube URL
    if "youtube.com" in channel_input:
        # Strip trailing /videos if present so we can re-add it cleanly
        base = channel_input.rstrip("/")
        if base.endswith("/videos"):
            return base
        return base + "/videos"

    # Raw UC... channel ID
    if channel_input.startswith("UC") and len(channel_input) == 24:
        return f"https://www.youtube.com/channel/{channel_input}/videos"

    # Fallback: pass through as-is
    return channel_input


def fetch_channel_videos(
    channel_input: str, limit: Optional[int] = None
) -> list[dict]:
    """
    Fetch all video metadata from a YouTube channel using yt-dlp.

    Returns list of dicts with videoId, title, publishedTimeText,
    viewCountText, and lengthText.
    """
    url = _normalize_channel_url(channel_input)
    logger.info(f"Fetching videos from: {url}")

    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
    }
    if limit:
        ydl_opts["playlistend"] = limit

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        entries = info.get("entries", []) if info else []
        video_list = [
            {
                "videoId": e["id"],
                "title": e.get("title", "Unknown"),
                "publishedTimeText": e.get("upload_date", ""),  # YYYYMMDD
                "viewCountText": str(e.get("view_count", "") or ""),
                "lengthText": str(e.get("duration", "") or ""),
            }
            for e in entries
            if e and e.get("id")
        ]
        logger.info(f"Found {len(video_list)} videos")
        return video_list

    except Exception as e:
        logger.error(f"Error fetching channel videos: {e}")
        raise


def fetch_transcript(video_id: str) -> Optional[str]:
    """
    Extract the spoken transcript from a YouTube video using Gemini.

    Works even for videos without captions by using Gemini's multimodal
    video understanding. Returns plain text or None on failure.
    """
    if not config.gemini_api_key:
        logger.error("GEMINI_API_KEY not set — cannot fetch transcript")
        return None

    client = genai.Client(api_key=config.gemini_api_key)
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    prompt = [
        "Extract the complete spoken transcript from this YouTube video. "
        "Return only the spoken words as plain text. "
        "No timestamps, no speaker labels, no descriptions.",
        types.Part.from_uri(file_uri=video_url, mime_type="video/mp4"),
    ]

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            text = response.text.strip() if response.text else ""
            return text if text else None

        except Exception as e:
            err_str = str(e)
            # On rate-limit, wait and retry once
            if "429" in err_str and attempt == 0:
                import re
                delay_match = re.search(r"retry in (\d+)", err_str)
                wait = int(delay_match.group(1)) + 2 if delay_match else 60
                logger.warning(f"Rate limited on {video_id}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            logger.warning(f"Gemini could not extract transcript for {video_id}: {e}")
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
        time.sleep(2)  # avoid per-minute rate limits

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
