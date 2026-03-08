"""
Document cleaning and normalization for YouTube transcripts.

Uses text processing inspired by Docling's approach:
- Structural normalization (paragraphs, sentences)
- Noise removal (filler words, repeated phrases)
- Metadata enrichment
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Common filler patterns in YouTube transcripts
FILLER_PATTERNS = [
    r"\b(um|uh|erm|hmm|ah|oh)\b",
    r"\b(you know|i mean|like)\b(?=\s+(you know|i mean|like)\b)",  # repeated fillers
    r"(?i)\[music\]",
    r"(?i)\[applause\]",
    r"(?i)\[laughter\]",
    r"(?i)\[inaudible\]",
    r"(?i)\[silence\]",
]

# Hindi filler patterns (common in Indian YouTube channels)
HINDI_FILLER_PATTERNS = [
    r"\b(matlab|toh|acha|haan|ji)\b(?=\s+\b(matlab|toh|acha|haan|ji)\b)",
]


@dataclass
class CleanedDocument:
    """A cleaned and structured document ready for chunking."""

    video_id: str
    title: str
    channel_name: str
    video_url: str
    clean_text: str
    word_count: int
    paragraph_count: int
    metadata: dict


def clean_transcript(raw_text: str) -> str:
    """
    Clean a raw YouTube transcript.

    Steps:
    1. Remove bracketed annotations [Music], etc.
    2. Remove excessive filler words
    3. Normalize whitespace
    4. Reconstruct paragraph boundaries using sentence heuristics
    5. Fix common OCR/auto-caption errors
    """
    if not raw_text:
        return ""

    text = raw_text

    # Step 1: Remove bracketed annotations
    text = re.sub(r"\[.*?\]", "", text)

    # Step 2: Remove filler patterns
    for pattern in FILLER_PATTERNS + HINDI_FILLER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Step 3: Fix common auto-caption artifacts
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # Fix hyphenated line breaks
    text = re.sub(r"(?<=[a-z])\n(?=[a-z])", " ", text)  # Join mid-sentence breaks
    text = re.sub(r"\n{2,}", "\n\n", text)  # Normalize paragraph breaks

    # Step 4: Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n +", "\n", text)

    # Step 5: Reconstruct paragraphs from continuous text
    # YouTube transcripts often lack paragraph breaks
    # Insert breaks at natural topic transitions (after multiple sentences)
    lines = text.split("\n")
    paragraphs = []
    current_paragraph = []

    for line in lines:
        line = line.strip()
        if not line:
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
            continue

        current_paragraph.append(line)

        # Check if we should break the paragraph
        joined = " ".join(current_paragraph)
        sentence_count = len(re.findall(r"[.!?]+\s", joined))
        if sentence_count >= 5:
            paragraphs.append(joined)
            current_paragraph = []

    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    text = "\n\n".join(paragraphs)

    # Final cleanup
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


def enrich_metadata(video_id: str, title: str, channel_name: str, **extra) -> dict:
    """Create a rich metadata dict for vector store."""
    metadata = {
        "video_id": video_id,
        "title": title,
        "channel_name": channel_name,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "source": "youtube",
        "content_type": "transcript",
    }
    metadata.update({k: v for k, v in extra.items() if v})
    return metadata


def process_transcript(
    video_id: str,
    title: str,
    channel_name: str,
    raw_transcript: str,
    **extra_metadata,
) -> CleanedDocument:
    """
    Full cleaning pipeline for a single video transcript.

    Returns a CleanedDocument ready for chunking and embedding.
    """
    logger.info(f"Cleaning transcript for: {title} ({video_id})")

    clean_text = clean_transcript(raw_transcript)
    paragraphs = [p for p in clean_text.split("\n\n") if p.strip()]
    word_count = len(clean_text.split())

    metadata = enrich_metadata(
        video_id=video_id,
        title=title,
        channel_name=channel_name,
        **extra_metadata,
    )

    doc = CleanedDocument(
        video_id=video_id,
        title=title,
        channel_name=channel_name,
        video_url=f"https://www.youtube.com/watch?v={video_id}",
        clean_text=clean_text,
        word_count=word_count,
        paragraph_count=len(paragraphs),
        metadata=metadata,
    )

    logger.info(
        f"Cleaned: {word_count} words, {len(paragraphs)} paragraphs"
    )
    return doc
