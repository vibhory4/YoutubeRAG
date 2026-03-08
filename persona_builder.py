"""
Persona builder for YouTube channel agents.

At index time, samples transcripts and uses gpt-4o-mini to extract
a creator persona profile, saved to data/personas/{handle}.json.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config import config
from document_cleaner import CleanedDocument

logger = logging.getLogger(__name__)

_SAMPLE_SIZE = 20
_EXCERPT_CHARS = 500


def _normalize_handle(channel_input: str) -> str:
    """Normalize a channel URL/handle to a safe filename stem."""
    name = channel_input.strip().lower()
    name = re.sub(r"https?://[^/]*youtube\.com/", "", name)
    name = re.sub(r"[^a-z0-9_-]", "_", name)
    name = name.strip("_")[:50]
    if not name or not name[0].isalnum():
        name = "yt_" + name
    if not name[-1].isalnum():
        name = name + "_col"
    return name


def _persona_path(channel_input: str) -> Path:
    return Path(config.personas_dir) / f"{_normalize_handle(channel_input)}.json"


def build_and_save_persona(
    channel_input: str,
    cleaned_docs: list[CleanedDocument],
) -> dict:
    """
    Build a creator persona from cleaned transcript docs and persist it.

    Samples up to SAMPLE_SIZE docs evenly, makes one gpt-4o-mini call,
    saves result to data/personas/{handle}.json, and returns the persona dict.
    """
    if not cleaned_docs:
        logger.warning("[PERSONA] No docs provided, skipping persona build")
        return {}

    # Sample evenly across all docs
    step = max(1, len(cleaned_docs) // _SAMPLE_SIZE)
    sample = cleaned_docs[::step][:_SAMPLE_SIZE]

    titles = [doc.title for doc in sample]
    excerpts = [doc.clean_text[:_EXCERPT_CHARS] for doc in sample]

    titles_block = "\n".join(f"- {t}" for t in titles)
    excerpts_block = "\n\n".join(
        f"[Video {i + 1}: {titles[i]}]\n{e}" for i, e in enumerate(excerpts)
    )

    prompt = f"""Analyze these YouTube video titles and transcript excerpts from a single creator's channel.
Your job is to extract a detailed persona profile so an AI can impersonate this creator.

Channel identifier: {channel_input}
Total videos sampled: {len(sample)} (out of {len(cleaned_docs)} indexed)

VIDEO TITLES:
{titles_block}

TRANSCRIPT EXCERPTS:
{excerpts_block}

Return a JSON object with EXACTLY these fields:
{{
  "display_name": "The creator's name or best guess from content (e.g. 'Zerodha Varsity', 'MKBHD')",
  "topics": ["topic1", "topic2"],
  "tone": "one of: casual, formal, conversational, educational, motivational, analytical",
  "style": "2-3 sentence description of HOW they communicate — vocabulary level, use of examples, pacing, analogies, rhetorical patterns",
  "common_phrases": ["phrase1", "phrase2", "phrase3"],
  "expertise_level": "one of: beginner-friendly, intermediate, expert",
  "language": "primary language of the content",
  "persona_summary": "3-4 sentences that capture the creator's unique voice and approach, written so an AI can use it as a system prompt to impersonate them"
}}"""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    persona = json.loads(response.choices[0].message.content)
    persona["channel_input"] = channel_input
    persona["built_at"] = datetime.utcnow().isoformat()
    persona["video_count"] = len(cleaned_docs)

    path = _persona_path(channel_input)
    path.write_text(json.dumps(persona, indent=2, ensure_ascii=False))
    logger.info(
        f"[PERSONA] Built for '{persona.get('display_name', channel_input)}' "
        f"→ {path}"
    )
    return persona


def load_persona(channel_input: str) -> Optional[dict]:
    """Load a previously built persona, or None if not found."""
    path = _persona_path(channel_input)
    if path.exists():
        return json.loads(path.read_text())
    return None
