"""
Channel agent — a conversational AI that talks like a YouTube creator.

Each ChannelAgent wraps gpt-4o-mini with:
  - A persona-aware system prompt (built at index time)
  - A single RAG tool: search_videos → ChromaDB
  - Persistent per-channel session history (data/sessions/{handle}.json)
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config import config
from persona_builder import load_persona
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

_MAX_HISTORY = 20   # messages kept (= 10 conversation turns)
_MAX_TOOL_LOOPS = 3  # max RAG lookups per user turn


def _normalize_handle(channel_input: str) -> str:
    """Normalize a channel URL/handle to a safe session ID."""
    name = channel_input.strip().lower()
    name = re.sub(r"https?://[^/]*youtube\.com/", "", name)
    name = re.sub(r"[^a-z0-9_-]", "_", name)
    name = name.strip("_")[:50]
    if not name or not name[0].isalnum():
        name = "yt_" + name
    if not name[-1].isalnum():
        name = name + "_col"
    return name


class ChannelAgent:
    """Conversational agent backed by a YouTube channel's indexed transcripts."""

    def __init__(self, channel_input: str):
        self.channel_input = channel_input
        self.session_id = _normalize_handle(channel_input)
        self.persona: Optional[dict] = load_persona(channel_input)
        self.store = VectorStoreManager()
        self.client = OpenAI()
        self.messages: list[dict] = self._load_session()
        self.on_tool_call = None  # Optional[Callable[[str, str], None]] — (query, result)

    # ──────────────────────────────────────────
    # Session persistence
    # ──────────────────────────────────────────

    def _session_path(self) -> Path:
        return Path(config.sessions_dir) / f"{self.session_id}.json"

    def _load_session(self) -> list[dict]:
        path = self._session_path()
        if path.exists():
            data = json.loads(path.read_text())
            return data.get("messages", [])
        return []

    def _save_session(self):
        path = self._session_path()
        path.write_text(json.dumps({
            "channel_input": self.channel_input,
            "session_id": self.session_id,
            "last_active": datetime.utcnow().isoformat(),
            "messages": self.messages,
        }, indent=2, ensure_ascii=False))

    def ensure_session_file(self):
        """Write a session file (with empty history) if one doesn't exist yet.

        Called from get_channel_agent MCP tool so that chat_with_channel_agent
        can always resolve session_id → channel_input even before the first chat.
        """
        path = self._session_path()
        if not path.exists():
            path.write_text(json.dumps({
                "channel_input": self.channel_input,
                "session_id": self.session_id,
                "created_at": datetime.utcnow().isoformat(),
                "last_active": datetime.utcnow().isoformat(),
                "messages": [],
            }, indent=2, ensure_ascii=False))

    # ──────────────────────────────────────────
    # System prompt
    # ──────────────────────────────────────────

    def _system_prompt(self) -> str:
        p = self.persona
        if not p:
            return (
                f"You are a YouTube creator from the channel '{self.channel_input}'. "
                "Answer questions based on your video content. "
                "Use first person and speak naturally. "
                "Search your videos when you need specific information."
            )

        topics = ", ".join(p.get("topics", []))
        phrases = p.get("common_phrases", [])
        phrase_block = ""
        if phrases:
            phrase_block = (
                "\nYou often say things like: "
                + ", ".join(f'"{ph}"' for ph in phrases[:5])
            )

        return f"""You are {p.get("display_name", self.channel_input)}, a YouTube creator.

Your content focuses on: {topics}
Tone: {p.get("tone", "conversational")}
Style: {p.get("style", "")}
{p.get("persona_summary", "")}
{phrase_block}

When answering:
- Search your videos when you need specific information to answer accurately
- Respond in first person, in your natural voice
- Reference your videos naturally ("In my video about...", "I talked about this when...", "I covered this in...")
- Stay in character — match your tone and style throughout
- If something isn't covered in your videos, say so honestly rather than making things up"""

    # ──────────────────────────────────────────
    # RAG tool
    # ──────────────────────────────────────────

    def _search_videos(self, query: str) -> str:
        results = self.store.query(
            channel_name=self.channel_input,
            query_text=query,
            n_results=5,
        )
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        if not docs:
            return "No relevant content found in your indexed videos for this query."

        parts = []
        for doc, meta in zip(docs, metas):
            title = meta.get("title", "Unknown video")
            parts.append(f"[From: {title}]\n{doc}")

        return "\n\n---\n\n".join(parts)

    # ──────────────────────────────────────────
    # Chat
    # ──────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """Send a message and get a response in the creator's voice."""
        self.messages.append({"role": "user", "content": user_message})

        tools = [{
            "type": "function",
            "function": {
                "name": "search_videos",
                "description": (
                    "Search your indexed video transcripts for relevant content. "
                    "Call this when you need specific facts, quotes, or context "
                    "from your videos to answer the question."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant transcript content",
                        }
                    },
                    "required": ["query"],
                },
            },
        }]

        # Build the full context: system + history
        loop_messages = [
            {"role": "system", "content": self._system_prompt()}
        ] + self.messages

        reply = None
        for _ in range(_MAX_TOOL_LOOPS):
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=loop_messages,
                tools=tools,
                temperature=0.7,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                # Execute each tool call and feed results back
                loop_messages.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    logger.debug(f"[AGENT] search_videos('{args['query']}')")
                    result = self._search_videos(args["query"])
                    if self.on_tool_call:
                        self.on_tool_call(args["query"], result)
                    loop_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                reply = msg.content
                break

        if reply is None:
            reply = (
                "Sorry, I had trouble pulling the right information. "
                "Could you try rephrasing your question?"
            )

        self.messages.append({"role": "assistant", "content": reply})

        # Sliding window: keep last MAX_HISTORY messages
        if len(self.messages) > _MAX_HISTORY:
            self.messages = self.messages[-_MAX_HISTORY:]

        self._save_session()
        return reply
