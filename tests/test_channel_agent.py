"""
Comprehensive tests for channel_agent.py.

Covers:
- _normalize_handle()        — URL/handle normalisation, every branch
- _system_prompt()           — with/without persona, partial/empty persona fields
- _search_videos()           — results formatting, empty results, None documents,
                               full metadata fields, partial metadata
- _get_channel_stats()       — happy path, channel not in state (None info),
                               fallback from vs_stats
- chat()                     — no-tool path, single tool call, multiple tool calls
                               in one response, max-loop exhaustion (fallback),
                               get_channel_stats tool dispatch, on_tool_call callback
                               for search_videos, on_tool_call callback for stats,
                               empty user message, OpenAI exception propagation
- Session persistence        — _load_session (missing file, existing file, corrupt JSON),
                               _save_session fields, ensure_session_file (create /
                               no-overwrite), session reload across agent instances
- Sliding window             — trims at boundary, stabilises after repeated calls
- Adversarial                — malformed tool-call arguments JSON, unicode content,
                               very long channel names, all-whitespace input

All tests run offline: OpenAI client, VectorStoreManager, load_persona, and
PipelineState are monkey-patched — no network, no ChromaDB, no real API key needed.
"""

import json
import re
from unittest.mock import MagicMock, patch, call

import pytest

import channel_agent as ca


# ──────────────────────────────────────────────────────────────────────────────
# Test-doubles shared across this module
# ──────────────────────────────────────────────────────────────────────────────

class _DummyStore:
    """Minimal VectorStoreManager replacement."""

    def __init__(self, query_result=None, stats_result=None):
        self._query_result = query_result or {"documents": [], "metadatas": []}
        self._stats_result = stats_result or {
            "total_videos": 0,
            "total_chunks": 0,
            "video_ids": [],
        }
        self.query_calls: list = []
        self.stats_calls: list = []

    def query(self, *, channel_name, query_text, n_results=5):
        self.query_calls.append({"channel_name": channel_name, "query_text": query_text})
        return self._query_result

    def get_stats(self, channel_name: str) -> dict:
        self.stats_calls.append(channel_name)
        return self._stats_result


class _ToolCall:
    """Minimal OpenAI ToolCall stub."""

    def __init__(self, tc_id: str, name: str, args: dict):
        self.id = tc_id
        self.function = type("_F", (), {
            "arguments": json.dumps(args),
            "name": name,
        })()


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _Resp:
    def __init__(self, msg: _Msg):
        self.choices = [type("_C", (), {"message": msg})()]


def _make_openai_client(responses: list):
    """Build a minimal OpenAI client that returns responses in FIFO order."""

    class _Completions:
        def __init__(self, seq):
            self._seq = list(seq)

        def create(self, **kwargs):
            return self._seq.pop(0)

    class _Chat:
        def __init__(self, seq):
            self.completions = _Completions(seq)

    class _Client:
        def __init__(self, seq):
            self.chat = _Chat(seq)

    return _Client(responses)


def _make_pipeline_state(channel_info=None):
    """Return a MagicMock that behaves like PipelineState."""
    mock = MagicMock()
    mock.get_channel_info.return_value = channel_info
    return mock


def _build_agent(
    monkeypatch,
    tmp_path,
    channel="@creator",
    persona=None,
    query_result=None,
    stats_result=None,
    responses=None,
    pipeline_channel_info=None,
):
    """
    Construct a ChannelAgent with all external dependencies patched.

    Keyword args:
      channel               -- channel_input passed to ChannelAgent()
      persona               -- value returned by load_persona (None = no persona)
      query_result          -- dict returned by VectorStoreManager.query()
      stats_result          -- dict returned by VectorStoreManager.get_stats()
      responses             -- list of _Resp objects consumed by OpenAI client
      pipeline_channel_info -- dict returned by PipelineState.get_channel_info()
    """
    monkeypatch.setattr(ca.config, "sessions_dir", str(tmp_path))
    monkeypatch.setattr(ca, "load_persona", lambda _: persona)

    store = _DummyStore(
        query_result=query_result,
        stats_result=stats_result,
    )
    monkeypatch.setattr(ca, "VectorStoreManager", lambda: store)

    pipeline_mock = _make_pipeline_state(pipeline_channel_info)
    monkeypatch.setattr(ca, "PipelineState", lambda: pipeline_mock)

    if responses is None:
        responses = [_Resp(_Msg(content="default reply"))]
    monkeypatch.setattr(ca, "OpenAI", lambda: _make_openai_client(responses))

    return ca.ChannelAgent(channel)


# ──────────────────────────────────────────────────────────────────────────────
# 1. _normalize_handle — every branch
# ──────────────────────────────────────────────────────────────────────────────

class TestNormalizeHandle:
    """Unit tests for the module-level _normalize_handle() function."""

    def test_at_handle_strips_at_sign(self):
        result = ca._normalize_handle("@CreatorName")
        assert result == "creatorname"

    def test_plain_lowercase_handle_unchanged(self):
        assert ca._normalize_handle("creator") == "creator"

    def test_strips_full_youtube_url_with_www(self):
        result = ca._normalize_handle("https://www.youtube.com/@CreatorName")
        assert result == "creatorname"

    def test_strips_youtube_url_without_www(self):
        result = ca._normalize_handle("https://youtube.com/@handle")
        assert result == "handle"

    def test_leading_nonalnum_triggers_yt_prefix(self):
        # "-creator-" → after regex: "-creator-" → strip("_") = "-creator-"
        # first char "-" is not alnum → "yt_" prefix
        result = ca._normalize_handle("-creator-")
        assert result.startswith("yt_")

    def test_trailing_nonalnum_triggers_col_suffix(self):
        result = ca._normalize_handle("creator-")
        assert result.endswith("_col")

    def test_both_bad_ends_get_both_fixes(self):
        result = ca._normalize_handle("-creator-")
        assert result.startswith("yt_")
        assert result.endswith("_col")

    def test_internal_special_chars_replaced_with_underscore(self):
        result = ca._normalize_handle("my channel!name")
        assert "!" not in result
        assert " " not in result

    def test_truncated_to_50_chars(self):
        long_input = "a" * 100
        assert len(ca._normalize_handle(long_input)) <= 50

    def test_empty_string_gets_yt_prefix(self):
        result = ca._normalize_handle("")
        assert result.startswith("yt_")

    def test_whitespace_only_treated_as_empty(self):
        result = ca._normalize_handle("   ")
        assert result.startswith("yt_")

    def test_uppercase_lowercased(self):
        assert ca._normalize_handle("CREATOR") == "creator"

    def test_numeric_only_handle_valid(self):
        result = ca._normalize_handle("12345")
        assert result == "12345"

    def test_result_always_starts_with_alnum(self):
        for inp in ["@foo", "-bar", "!baz", "", "   ", "@", "https://www.youtube.com/@x"]:
            result = ca._normalize_handle(inp)
            assert result[0].isalnum(), f"Expected alnum start for input {inp!r}, got {result!r}"

    def test_result_always_ends_with_alnum(self):
        for inp in ["@foo", "creator-", "-bar-", "foo_"]:
            result = ca._normalize_handle(inp)
            assert result[-1].isalnum(), f"Expected alnum end for input {inp!r}, got {result!r}"

    def test_http_url_also_stripped(self):
        result = ca._normalize_handle("http://www.youtube.com/@handle")
        assert result == "handle"

    def test_unicode_channel_name_replaced(self):
        result = ca._normalize_handle("日本語チャンネル")
        # All non-ascii chars become underscores, which then get stripped/fixed
        assert re.match(r"^[a-z0-9]", result)

    def test_channel_url_with_path_suffix(self):
        # e.g. "https://www.youtube.com/@handle/videos" — the regex only strips up to first "/"
        result = ca._normalize_handle("https://www.youtube.com/@handle/videos")
        # "@handle/videos" → "@handle_videos" → "handle_videos"
        assert "handle" in result


# ──────────────────────────────────────────────────────────────────────────────
# 2. _system_prompt — with/without persona, edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestSystemPrompt:
    def test_no_persona_mentions_channel_input(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path, persona=None)
        prompt = agent._system_prompt()
        assert "@creator" in prompt
        assert "Answer questions based on your video content" in prompt

    def test_no_persona_uses_first_person(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path, persona=None)
        assert "first person" in agent._system_prompt()

    def test_with_persona_uses_display_name(self, monkeypatch, tmp_path):
        persona = {
            "display_name": "TechGuru",
            "topics": ["Python", "AI"],
            "tone": "educational",
            "style": "clear explanations",
            "persona_summary": "A great teacher.",
            "common_phrases": ["Let me show you", "Trust the process"],
        }
        agent = _build_agent(monkeypatch, tmp_path, persona=persona)
        prompt = agent._system_prompt()
        assert "TechGuru" in prompt

    def test_with_persona_includes_topics(self, monkeypatch, tmp_path):
        persona = {"display_name": "X", "topics": ["finance", "investing"], "tone": "formal",
                   "style": "", "persona_summary": "", "common_phrases": []}
        agent = _build_agent(monkeypatch, tmp_path, persona=persona)
        assert "finance, investing" in agent._system_prompt()

    def test_with_persona_includes_phrases(self, monkeypatch, tmp_path):
        persona = {"display_name": "X", "topics": [], "tone": "casual",
                   "style": "", "persona_summary": "", "common_phrases": ["yo what's up", "let's go"]}
        agent = _build_agent(monkeypatch, tmp_path, persona=persona)
        assert '"yo what\'s up"' in agent._system_prompt()

    def test_with_persona_limits_phrases_to_five(self, monkeypatch, tmp_path):
        phrases = [f"phrase{i}" for i in range(10)]
        persona = {"display_name": "X", "topics": [], "tone": "casual",
                   "style": "", "persona_summary": "", "common_phrases": phrases}
        agent = _build_agent(monkeypatch, tmp_path, persona=persona)
        prompt = agent._system_prompt()
        # Only first 5 should appear
        for i in range(5):
            assert f'"phrase{i}"' in prompt
        assert '"phrase5"' not in prompt

    def test_with_persona_no_phrases_no_phrase_block(self, monkeypatch, tmp_path):
        persona = {"display_name": "X", "topics": [], "tone": "casual",
                   "style": "", "persona_summary": "", "common_phrases": []}
        agent = _build_agent(monkeypatch, tmp_path, persona=persona)
        assert "You often say things like" not in agent._system_prompt()

    def test_partial_persona_missing_display_name_uses_channel_input(self, monkeypatch, tmp_path):
        # display_name missing — persona.get("display_name", self.channel_input) should fall back
        persona = {"topics": ["tech"], "tone": "casual", "style": "", "persona_summary": ""}
        agent = _build_agent(monkeypatch, tmp_path, persona=persona)
        prompt = agent._system_prompt()
        assert "@creator" in prompt

    def test_partial_persona_missing_topics_no_crash(self, monkeypatch, tmp_path):
        persona = {"display_name": "Creator", "tone": "casual"}
        agent = _build_agent(monkeypatch, tmp_path, persona=persona)
        prompt = agent._system_prompt()
        assert "Creator" in prompt
        assert isinstance(prompt, str) and len(prompt) > 10

    def test_all_empty_persona_fields_no_crash(self, monkeypatch, tmp_path):
        persona = {"display_name": "", "topics": [], "tone": "", "style": "",
                   "persona_summary": "", "common_phrases": []}
        agent = _build_agent(monkeypatch, tmp_path, persona=persona)
        prompt = agent._system_prompt()
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_system_prompt_is_a_string(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path)
        assert isinstance(agent._system_prompt(), str)


# ──────────────────────────────────────────────────────────────────────────────
# 3. _search_videos — result formatting, edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestSearchVideos:
    def test_empty_documents_returns_no_results_message(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path,
                             query_result={"documents": [], "metadatas": []})
        assert "No relevant content found" in agent._search_videos("q")

    def test_none_documents_returns_no_results_message(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path,
                             query_result={"documents": None, "metadatas": []})
        assert "No relevant content found" in agent._search_videos("q")

    def test_formats_result_number(self, monkeypatch, tmp_path):
        result = {"documents": ["text"], "metadatas": [{"title": "T"}]}
        agent = _build_agent(monkeypatch, tmp_path, query_result=result)
        assert "[Result 1]" in agent._search_videos("q")

    def test_formats_title_in_from_block(self, monkeypatch, tmp_path):
        result = {"documents": ["chunk"], "metadatas": [{"title": "My Video"}]}
        agent = _build_agent(monkeypatch, tmp_path, query_result=result)
        assert "[From: My Video]" in agent._search_videos("q")

    def test_includes_url_in_output(self, monkeypatch, tmp_path):
        result = {
            "documents": ["chunk"],
            "metadatas": [{"title": "T", "video_url": "https://youtu.be/abc"}],
        }
        agent = _build_agent(monkeypatch, tmp_path, query_result=result)
        assert "https://youtu.be/abc" in agent._search_videos("q")

    def test_includes_chunk_index_in_output(self, monkeypatch, tmp_path):
        result = {
            "documents": ["chunk"],
            "metadatas": [{"title": "T", "chunk_index": 7}],
        }
        agent = _build_agent(monkeypatch, tmp_path, query_result=result)
        assert "Chunk: 7" in agent._search_videos("q")

    def test_missing_optional_metadata_fields_use_defaults(self, monkeypatch, tmp_path):
        # No title, no video_url, no chunk_index in metadata
        result = {"documents": ["text"], "metadatas": [{}]}
        agent = _build_agent(monkeypatch, tmp_path, query_result=result)
        out = agent._search_videos("q")
        assert "Unknown video" in out
        assert "Chunk: 0" in out

    def test_multiple_results_separated_by_divider(self, monkeypatch, tmp_path):
        result = {
            "documents": ["d1", "d2"],
            "metadatas": [{"title": "A"}, {"title": "B"}],
        }
        agent = _build_agent(monkeypatch, tmp_path, query_result=result)
        out = agent._search_videos("q")
        assert "---" in out
        assert "[Result 1]" in out
        assert "[Result 2]" in out

    def test_query_forwarded_to_store(self, monkeypatch, tmp_path):
        store = _DummyStore(query_result={"documents": [], "metadatas": []})
        monkeypatch.setattr(ca.config, "sessions_dir", str(tmp_path))
        monkeypatch.setattr(ca, "load_persona", lambda _: None)
        monkeypatch.setattr(ca, "VectorStoreManager", lambda: store)
        monkeypatch.setattr(ca, "PipelineState", lambda: _make_pipeline_state())
        monkeypatch.setattr(ca, "OpenAI", lambda: _make_openai_client([_Resp(_Msg(content="x"))]))
        agent = ca.ChannelAgent("@creator")

        agent._search_videos("machine learning basics")
        assert store.query_calls[0]["query_text"] == "machine learning basics"

    def test_channel_name_forwarded_to_store(self, monkeypatch, tmp_path):
        store = _DummyStore(query_result={"documents": [], "metadatas": []})
        monkeypatch.setattr(ca.config, "sessions_dir", str(tmp_path))
        monkeypatch.setattr(ca, "load_persona", lambda _: None)
        monkeypatch.setattr(ca, "VectorStoreManager", lambda: store)
        monkeypatch.setattr(ca, "PipelineState", lambda: _make_pipeline_state())
        monkeypatch.setattr(ca, "OpenAI", lambda: _make_openai_client([_Resp(_Msg(content="x"))]))
        agent = ca.ChannelAgent("@mychannel")

        agent._search_videos("query")
        assert store.query_calls[0]["channel_name"] == "@mychannel"

    def test_unicode_query_handled(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path,
                             query_result={"documents": [], "metadatas": []})
        # Should not raise
        result = agent._search_videos("日本語のクエリ")
        assert "No relevant content found" in result


# ──────────────────────────────────────────────────────────────────────────────
# 4. _get_channel_stats — covers lines 169-181
# ──────────────────────────────────────────────────────────────────────────────

class TestGetChannelStats:
    def test_returns_json_string(self, monkeypatch, tmp_path):
        agent = _build_agent(
            monkeypatch, tmp_path,
            stats_result={"total_videos": 3, "total_chunks": 90, "video_ids": ["v1", "v2", "v3"]},
            pipeline_channel_info={"total_videos_indexed": 3, "last_checked": "2025-01-01"},
        )
        out = agent._get_channel_stats()
        parsed = json.loads(out)
        assert isinstance(parsed, dict)

    def test_includes_channel_field(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path)
        parsed = json.loads(agent._get_channel_stats())
        assert parsed["channel"] == "@creator"

    def test_videos_indexed_from_state_info_takes_priority(self, monkeypatch, tmp_path):
        agent = _build_agent(
            monkeypatch, tmp_path,
            stats_result={"total_videos": 2, "total_chunks": 40, "video_ids": []},
            pipeline_channel_info={"total_videos_indexed": 5, "last_checked": None},
        )
        parsed = json.loads(agent._get_channel_stats())
        assert parsed["videos_indexed"] == 5

    def test_videos_indexed_falls_back_to_vs_stats_when_state_none(self, monkeypatch, tmp_path):
        # PipelineState.get_channel_info() returns None → fallback to vs_stats["total_videos"]
        agent = _build_agent(
            monkeypatch, tmp_path,
            stats_result={"total_videos": 7, "total_chunks": 210, "video_ids": ["v1"]},
            pipeline_channel_info=None,  # channel not in state
        )
        parsed = json.loads(agent._get_channel_stats())
        assert parsed["videos_indexed"] == 7

    def test_chunks_indexed_comes_from_vs_stats(self, monkeypatch, tmp_path):
        agent = _build_agent(
            monkeypatch, tmp_path,
            stats_result={"total_videos": 1, "total_chunks": 55, "video_ids": []},
        )
        parsed = json.loads(agent._get_channel_stats())
        assert parsed["chunks_indexed"] == 55

    def test_known_video_ids_comes_from_vs_stats(self, monkeypatch, tmp_path):
        agent = _build_agent(
            monkeypatch, tmp_path,
            stats_result={"total_videos": 2, "total_chunks": 10, "video_ids": ["abc", "def"]},
        )
        parsed = json.loads(agent._get_channel_stats())
        assert parsed["known_video_ids"] == ["abc", "def"]

    def test_last_checked_from_state_info(self, monkeypatch, tmp_path):
        agent = _build_agent(
            monkeypatch, tmp_path,
            pipeline_channel_info={"total_videos_indexed": 1, "last_checked": "2025-06-01"},
        )
        parsed = json.loads(agent._get_channel_stats())
        assert parsed["last_checked"] == "2025-06-01"

    def test_last_checked_none_when_channel_missing(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path, pipeline_channel_info=None)
        parsed = json.loads(agent._get_channel_stats())
        assert parsed["last_checked"] is None

    def test_get_stats_called_with_channel_input(self, monkeypatch, tmp_path):
        store = _DummyStore(stats_result={"total_videos": 0, "total_chunks": 0, "video_ids": []})
        monkeypatch.setattr(ca.config, "sessions_dir", str(tmp_path))
        monkeypatch.setattr(ca, "load_persona", lambda _: None)
        monkeypatch.setattr(ca, "VectorStoreManager", lambda: store)
        monkeypatch.setattr(ca, "PipelineState", lambda: _make_pipeline_state(None))
        monkeypatch.setattr(ca, "OpenAI", lambda: _make_openai_client([_Resp(_Msg(content="x"))]))
        agent = ca.ChannelAgent("@testchannel")
        agent._get_channel_stats()
        assert store.stats_calls == ["@testchannel"]

    def test_transcript_videos_in_vector_store_field_present(self, monkeypatch, tmp_path):
        agent = _build_agent(
            monkeypatch, tmp_path,
            stats_result={"total_videos": 4, "total_chunks": 120, "video_ids": []},
        )
        parsed = json.loads(agent._get_channel_stats())
        assert "transcript_videos_in_vector_store" in parsed
        assert parsed["transcript_videos_in_vector_store"] == 4


# ──────────────────────────────────────────────────────────────────────────────
# 5. chat() — every execution path
# ──────────────────────────────────────────────────────────────────────────────

class TestChat:

    # ── No-tool path ────────────────────────────────────────────────────────

    def test_no_tool_call_returns_content(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path,
                             responses=[_Resp(_Msg(content="Hello there!"))])
        assert agent.chat("Hi") == "Hello there!"

    def test_no_tool_call_appends_user_and_assistant_messages(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path,
                             responses=[_Resp(_Msg(content="reply"))])
        agent.chat("question")
        assert agent.messages[0] == {"role": "user", "content": "question"}
        assert agent.messages[1] == {"role": "assistant", "content": "reply"}

    def test_empty_user_message_accepted(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path,
                             responses=[_Resp(_Msg(content="ok"))])
        assert agent.chat("") == "ok"
        assert agent.messages[0]["content"] == ""

    # ── Single tool call ─────────────────────────────────────────────────────

    def test_single_search_videos_tool_call_resolves(self, monkeypatch, tmp_path):
        responses = [
            _Resp(_Msg(tool_calls=[_ToolCall("tc1", "search_videos", {"query": "SIP"})])),
            _Resp(_Msg(content="SIP stands for...")),
        ]
        agent = _build_agent(monkeypatch, tmp_path,
                             query_result={"documents": ["ctx"], "metadatas": [{"title": "V"}]},
                             responses=responses)
        assert agent.chat("Explain SIP") == "SIP stands for..."

    def test_single_tool_call_history_has_two_messages(self, monkeypatch, tmp_path):
        responses = [
            _Resp(_Msg(tool_calls=[_ToolCall("tc1", "search_videos", {"query": "q"})])),
            _Resp(_Msg(content="answer")),
        ]
        agent = _build_agent(monkeypatch, tmp_path, responses=responses)
        agent.chat("ask")
        # user + assistant = 2 messages in self.messages
        assert len(agent.messages) == 2

    # ── get_channel_stats tool dispatch (lines 251-254) ──────────────────────

    def test_get_channel_stats_tool_dispatch_returns_json(self, monkeypatch, tmp_path):
        """chat() must call _get_channel_stats when the model requests it."""
        responses = [
            _Resp(_Msg(tool_calls=[_ToolCall("tc1", "get_channel_stats", {})])),
            _Resp(_Msg(content="You have 5 videos")),
        ]
        agent = _build_agent(
            monkeypatch, tmp_path,
            stats_result={"total_videos": 5, "total_chunks": 50, "video_ids": []},
            responses=responses,
        )
        reply = agent.chat("How many videos do you have?")
        assert reply == "You have 5 videos"

    def test_get_channel_stats_tool_calls_store_get_stats(self, monkeypatch, tmp_path):
        store = _DummyStore(stats_result={"total_videos": 3, "total_chunks": 30, "video_ids": []})
        monkeypatch.setattr(ca.config, "sessions_dir", str(tmp_path))
        monkeypatch.setattr(ca, "load_persona", lambda _: None)
        monkeypatch.setattr(ca, "VectorStoreManager", lambda: store)
        monkeypatch.setattr(ca, "PipelineState", lambda: _make_pipeline_state(None))
        responses = [
            _Resp(_Msg(tool_calls=[_ToolCall("tc1", "get_channel_stats", {})])),
            _Resp(_Msg(content="done")),
        ]
        monkeypatch.setattr(ca, "OpenAI", lambda: _make_openai_client(responses))
        agent = ca.ChannelAgent("@creator")
        agent.chat("stats?")
        assert len(store.stats_calls) == 1

    # ── on_tool_call callback ────────────────────────────────────────────────

    def test_on_tool_call_fired_for_search_videos(self, monkeypatch, tmp_path):
        """on_tool_call(query, result) is called for search_videos — line 260."""
        callback_calls: list = []

        responses = [
            _Resp(_Msg(tool_calls=[_ToolCall("tc1", "search_videos", {"query": "topic"})])),
            _Resp(_Msg(content="found it")),
        ]
        agent = _build_agent(
            monkeypatch, tmp_path,
            query_result={"documents": ["doc"], "metadatas": [{"title": "V"}]},
            responses=responses,
        )
        agent.on_tool_call = lambda q, r: callback_calls.append((q, r))
        agent.chat("tell me about topic")

        assert len(callback_calls) == 1
        query_arg, result_arg = callback_calls[0]
        assert query_arg == "topic"
        assert "doc" in result_arg

    def test_on_tool_call_fired_for_get_channel_stats(self, monkeypatch, tmp_path):
        """on_tool_call("__channel_stats__", result) is called for stats tool — lines 253-254."""
        callback_calls: list = []

        responses = [
            _Resp(_Msg(tool_calls=[_ToolCall("tc1", "get_channel_stats", {})])),
            _Resp(_Msg(content="stats reply")),
        ]
        agent = _build_agent(
            monkeypatch, tmp_path,
            stats_result={"total_videos": 1, "total_chunks": 5, "video_ids": []},
            responses=responses,
        )
        agent.on_tool_call = lambda q, r: callback_calls.append((q, r))
        agent.chat("show stats")

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "__channel_stats__"
        parsed = json.loads(callback_calls[0][1])
        assert "channel" in parsed

    def test_on_tool_call_not_set_does_not_raise(self, monkeypatch, tmp_path):
        """on_tool_call is None by default — no AttributeError should occur."""
        responses = [
            _Resp(_Msg(tool_calls=[_ToolCall("tc1", "search_videos", {"query": "q"})])),
            _Resp(_Msg(content="fine")),
        ]
        agent = _build_agent(monkeypatch, tmp_path, responses=responses)
        assert agent.on_tool_call is None
        # Should not raise
        assert agent.chat("hello") == "fine"

    # ── Multiple tool calls in one response ──────────────────────────────────

    def test_multiple_tool_calls_in_single_response(self, monkeypatch, tmp_path):
        query_log: list = []

        class _TrackingStore:
            def query(self, *, channel_name, query_text, n_results=5):
                query_log.append(query_text)
                return {"documents": ["ctx"], "metadatas": [{"title": "V"}]}

            def get_stats(self, _):
                return {"total_videos": 0, "total_chunks": 0, "video_ids": []}

        monkeypatch.setattr(ca.config, "sessions_dir", str(tmp_path))
        monkeypatch.setattr(ca, "load_persona", lambda _: None)
        monkeypatch.setattr(ca, "VectorStoreManager", lambda: _TrackingStore())
        monkeypatch.setattr(ca, "PipelineState", lambda: _make_pipeline_state())
        responses = [
            _Resp(_Msg(tool_calls=[
                _ToolCall("tc1", "search_videos", {"query": "first"}),
                _ToolCall("tc2", "search_videos", {"query": "second"}),
            ])),
            _Resp(_Msg(content="combined answer")),
        ]
        monkeypatch.setattr(ca, "OpenAI", lambda: _make_openai_client(responses))
        agent = ca.ChannelAgent("@creator")
        reply = agent.chat("multi tool")
        assert reply == "combined answer"
        assert query_log == ["first", "second"]

    # ── Max tool loop exhaustion ──────────────────────────────────────────────

    def test_max_loop_exhaustion_returns_fallback(self, monkeypatch, tmp_path):
        looping = [
            _Resp(_Msg(tool_calls=[_ToolCall(f"tc{i}", "search_videos", {"query": "q"})]))
            for i in range(ca._MAX_TOOL_LOOPS)
        ]
        agent = _build_agent(monkeypatch, tmp_path, responses=looping)
        reply = agent.chat("hi")
        assert "trouble pulling" in reply

    def test_max_loop_exhaustion_still_saves_session(self, monkeypatch, tmp_path):
        looping = [
            _Resp(_Msg(tool_calls=[_ToolCall(f"tc{i}", "search_videos", {"query": "q"})]))
            for i in range(ca._MAX_TOOL_LOOPS)
        ]
        agent = _build_agent(monkeypatch, tmp_path, responses=looping)
        agent.chat("hi")
        path = tmp_path / f"{agent.session_id}.json"
        assert path.exists()

    def test_max_loop_exhaustion_appends_assistant_fallback_message(self, monkeypatch, tmp_path):
        looping = [
            _Resp(_Msg(tool_calls=[_ToolCall(f"tc{i}", "search_videos", {"query": "q"})]))
            for i in range(ca._MAX_TOOL_LOOPS)
        ]
        agent = _build_agent(monkeypatch, tmp_path, responses=looping)
        agent.chat("hi")
        last_msg = agent.messages[-1]
        assert last_msg["role"] == "assistant"
        assert "trouble pulling" in last_msg["content"]

    # ── Exception propagation ─────────────────────────────────────────────────

    def test_openai_exception_propagates(self, monkeypatch, tmp_path):
        class _BoomClient:
            class _Chat:
                class _Completions:
                    def create(self, **kwargs):
                        raise RuntimeError("API down")
                completions = _Completions()
            chat = _Chat()

        monkeypatch.setattr(ca.config, "sessions_dir", str(tmp_path))
        monkeypatch.setattr(ca, "load_persona", lambda _: None)
        monkeypatch.setattr(ca, "VectorStoreManager",
                            lambda: _DummyStore())
        monkeypatch.setattr(ca, "PipelineState", lambda: _make_pipeline_state())
        monkeypatch.setattr(ca, "OpenAI", lambda: _BoomClient())
        agent = ca.ChannelAgent("@creator")
        with pytest.raises(RuntimeError, match="API down"):
            agent.chat("hello")

    # ── Multi-turn context ────────────────────────────────────────────────────

    def test_multi_turn_context_preserved(self, monkeypatch, tmp_path):
        responses = [_Resp(_Msg(content="r1")), _Resp(_Msg(content="r2"))]
        agent = _build_agent(monkeypatch, tmp_path, responses=responses)
        agent.chat("q1")
        agent.chat("q2")
        assert agent.messages[0]["content"] == "q1"
        assert agent.messages[1]["content"] == "r1"
        assert agent.messages[2]["content"] == "q2"
        assert agent.messages[3]["content"] == "r2"

    # ── Tool call with missing query key in args ──────────────────────────────

    def test_search_videos_called_with_empty_query_when_key_missing(self, monkeypatch, tmp_path):
        """args.get("query", "") falls back to "" when the model omits the key."""
        store = _DummyStore(query_result={"documents": [], "metadatas": []})
        monkeypatch.setattr(ca.config, "sessions_dir", str(tmp_path))
        monkeypatch.setattr(ca, "load_persona", lambda _: None)
        monkeypatch.setattr(ca, "VectorStoreManager", lambda: store)
        monkeypatch.setattr(ca, "PipelineState", lambda: _make_pipeline_state())
        # ToolCall with empty args dict (no "query" key)
        responses = [
            _Resp(_Msg(tool_calls=[_ToolCall("tc1", "search_videos", {})])),
            _Resp(_Msg(content="ok")),
        ]
        monkeypatch.setattr(ca, "OpenAI", lambda: _make_openai_client(responses))
        agent = ca.ChannelAgent("@creator")
        reply = agent.chat("search for nothing")
        assert reply == "ok"
        assert store.query_calls[0]["query_text"] == ""


# ──────────────────────────────────────────────────────────────────────────────
# 6. Session persistence — _load_session, _save_session, ensure_session_file
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionPersistence:

    def test_load_session_returns_empty_list_when_no_file(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path)
        # No file on disk → messages list should be empty at construction
        assert agent.messages == []

    def test_load_session_reads_existing_messages(self, monkeypatch, tmp_path):
        saved = [{"role": "user", "content": "prev"}]
        (tmp_path / "creator.json").write_text(
            json.dumps({"channel_input": "@creator", "messages": saved})
        )
        agent = _build_agent(monkeypatch, tmp_path)
        assert agent.messages == saved

    def test_load_session_corrupt_json_raises(self, monkeypatch, tmp_path):
        (tmp_path / "creator.json").write_text("not-json{{{")
        monkeypatch.setattr(ca.config, "sessions_dir", str(tmp_path))
        monkeypatch.setattr(ca, "load_persona", lambda _: None)
        monkeypatch.setattr(ca, "VectorStoreManager", lambda: _DummyStore())
        monkeypatch.setattr(ca, "PipelineState", lambda: _make_pipeline_state())
        monkeypatch.setattr(ca, "OpenAI", lambda: _make_openai_client([]))
        with pytest.raises(json.JSONDecodeError):
            ca.ChannelAgent("@creator")

    def test_save_session_writes_channel_input(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path,
                             responses=[_Resp(_Msg(content="ok"))])
        agent.chat("hello")
        path = tmp_path / f"{agent.session_id}.json"
        data = json.loads(path.read_text())
        assert data["channel_input"] == "@creator"

    def test_save_session_writes_session_id(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path,
                             responses=[_Resp(_Msg(content="ok"))])
        agent.chat("hello")
        path = tmp_path / f"{agent.session_id}.json"
        data = json.loads(path.read_text())
        assert data["session_id"] == agent.session_id

    def test_save_session_writes_last_active(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path,
                             responses=[_Resp(_Msg(content="ok"))])
        agent.chat("hello")
        path = tmp_path / f"{agent.session_id}.json"
        data = json.loads(path.read_text())
        assert "last_active" in data

    def test_save_session_writes_messages(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path,
                             responses=[_Resp(_Msg(content="reply"))])
        agent.chat("user message")
        path = tmp_path / f"{agent.session_id}.json"
        data = json.loads(path.read_text())
        assert data["messages"][0] == {"role": "user", "content": "user message"}

    def test_save_session_valid_utf8_unicode(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path,
                             responses=[_Resp(_Msg(content="回答"))])
        agent.chat("日本語")
        path = tmp_path / f"{agent.session_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["messages"][0]["content"] == "日本語"
        assert data["messages"][1]["content"] == "回答"

    def test_ensure_session_file_creates_if_missing(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path)
        path = tmp_path / f"{agent.session_id}.json"
        if path.exists():
            path.unlink()
        agent.ensure_session_file()
        assert path.exists()

    def test_ensure_session_file_does_not_overwrite_existing(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path,
                             responses=[_Resp(_Msg(content="ok"))])
        agent.chat("hello")
        path = tmp_path / f"{agent.session_id}.json"
        mtime_before = path.stat().st_mtime
        agent.ensure_session_file()
        assert path.stat().st_mtime == mtime_before

    def test_ensure_session_file_writes_empty_messages(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path)
        path = tmp_path / f"{agent.session_id}.json"
        if path.exists():
            path.unlink()
        agent.ensure_session_file()
        data = json.loads(path.read_text())
        assert data["messages"] == []

    def test_ensure_session_file_has_created_at_field(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path)
        path = tmp_path / f"{agent.session_id}.json"
        if path.exists():
            path.unlink()
        agent.ensure_session_file()
        data = json.loads(path.read_text())
        assert "created_at" in data

    def test_session_reload_across_agent_instances(self, monkeypatch, tmp_path):
        agent1 = _build_agent(monkeypatch, tmp_path,
                              responses=[_Resp(_Msg(content="reply1"))])
        agent1.chat("hello")

        agent2 = _build_agent(monkeypatch, tmp_path,
                              responses=[_Resp(_Msg(content="reply2"))])
        assert len(agent2.messages) == 2
        assert agent2.messages[0]["content"] == "hello"
        assert agent2.messages[1]["content"] == "reply1"

    def test_session_path_uses_session_id_as_filename(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path)
        expected = tmp_path / f"{agent.session_id}.json"
        assert agent._session_path() == expected


# ──────────────────────────────────────────────────────────────────────────────
# 7. Sliding window
# ──────────────────────────────────────────────────────────────────────────────

class TestSlidingWindow:

    def test_trims_exactly_at_boundary(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path,
                             responses=[_Resp(_Msg(content="ok"))])
        # Pre-fill to MAX_HISTORY - 1 messages; one chat() adds 2 → trim fires
        agent.messages = [{"role": "user", "content": "x"}] * (ca._MAX_HISTORY - 1)
        agent.chat("trigger")
        assert len(agent.messages) == ca._MAX_HISTORY

    def test_does_not_trim_below_boundary(self, monkeypatch, tmp_path):
        """Two messages added when history was MAX_HISTORY - 2 → exactly MAX_HISTORY; no trim."""
        agent = _build_agent(monkeypatch, tmp_path,
                             responses=[_Resp(_Msg(content="ok"))])
        agent.messages = [{"role": "user", "content": "x"}] * (ca._MAX_HISTORY - 2)
        agent.chat("trigger")
        assert len(agent.messages) == ca._MAX_HISTORY

    def test_repeated_calls_never_exceed_max_history(self, monkeypatch, tmp_path):
        n = 15
        responses = [_Resp(_Msg(content="ok"))] * n
        agent = _build_agent(monkeypatch, tmp_path, responses=responses)
        for _ in range(n):
            agent.chat("msg")
        assert len(agent.messages) <= ca._MAX_HISTORY

    def test_trimmed_messages_are_oldest(self, monkeypatch, tmp_path):
        """After trimming, the kept messages should be the most recent ones."""
        agent = _build_agent(monkeypatch, tmp_path,
                             responses=[_Resp(_Msg(content="new_reply"))])
        # Flood with old messages
        agent.messages = [{"role": "user", "content": f"old_{i}"} for i in range(ca._MAX_HISTORY - 1)]
        agent.chat("new_question")
        # The last message is the new assistant reply
        assert agent.messages[-1]["content"] == "new_reply"
        # The first message is NOT the oldest ("old_0")
        assert agent.messages[0]["content"] != "old_0"


# ──────────────────────────────────────────────────────────────────────────────
# 8. Adversarial / edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestAdversarial:

    def test_malformed_tool_call_arguments_json_raises(self, monkeypatch, tmp_path):
        class _BadToolCall:
            id = "tc1"
            function = type("_F", (), {"arguments": "<<<not json>>>", "name": "search_videos"})()

        responses = [_Resp(_Msg(tool_calls=[_BadToolCall()]))]
        agent = _build_agent(monkeypatch, tmp_path, responses=responses)
        with pytest.raises(json.JSONDecodeError):
            agent.chat("hi")

    def test_very_long_user_message(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path,
                             responses=[_Resp(_Msg(content="ok"))])
        big_msg = "word " * 10_000
        reply = agent.chat(big_msg)
        assert reply == "ok"

    def test_unicode_user_message(self, monkeypatch, tmp_path):
        agent = _build_agent(monkeypatch, tmp_path,
                             responses=[_Resp(_Msg(content="ok"))])
        reply = agent.chat("こんにちは、元気ですか？")
        assert reply == "ok"

    def test_very_long_channel_name_truncated(self):
        long_name = "@" + "a" * 200
        result = ca._normalize_handle(long_name)
        assert len(result) <= 50

    def test_channel_name_with_only_special_chars(self):
        result = ca._normalize_handle("!@#$%")
        # Everything replaced → all underscores → start/end fixes applied
        assert result[0].isalnum()
        assert result[-1].isalnum()

    def test_search_videos_store_returns_mismatched_lengths(self, monkeypatch, tmp_path):
        """More documents than metadatas — zip() stops at shorter list; must not crash."""
        result = {
            "documents": ["d1", "d2", "d3"],
            "metadatas": [{"title": "T1"}],  # shorter
        }
        agent = _build_agent(monkeypatch, tmp_path, query_result=result)
        out = agent._search_videos("q")
        # Only first pair should appear
        assert "[Result 1]" in out
        assert "[Result 2]" not in out

    def test_chat_with_none_tool_call_name_defaults_to_search_videos(self, monkeypatch, tmp_path):
        """getattr(tc.function, 'name', 'search_videos') fallback when name attribute absent."""

        class _NoNameToolCall:
            id = "tc1"
            # function has no 'name' attribute
            function = type("_F", (), {"arguments": json.dumps({"query": "test"})})()

        responses = [
            _Resp(_Msg(tool_calls=[_NoNameToolCall()])),
            _Resp(_Msg(content="fallback search worked")),
        ]
        agent = _build_agent(
            monkeypatch, tmp_path,
            query_result={"documents": ["found"], "metadatas": [{"title": "T"}]},
            responses=responses,
        )
        reply = agent.chat("find something")
        assert reply == "fallback search worked"
