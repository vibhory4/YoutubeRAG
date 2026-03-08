import asyncio
import json

import mcp_server as ms
from tests.conftest import make_raising_class, make_session_file


def test_youtube_rag_query_success(monkeypatch):
    class DummyStore:
        def query(self, **kwargs):
            return {
                "documents": ["chunk"],
                "metadatas": [{"title": "Video", "video_url": "u", "video_id": "v", "chunk_index": 0}],
                "distances": [0.25],
            }

    monkeypatch.setattr(ms, "VectorStoreManager", lambda: DummyStore())
    params = ms.QueryInput(channel="@abc", query="q", n_results=3)
    out = asyncio.run(ms.youtube_rag_query(params))
    payload = json.loads(out)
    assert payload["status"] == "success"
    assert payload["results"][0]["relevance_score"] == 0.75


def test_youtube_rag_query_no_results(monkeypatch):
    class DummyStore:
        def query(self, **kwargs):
            return {"documents": [], "metadatas": [], "distances": []}

    monkeypatch.setattr(ms, "VectorStoreManager", lambda: DummyStore())
    out = asyncio.run(ms.youtube_rag_query(ms.QueryInput(channel="@abc", query="q")))
    assert json.loads(out)["status"] == "no_results"


def test_youtube_rag_index_channel(monkeypatch):
    monkeypatch.setattr(ms, "run_pipeline", lambda **kwargs: {"status": "ok", "new_video_count": 2, "clean_count": 2, "ingestion_stats": {"total_chunks": 5}, "errors": []})
    out = asyncio.run(ms.youtube_rag_index_channel(ms.IndexChannelInput(channel="@abc", limit=2)))
    assert json.loads(out)["status"] == "ok"


def test_youtube_rag_list_channels(monkeypatch):
    class S:
        def get_summary(self):
            return {"total_channels": 1}

    class V:
        def list_collections(self):
            return [{"name": "c1"}]

    monkeypatch.setattr(ms, "PipelineState", lambda: S())
    monkeypatch.setattr(ms, "VectorStoreManager", lambda: V())
    out = asyncio.run(ms.youtube_rag_list_channels())
    payload = json.loads(out)
    assert payload["status"] == "success"


def test_youtube_rag_channel_stats(monkeypatch):
    class V:
        def get_stats(self, channel):
            return {"channel": channel, "total_chunks": 1}

    class S:
        def get_channel_info(self, channel):
            return {"total_videos_indexed": 1}

    monkeypatch.setattr(ms, "VectorStoreManager", lambda: V())
    monkeypatch.setattr(ms, "PipelineState", lambda: S())
    out = asyncio.run(ms.youtube_rag_channel_stats(ms.ChannelInput(channel="@abc")))
    assert json.loads(out)["status"] == "success"


def test_youtube_rag_delete_channel(monkeypatch):
    class V:
        def delete_collection(self, channel):
            return True

    class S:
        def remove_channel(self, channel):
            self.channel = channel

    monkeypatch.setattr(ms, "VectorStoreManager", lambda: V())
    monkeypatch.setattr(ms, "PipelineState", lambda: S())
    out = asyncio.run(ms.youtube_rag_delete_channel(ms.ChannelInput(channel="@abc")))
    assert json.loads(out)["status"] == "deleted"


def test_get_channel_agent_not_indexed(monkeypatch):
    class S:
        def get_channel_info(self, channel):
            return None

    monkeypatch.setattr(ms, "PipelineState", lambda: S())
    out = asyncio.run(ms.get_channel_agent(ms.GetAgentInput(channel="@abc")))
    assert json.loads(out)["status"] == "not_indexed"


def test_get_channel_agent_ready(monkeypatch):
    class S:
        def get_channel_info(self, channel):
            return {"total_videos_indexed": 3}

    class A:
        def __init__(self, channel):
            self.session_id = "sess"
            self.messages = []

        def ensure_session_file(self):
            return None

    monkeypatch.setattr(ms, "PipelineState", lambda: S())
    monkeypatch.setitem(
        __import__("sys").modules,
        "channel_agent",
        type("M", (), {"ChannelAgent": A}),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "persona_builder",
        type("M", (), {"load_persona": lambda channel: {"display_name": "Name", "topics": ["x"], "tone": "t"}}),
    )

    out = asyncio.run(ms.get_channel_agent(ms.GetAgentInput(channel="@abc")))
    payload = json.loads(out)
    assert payload["status"] == "ready"
    assert payload["session_id"] == "sess"


def test_chat_with_channel_agent_session_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(ms.config, "sessions_dir", str(tmp_path))
    out = asyncio.run(ms.chat_with_channel_agent(ms.ChatInput(session_id="missing", message="hi")))
    assert json.loads(out)["status"] == "session_not_found"


def test_chat_with_channel_agent_success(monkeypatch, tmp_path):
    monkeypatch.setattr(ms.config, "sessions_dir", str(tmp_path))
    (tmp_path / "sess.json").write_text(json.dumps({"channel_input": "@abc"}))

    class A:
        def __init__(self, channel):
            self.channel = channel

        def chat(self, msg):
            return f"reply:{msg}"

    monkeypatch.setitem(__import__("sys").modules, "channel_agent", type("M", (), {"ChannelAgent": A}))

    out = asyncio.run(ms.chat_with_channel_agent(ms.ChatInput(session_id="sess", message="hello")))
    assert out == "reply:hello"


# ──────────────────────────────────────────
# Coverage gap tests — exception handlers (P0)
# ──────────────────────────────────────────

def test_youtube_rag_query_exception_returns_error_json(monkeypatch):
    monkeypatch.setattr(ms, "VectorStoreManager", make_raising_class())
    out = asyncio.run(ms.youtube_rag_query(ms.QueryInput(channel="@abc", query="q")))
    payload = json.loads(out)
    assert payload["status"] == "error"
    assert "boom" in payload["message"]


def test_youtube_rag_index_channel_exception_returns_error_json(monkeypatch):
    monkeypatch.setattr(ms, "run_pipeline", make_raising_class(msg="boom"))
    out = asyncio.run(ms.youtube_rag_index_channel(ms.IndexChannelInput(channel="@abc")))
    payload = json.loads(out)
    assert payload["status"] == "error"
    assert "boom" in payload["message"]


def test_youtube_rag_list_channels_exception_returns_error_json(monkeypatch):
    monkeypatch.setattr(ms, "PipelineState", make_raising_class())
    out = asyncio.run(ms.youtube_rag_list_channels())
    payload = json.loads(out)
    assert payload["status"] == "error"
    assert "boom" in payload["message"]


def test_youtube_rag_channel_stats_exception_returns_error_json(monkeypatch):
    monkeypatch.setattr(ms, "VectorStoreManager", make_raising_class())
    out = asyncio.run(ms.youtube_rag_channel_stats(ms.ChannelInput(channel="@abc")))
    payload = json.loads(out)
    assert payload["status"] == "error"
    assert "boom" in payload["message"]


def test_youtube_rag_delete_channel_exception_returns_error_json(monkeypatch):
    monkeypatch.setattr(ms, "VectorStoreManager", make_raising_class())
    out = asyncio.run(ms.youtube_rag_delete_channel(ms.ChannelInput(channel="@abc")))
    payload = json.loads(out)
    assert payload["status"] == "error"
    assert "boom" in payload["message"]


def test_get_channel_agent_exception_returns_error_json(monkeypatch):
    monkeypatch.setattr(ms, "PipelineState", make_raising_class())
    out = asyncio.run(ms.get_channel_agent(ms.GetAgentInput(channel="@abc")))
    payload = json.loads(out)
    assert payload["status"] == "error"
    assert "boom" in payload["message"]


def test_chat_with_channel_agent_exception_returns_error_json(monkeypatch, tmp_path):
    monkeypatch.setattr(ms.config, "sessions_dir", str(tmp_path))
    make_session_file(tmp_path, "sess", "@abc")
    monkeypatch.setitem(
        __import__("sys").modules,
        "channel_agent",
        type("M", (), {"ChannelAgent": make_raising_class()}),
    )
    out = asyncio.run(ms.chat_with_channel_agent(ms.ChatInput(session_id="sess", message="hi")))
    payload = json.loads(out)
    assert payload["status"] == "error"
    assert "boom" in payload["message"]


# ──────────────────────────────────────────
# MCP integration tests — response contracts (P1a)
# ──────────────────────────────────────────

def test_get_channel_agent_persona_absent(monkeypatch):
    class S:
        def get_channel_info(self, channel):
            return {"total_videos_indexed": 5}

    class A:
        def __init__(self, channel):
            self.session_id = "sess"
            self.messages = []

        def ensure_session_file(self):
            pass

    monkeypatch.setattr(ms, "PipelineState", lambda: S())
    monkeypatch.setitem(
        __import__("sys").modules, "channel_agent", type("M", (), {"ChannelAgent": A})
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "persona_builder",
        type("M", (), {"load_persona": lambda channel: None}),
    )

    out = asyncio.run(ms.get_channel_agent(ms.GetAgentInput(channel="@abc")))
    payload = json.loads(out)
    assert payload["status"] == "ready"
    assert payload["persona_available"] is False
    assert payload["display_name"] == "@abc"


def test_get_channel_agent_response_all_fields_present(monkeypatch):
    class S:
        def get_channel_info(self, channel):
            return {"total_videos_indexed": 3}

    class A:
        def __init__(self, channel):
            self.session_id = "sess"
            self.messages = []

        def ensure_session_file(self):
            pass

    monkeypatch.setattr(ms, "PipelineState", lambda: S())
    monkeypatch.setitem(
        __import__("sys").modules, "channel_agent", type("M", (), {"ChannelAgent": A})
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "persona_builder",
        type("M", (), {"load_persona": lambda ch: {"display_name": "X", "topics": ["t"], "tone": "casual"}}),
    )

    out = asyncio.run(ms.get_channel_agent(ms.GetAgentInput(channel="@abc")))
    payload = json.loads(out)
    expected_keys = {"status", "session_id", "channel", "display_name", "topics", "tone",
                     "videos_indexed", "session_messages", "persona_available", "instructions"}
    assert expected_keys <= set(payload.keys())


def test_chat_with_channel_agent_corrupted_session_json(monkeypatch, tmp_path):
    monkeypatch.setattr(ms.config, "sessions_dir", str(tmp_path))
    (tmp_path / "sess.json").write_text("not-json")
    out = asyncio.run(ms.chat_with_channel_agent(ms.ChatInput(session_id="sess", message="hi")))
    payload = json.loads(out)
    assert payload["status"] == "error"
    assert "Chat failed" in payload["message"]


def test_youtube_rag_query_response_format(monkeypatch):
    class DummyStore:
        def query(self, **kwargs):
            return {
                "documents": ["text1", "text2"],
                "metadatas": [
                    {"title": "T1", "video_url": "u1", "video_id": "v1", "chunk_index": 0},
                    {"title": "T2", "video_url": "u2", "video_id": "v2", "chunk_index": 1},
                ],
                "distances": [0.1, 0.5],
            }

    monkeypatch.setattr(ms, "VectorStoreManager", lambda: DummyStore())
    out = asyncio.run(ms.youtube_rag_query(ms.QueryInput(channel="@abc", query="q")))
    payload = json.loads(out)
    assert {"status", "query", "channel", "total_results", "results"} <= set(payload.keys())
    r = payload["results"][0]
    assert {"content", "video_title", "relevance_score"} <= set(r.keys())


def test_youtube_rag_delete_channel_not_found(monkeypatch):
    class V:
        def delete_collection(self, channel):
            return False

    class S:
        def remove_channel(self, channel):
            pass

    monkeypatch.setattr(ms, "VectorStoreManager", lambda: V())
    monkeypatch.setattr(ms, "PipelineState", lambda: S())
    out = asyncio.run(ms.youtube_rag_delete_channel(ms.ChannelInput(channel="@abc")))
    assert json.loads(out)["status"] == "not_found"


def test_youtube_rag_index_channel_response_format(monkeypatch):
    monkeypatch.setattr(
        ms, "run_pipeline",
        lambda **kw: {"status": "ok", "new_video_count": 1, "clean_count": 1,
                      "ingestion_stats": {"total_chunks": 3}, "errors": []},
    )
    out = asyncio.run(ms.youtube_rag_index_channel(ms.IndexChannelInput(channel="@abc")))
    payload = json.loads(out)
    assert {"status", "channel", "new_videos_found", "documents_cleaned", "ingestion_stats", "errors"} <= set(payload.keys())


def test_youtube_rag_channel_stats_response_format(monkeypatch):
    class V:
        def get_stats(self, ch):
            return {"total_chunks": 10}

    class S:
        def get_channel_info(self, ch):
            return {"total_videos_indexed": 2}

    monkeypatch.setattr(ms, "VectorStoreManager", lambda: V())
    monkeypatch.setattr(ms, "PipelineState", lambda: S())
    out = asyncio.run(ms.youtube_rag_channel_stats(ms.ChannelInput(channel="@abc")))
    payload = json.loads(out)
    assert "vector_store_stats" in payload
    assert "pipeline_state" in payload


def test_chat_with_channel_agent_missing_channel_input_key(monkeypatch, tmp_path):
    monkeypatch.setattr(ms.config, "sessions_dir", str(tmp_path))
    # Session file has no "channel_input" key — fallback is the session_id itself
    (tmp_path / "sess.json").write_text(json.dumps({}))

    constructed_with = []

    class A:
        def __init__(self, channel):
            constructed_with.append(channel)

        def chat(self, msg):
            return "ok"

    monkeypatch.setitem(__import__("sys").modules, "channel_agent", type("M", (), {"ChannelAgent": A}))
    asyncio.run(ms.chat_with_channel_agent(ms.ChatInput(session_id="sess", message="hi")))
    assert constructed_with == ["sess"]


def test_youtube_rag_list_channels_response_format(monkeypatch):
    class S:
        def get_summary(self):
            return {"total_channels": 0}

    class V:
        def list_collections(self):
            return []

    monkeypatch.setattr(ms, "PipelineState", lambda: S())
    monkeypatch.setattr(ms, "VectorStoreManager", lambda: V())
    out = asyncio.run(ms.youtube_rag_list_channels())
    payload = json.loads(out)
    assert "pipeline_summary" in payload
    assert "vector_store_collections" in payload


def test_youtube_rag_query_relevance_score_math(monkeypatch):
    class DummyStore:
        def query(self, **kwargs):
            return {
                "documents": ["d1", "d2"],
                "metadatas": [{"title": "A"}, {"title": "B"}],
                "distances": [0.0, 1.0],
            }

    monkeypatch.setattr(ms, "VectorStoreManager", lambda: DummyStore())
    out = asyncio.run(ms.youtube_rag_query(ms.QueryInput(channel="@abc", query="q")))
    results = json.loads(out)["results"]
    assert results[0]["relevance_score"] == 1.0
    assert results[1]["relevance_score"] == 0.0


# ──────────────────────────────────────────
# Robustness tests (P2)
# ──────────────────────────────────────────

def test_chat_with_channel_agent_empty_message(monkeypatch, tmp_path):
    monkeypatch.setattr(ms.config, "sessions_dir", str(tmp_path))
    make_session_file(tmp_path, "sess", "@abc")

    class A:
        def __init__(self, channel):
            pass

        def chat(self, msg):
            return "ok"

    monkeypatch.setitem(__import__("sys").modules, "channel_agent", type("M", (), {"ChannelAgent": A}))
    out = asyncio.run(ms.chat_with_channel_agent(ms.ChatInput(session_id="sess", message="")))
    assert out == "ok"
