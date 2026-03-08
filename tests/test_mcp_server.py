import asyncio
import json

import mcp_server as ms


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
