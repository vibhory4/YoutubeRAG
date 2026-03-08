import json

import channel_agent as ca


class DummyStore:
    def __init__(self, result):
        self.result = result

    def query(self, **kwargs):
        return self.result


class ToolCall:
    def __init__(self, id_, args):
        self.id = id_
        self.function = type("F", (), {"arguments": json.dumps(args)})()


class Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class Resp:
    def __init__(self, msg):
        self.choices = [type("C", (), {"message": msg})()]


def _build_agent(monkeypatch, tmp_path, persona=None, store_result=None, responses=None):
    monkeypatch.setattr(ca.config, "sessions_dir", str(tmp_path))
    monkeypatch.setattr(ca, "load_persona", lambda _: persona)
    monkeypatch.setattr(ca, "VectorStoreManager", lambda: DummyStore(store_result or {"documents": [], "metadatas": []}))

    responses = responses or [Resp(Msg(content="final answer"))]

    class DummyClient:
        class Chat:
            class Completions:
                def __init__(self, seq):
                    self.seq = seq

                def create(self, **kwargs):
                    return self.seq.pop(0)

            def __init__(self, seq):
                self.completions = DummyClient.Chat.Completions(seq)

        def __init__(self, seq):
            self.chat = DummyClient.Chat(seq)

    monkeypatch.setattr(ca, "OpenAI", lambda: DummyClient(list(responses)))
    return ca.ChannelAgent("@creator")


def test_system_prompt_without_persona(monkeypatch, tmp_path):
    agent = _build_agent(monkeypatch, tmp_path, persona=None)
    prompt = agent._system_prompt()
    assert "@creator" in prompt
    assert "Answer questions based on your video content" in prompt


def test_system_prompt_with_persona(monkeypatch, tmp_path):
    persona = {
        "display_name": "Creator",
        "topics": ["finance", "investing"],
        "tone": "analytical",
        "style": "clear",
        "persona_summary": "summary",
        "common_phrases": ["phrase1"],
    }
    agent = _build_agent(monkeypatch, tmp_path, persona=persona)
    prompt = agent._system_prompt()
    assert "Creator" in prompt
    assert "finance, investing" in prompt
    assert '"phrase1"' in prompt


def test_search_videos_no_results(monkeypatch, tmp_path):
    agent = _build_agent(monkeypatch, tmp_path, store_result={"documents": [], "metadatas": []})
    out = agent._search_videos("q")
    assert "No relevant content found" in out


def test_search_videos_formats_results(monkeypatch, tmp_path):
    result = {
        "documents": ["chunk1", "chunk2"],
        "metadatas": [{"title": "T1"}, {"title": "T2"}],
    }
    agent = _build_agent(monkeypatch, tmp_path, store_result=result)
    out = agent._search_videos("q")
    assert "[From: T1]" in out
    assert "chunk2" in out


def test_chat_with_tool_call_and_session_save(monkeypatch, tmp_path):
    responses = [
        Resp(Msg(tool_calls=[ToolCall("tc1", {"query": "what is sip"})])),
        Resp(Msg(content="SIP is ...")),
    ]
    result = {"documents": ["ctx"], "metadatas": [{"title": "Video"}]}
    agent = _build_agent(monkeypatch, tmp_path, store_result=result, responses=responses)

    reply = agent.chat("Explain SIP")
    assert reply == "SIP is ..."
    assert len(agent.messages) == 2

    path = tmp_path / f"{agent.session_id}.json"
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["messages"][0]["role"] == "user"


def test_chat_fallback_after_tool_loop_limit(monkeypatch, tmp_path):
    looping = [Resp(Msg(tool_calls=[ToolCall("tc", {"query": "q"})])) for _ in range(ca._MAX_TOOL_LOOPS)]
    agent = _build_agent(monkeypatch, tmp_path, responses=looping)

    reply = agent.chat("hi")
    assert "trouble pulling" in reply


def test_ensure_session_file_creates_if_missing(monkeypatch, tmp_path):
    agent = _build_agent(monkeypatch, tmp_path)
    path = tmp_path / f"{agent.session_id}.json"
    if path.exists():
        path.unlink()
    agent.ensure_session_file()
    assert path.exists()
