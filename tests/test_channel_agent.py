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


def _make_client(responses):
    """Build a DummyClient from a list of Resp objects (used when _build_agent is not convenient)."""
    class DummyClient:
        class Chat:
            class Completions:
                def __init__(self, seq):
                    self.seq = list(seq)
                def create(self, **kwargs):
                    return self.seq.pop(0)
            def __init__(self, seq):
                self.completions = DummyClient.Chat.Completions(seq)
        def __init__(self, seq):
            self.chat = DummyClient.Chat(seq)
    return DummyClient(responses)


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


# ──────────────────────────────────────────
# Coverage gap tests (P0)
# ──────────────────────────────────────────

def test_normalize_handle_starts_with_nonalnum():
    # "-creator-" → regex keeps hyphen, strip("_") keeps hyphen, [0]="-" triggers yt_ prefix
    # and [-1]="-" triggers _col suffix
    result = ca._normalize_handle("-creator-")
    assert result.startswith("yt_")
    assert result.endswith("_col")


def test_normalize_handle_ends_with_nonalnum_only():
    # "creator-" → [0]="c" is alnum (no prefix), [-1]="-" triggers _col suffix
    result = ca._normalize_handle("creator-")
    assert not result.startswith("yt_")
    assert result.endswith("_col")


def test_load_session_reads_existing_file(monkeypatch, tmp_path):
    # Write a session file BEFORE building the agent so __init__ picks it up
    saved_msgs = [{"role": "user", "content": "previous question"}]
    (tmp_path / "creator.json").write_text(
        json.dumps({"channel_input": "@creator", "messages": saved_msgs})
    )
    agent = _build_agent(monkeypatch, tmp_path)
    assert agent.messages == saved_msgs


def test_chat_sliding_window_trims_to_max_history(monkeypatch, tmp_path):
    # Pre-fill to MAX_HISTORY - 1; one chat() call pushes it over by 2, trim fires
    agent = _build_agent(monkeypatch, tmp_path, responses=[Resp(Msg(content="ok"))])
    agent.messages = [{"role": "user", "content": "x"}] * (ca._MAX_HISTORY - 1)
    agent.chat("trigger")
    assert len(agent.messages) == ca._MAX_HISTORY


def test_normalize_handle_strips_youtube_url():
    result = ca._normalize_handle("https://www.youtube.com/@CreatorName")
    assert result == "creatorname"


# ──────────────────────────────────────────
# Agent behavioral tests (P1b)
# ──────────────────────────────────────────

def test_chat_multi_turn_context_retention(monkeypatch, tmp_path):
    responses = [Resp(Msg(content="r1")), Resp(Msg(content="r2"))]
    agent = _build_agent(monkeypatch, tmp_path, responses=responses)
    agent.chat("q1")
    agent.chat("q2")
    assert len(agent.messages) == 4
    assert agent.messages[2]["content"] == "q2"
    assert agent.messages[3]["content"] == "r2"


def test_session_persistence_and_reload(monkeypatch, tmp_path):
    agent1 = _build_agent(monkeypatch, tmp_path, responses=[Resp(Msg(content="reply1"))])
    agent1.chat("hello")

    # Build a second agent from the same session dir — it should load agent1's history
    agent2 = _build_agent(monkeypatch, tmp_path, responses=[Resp(Msg(content="reply2"))])
    assert len(agent2.messages) == 2
    assert agent2.messages[0]["content"] == "hello"
    assert agent2.messages[1]["content"] == "reply1"


def test_chat_multiple_tool_calls_in_single_response(monkeypatch, tmp_path):
    query_calls = []

    class TrackingStore:
        def query(self, **kwargs):
            query_calls.append(kwargs.get("query_text", ""))
            return {"documents": ["ctx"], "metadatas": [{"title": "V"}]}

    monkeypatch.setattr(ca.config, "sessions_dir", str(tmp_path))
    monkeypatch.setattr(ca, "load_persona", lambda _: None)
    monkeypatch.setattr(ca, "VectorStoreManager", lambda: TrackingStore())

    responses = [
        Resp(Msg(tool_calls=[ToolCall("tc1", {"query": "q1"}), ToolCall("tc2", {"query": "q2"})])),
        Resp(Msg(content="final")),
    ]
    monkeypatch.setattr(ca, "OpenAI", lambda: _make_client(responses))
    agent = ca.ChannelAgent("@creator")

    reply = agent.chat("hi")
    assert reply == "final"
    assert query_calls == ["q1", "q2"]


def test_chat_partial_persona_fields(monkeypatch, tmp_path):
    # Persona missing optional fields — _system_prompt() must not raise
    persona = {"display_name": "Creator", "tone": "casual"}
    agent = _build_agent(monkeypatch, tmp_path, persona=persona)
    prompt = agent._system_prompt()
    assert "Creator" in prompt
    assert isinstance(prompt, str) and len(prompt) > 0


def test_chat_persona_all_empty_fields(monkeypatch, tmp_path):
    persona = {"display_name": "", "topics": [], "tone": "", "style": "", "persona_summary": "", "common_phrases": []}
    agent = _build_agent(monkeypatch, tmp_path, persona=persona)
    prompt = agent._system_prompt()
    assert isinstance(prompt, str) and len(prompt) > 0


def test_chat_history_sliding_window_multiple_calls(monkeypatch, tmp_path):
    # Fill to MAX_HISTORY - 2; two more chat() calls should stabilise at MAX_HISTORY
    n = ca._MAX_HISTORY - 2
    responses = [Resp(Msg(content="ok"))] * 4
    agent = _build_agent(monkeypatch, tmp_path, responses=responses)
    agent.messages = [{"role": "user", "content": "x"}] * n
    agent.chat("call1")
    assert len(agent.messages) == ca._MAX_HISTORY
    agent.chat("call2")
    assert len(agent.messages) == ca._MAX_HISTORY


def test_chat_openai_raises_propagates(monkeypatch, tmp_path):
    import pytest

    class ExplodingClient:
        class Chat:
            class Completions:
                def create(self, **kwargs):
                    raise RuntimeError("API down")
            completions = Completions()
        chat = Chat()

    monkeypatch.setattr(ca.config, "sessions_dir", str(tmp_path))
    monkeypatch.setattr(ca, "load_persona", lambda _: None)
    monkeypatch.setattr(ca, "VectorStoreManager", lambda: DummyStore({"documents": [], "metadatas": []}))
    monkeypatch.setattr(ca, "OpenAI", lambda: ExplodingClient())
    agent = ca.ChannelAgent("@creator")

    with pytest.raises(RuntimeError, match="API down"):
        agent.chat("hello")


def test_ensure_session_file_does_not_overwrite_existing(monkeypatch, tmp_path):
    agent = _build_agent(monkeypatch, tmp_path)
    path = tmp_path / f"{agent.session_id}.json"
    if path.exists():
        path.unlink()
    agent.ensure_session_file()
    mtime_before = path.stat().st_mtime
    agent.ensure_session_file()
    assert path.stat().st_mtime == mtime_before


def test_chat_empty_user_message(monkeypatch, tmp_path):
    agent = _build_agent(monkeypatch, tmp_path, responses=[Resp(Msg(content="ok"))])
    reply = agent.chat("")
    assert reply == "ok"
    assert agent.messages[0] == {"role": "user", "content": ""}


# ──────────────────────────────────────────
# Robustness / adversarial tests (P2)
# ──────────────────────────────────────────

def test_search_videos_store_returns_none_documents(monkeypatch, tmp_path):
    # documents=None is falsy — should return the "no results" message
    agent = _build_agent(monkeypatch, tmp_path, store_result={"documents": None, "metadatas": []})
    result = agent._search_videos("q")
    assert "No relevant content found" in result


def test_chat_tool_call_malformed_arguments_json(monkeypatch, tmp_path):
    import pytest

    class BadToolCall:
        id = "tc1"
        function = type("F", (), {"arguments": "not-valid-json"})()

    responses = [Resp(Msg(tool_calls=[BadToolCall()]))]
    agent = _build_agent(monkeypatch, tmp_path, responses=responses)

    with pytest.raises(json.JSONDecodeError):
        agent.chat("hi")


def test_load_session_invalid_json_raises(monkeypatch, tmp_path):
    import pytest
    (tmp_path / "creator.json").write_text("not-json")
    monkeypatch.setattr(ca.config, "sessions_dir", str(tmp_path))
    monkeypatch.setattr(ca, "load_persona", lambda _: None)
    monkeypatch.setattr(ca, "VectorStoreManager", lambda: DummyStore({"documents": [], "metadatas": []}))
    monkeypatch.setattr(ca, "OpenAI", lambda: _make_client([]))

    with pytest.raises(json.JSONDecodeError):
        ca.ChannelAgent("@creator")


def test_chat_sliding_window_fires_repeatedly(monkeypatch, tmp_path):
    # 12 consecutive calls; history must never exceed MAX_HISTORY
    n = 12
    responses = [Resp(Msg(content="ok"))] * n
    agent = _build_agent(monkeypatch, tmp_path, responses=responses)
    for _ in range(n):
        agent.chat("msg")
    assert len(agent.messages) <= ca._MAX_HISTORY
