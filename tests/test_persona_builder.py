import json

import persona_builder as pb
from document_cleaner import CleanedDocument


def test_normalize_handle_shapes_filename():
    assert pb._normalize_handle("https://www.youtube.com/@Foo Bar") == "foo_bar"
    assert pb._normalize_handle("@@@") == "yt__col"


def test_build_and_save_persona_no_docs_returns_empty():
    assert pb.build_and_save_persona("@abc", []) == {}


def test_build_and_save_persona_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(pb.config, "personas_dir", str(tmp_path))

    class DummyResponse:
        class Choice:
            class Message:
                content = json.dumps(
                    {
                        "display_name": "Creator",
                        "topics": ["finance"],
                        "tone": "analytical",
                        "style": "clear",
                        "common_phrases": ["let's break it down"],
                        "expertise_level": "intermediate",
                        "language": "English",
                        "persona_summary": "summary",
                    }
                )

            message = Message()

        choices = [Choice()]

    class DummyClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    assert kwargs["model"] == "gpt-4o-mini"
                    return DummyResponse()

            completions = Completions()

        chat = Chat()

    monkeypatch.setattr(pb, "OpenAI", lambda: DummyClient())

    docs = [
        CleanedDocument("v1", "Video 1", "@abc", "url", "text " * 50, 50, 1, {}),
        CleanedDocument("v2", "Video 2", "@abc", "url", "text " * 50, 50, 1, {}),
    ]

    persona = pb.build_and_save_persona("@abc", docs)
    assert persona["display_name"] == "Creator"
    assert persona["video_count"] == 2

    path = tmp_path / "abc.json"
    assert path.exists()


def test_load_persona(monkeypatch, tmp_path):
    monkeypatch.setattr(pb.config, "personas_dir", str(tmp_path))
    path = tmp_path / "abc.json"
    payload = {"display_name": "X"}
    path.write_text(json.dumps(payload))

    assert pb.load_persona("@abc") == payload
    assert pb.load_persona("@missing") is None


# ──────────────────────────────────────────
# Persona quality tests (P1c) + robustness (P2)
# ──────────────────────────────────────────

def _make_dummy_client(display_name="Creator"):
    """Build a DummyClient that returns a fixed persona JSON and records kwargs."""
    response_content = json.dumps({
        "display_name": display_name,
        "topics": ["topic1"],
        "tone": "analytical",
        "style": "clear",
        "common_phrases": ["let's go"],
        "expertise_level": "intermediate",
        "language": "English",
        "persona_summary": "summary",
    })

    class DummyResponse:
        class Choice:
            class Message:
                content = response_content
            message = Message()
        choices = [Choice()]

    class Completions:
        def __init__(self):
            self.last_kwargs = None

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return DummyResponse()

    class Chat:
        def __init__(self):
            self.completions = Completions()

    class DummyClient:
        def __init__(self):
            self.chat = Chat()

    return DummyClient()


def test_build_persona_samples_at_most_20_docs(monkeypatch, tmp_path):
    monkeypatch.setattr(pb.config, "personas_dir", str(tmp_path))
    client = _make_dummy_client()
    monkeypatch.setattr(pb, "OpenAI", lambda: client)

    docs = [
        CleanedDocument(f"v{i}", f"Title {i}", "@ch", "url", "word " * 50, 50, 1, {})
        for i in range(50)
    ]
    pb.build_and_save_persona("@abc", docs)

    prompt = client.chat.completions.last_kwargs["messages"][0]["content"]
    assert prompt.count("[Video ") <= 20


def test_build_persona_truncates_excerpts_at_500_chars(monkeypatch, tmp_path):
    monkeypatch.setattr(pb.config, "personas_dir", str(tmp_path))
    client = _make_dummy_client()
    monkeypatch.setattr(pb, "OpenAI", lambda: client)

    docs = [CleanedDocument("v1", "T", "@ch", "url", "x" * 600, 600, 1, {})]
    pb.build_and_save_persona("@abc", docs)

    prompt = client.chat.completions.last_kwargs["messages"][0]["content"]
    # The 501-char substring must not appear — excerpt was capped at 500
    assert "x" * 501 not in prompt


def test_build_persona_writes_built_at_and_video_count(monkeypatch, tmp_path):
    monkeypatch.setattr(pb.config, "personas_dir", str(tmp_path))
    monkeypatch.setattr(pb, "OpenAI", lambda: _make_dummy_client())

    docs = [
        CleanedDocument("v1", "T1", "@abc", "url", "text " * 50, 50, 1, {}),
        CleanedDocument("v2", "T2", "@abc", "url", "text " * 50, 50, 1, {}),
    ]
    persona = pb.build_and_save_persona("@abc", docs)

    assert "built_at" in persona and len(persona["built_at"]) > 0
    assert persona["video_count"] == 2


def test_build_persona_writes_channel_input_field(monkeypatch, tmp_path):
    monkeypatch.setattr(pb.config, "personas_dir", str(tmp_path))
    monkeypatch.setattr(pb, "OpenAI", lambda: _make_dummy_client())

    docs = [CleanedDocument("v1", "T", "@abc", "url", "text " * 50, 50, 1, {})]
    persona = pb.build_and_save_persona("@abc", docs)

    assert persona["channel_input"] == "@abc"


def test_build_persona_overwrites_existing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(pb.config, "personas_dir", str(tmp_path))
    docs = [CleanedDocument("v1", "T", "@abc", "url", "text " * 50, 50, 1, {})]

    monkeypatch.setattr(pb, "OpenAI", lambda: _make_dummy_client("V1"))
    pb.build_and_save_persona("@abc", docs)

    monkeypatch.setattr(pb, "OpenAI", lambda: _make_dummy_client("V2"))
    pb.build_and_save_persona("@abc", docs)

    assert pb.load_persona("@abc")["display_name"] == "V2"


def test_build_persona_single_doc_no_index_error(monkeypatch, tmp_path):
    monkeypatch.setattr(pb.config, "personas_dir", str(tmp_path))
    monkeypatch.setattr(pb, "OpenAI", lambda: _make_dummy_client())

    docs = [CleanedDocument("v1", "T", "@abc", "url", "text " * 50, 50, 1, {})]
    persona = pb.build_and_save_persona("@abc", docs)

    assert isinstance(persona, dict) and len(persona) > 0


def test_build_persona_gpt_raises_propagates(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr(pb.config, "personas_dir", str(tmp_path))

    class ExplodingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("OpenAI down")

    class ExplodingChat:
        completions = ExplodingCompletions()

    class ExplodingClient:
        chat = ExplodingChat()

    monkeypatch.setattr(pb, "OpenAI", lambda: ExplodingClient())

    docs = [CleanedDocument("v1", "T", "@abc", "url", "text " * 50, 50, 1, {})]
    with pytest.raises(RuntimeError, match="OpenAI down"):
        pb.build_and_save_persona("@abc", docs)


def test_build_persona_gpt_returns_non_json_raises(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr(pb.config, "personas_dir", str(tmp_path))

    class BadResponse:
        class Choice:
            class Message:
                content = "plain text not json"
            message = Message()
        choices = [Choice()]

    class BadCompletions:
        def create(self, **kwargs):
            return BadResponse()

    class BadChat:
        completions = BadCompletions()

    class BadClient:
        chat = BadChat()

    monkeypatch.setattr(pb, "OpenAI", lambda: BadClient())

    docs = [CleanedDocument("v1", "T", "@abc", "url", "text " * 50, 50, 1, {})]
    with pytest.raises(json.JSONDecodeError):
        pb.build_and_save_persona("@abc", docs)


def test_normalize_handle_strips_youtube_url_persona():
    # Hyphen is preserved by the regex [^a-z0-9_-], so "Foo-Bar" → "foo-bar"
    assert pb._normalize_handle("https://www.youtube.com/@Foo-Bar") == "foo-bar"
