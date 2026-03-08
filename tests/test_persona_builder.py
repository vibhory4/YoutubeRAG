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
