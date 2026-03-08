"""
Cross-component integration tests.

All external I/O (OpenAI, ChromaDB, YouTube APIs) is stubbed.
Real module functions are used for everything else — persona build → load → agent,
session hand-off between agent instances, and full pipeline runs.
"""
import json

import channel_agent as ca
import persona_builder as pb
import pipeline
from document_cleaner import CleanedDocument
from tests.conftest import FakeClient, make_cleaned_doc


# ──────────────────────────────────────────
# Helpers shared across integration tests
# ──────────────────────────────────────────

def _persona_openai_client(display_name="IntegCreator"):
    response_content = json.dumps({
        "display_name": display_name,
        "topics": ["investing"],
        "tone": "educational",
        "style": "clear and concise",
        "common_phrases": ["let's dive in"],
        "expertise_level": "intermediate",
        "language": "English",
        "persona_summary": "An educator who simplifies complex topics.",
    })

    class Response:
        class Choice:
            class Message:
                content = response_content
            message = Message()
        choices = [Choice()]

    class Completions:
        def create(self, **kwargs):
            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    return Client()


def _agent_openai_client(reply="agent reply"):
    class Msg:
        def __init__(self):
            self.content = reply
            self.tool_calls = []

    class Choice:
        def __init__(self):
            self.message = Msg()

    class Response:
        def __init__(self):
            self.choices = [Choice()]

    class Completions:
        def create(self, **kwargs):
            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    return Client()


# ──────────────────────────────────────────
# Integration test 1: persona build → load → agent uses it
# ──────────────────────────────────────────

def test_persona_build_then_load_then_agent_uses_it(monkeypatch, tmp_path):
    """build_and_save_persona writes a file; ChannelAgent.__init__ loads it via load_persona."""
    monkeypatch.setattr(pb.config, "personas_dir", str(tmp_path))
    monkeypatch.setattr(ca.config, "sessions_dir", str(tmp_path))
    monkeypatch.setattr(ca.config, "personas_dir", str(tmp_path))

    # Use real load_persona but pointing at tmp_path (already done via config patch above)
    monkeypatch.setattr(pb, "OpenAI", lambda: _persona_openai_client("IntegCreator"))
    monkeypatch.setattr(ca, "OpenAI", lambda: _agent_openai_client())

    docs = [make_cleaned_doc(f"v{i}", f"Title {i}") for i in range(3)]
    pb.build_and_save_persona("@creator", docs)

    agent = ca.ChannelAgent("@creator")
    assert agent.persona is not None
    assert agent.persona["display_name"] == "IntegCreator"

    prompt = agent._system_prompt()
    assert "IntegCreator" in prompt


# ──────────────────────────────────────────
# Integration test 2: session persisted by one agent, loaded by another
# ──────────────────────────────────────────

def test_session_persisted_by_one_agent_loaded_by_another(monkeypatch, tmp_path):
    """agent1.chat() saves a session; a fresh agent2 instance loads that history."""
    monkeypatch.setattr(ca.config, "sessions_dir", str(tmp_path))
    monkeypatch.setattr(ca, "load_persona", lambda _: None)

    class DummyStore:
        def query(self, **kwargs):
            return {"documents": [], "metadatas": []}

    monkeypatch.setattr(ca, "VectorStoreManager", lambda: DummyStore())
    monkeypatch.setattr(ca, "OpenAI", lambda: _agent_openai_client("first reply"))

    agent1 = ca.ChannelAgent("@creator")
    agent1.chat("hello from agent1")

    # Build agent2 with a fresh OpenAI stub; it should reload agent1's saved session
    monkeypatch.setattr(ca, "OpenAI", lambda: _agent_openai_client("second reply"))
    agent2 = ca.ChannelAgent("@creator")

    assert len(agent2.messages) == 2
    assert agent2.messages[0]["content"] == "hello from agent1"
    assert agent2.messages[1]["content"] == "first reply"


# ──────────────────────────────────────────
# Integration test 3: full pipeline — no new videos
# ──────────────────────────────────────────

def test_full_pipeline_no_new_videos(monkeypatch, tmp_path):
    """Pipeline with zero new videos exits cleanly with new_video_count == 0."""
    from state_manager import PipelineState

    monkeypatch.setattr(pipeline, "get_video_infos", lambda **kwargs: [])
    monkeypatch.setattr(pipeline, "PipelineState", lambda: PipelineState(state_file=str(tmp_path / "s.json")))

    result = pipeline.run_pipeline(channel_input="@creator", limit=None)

    assert result["new_video_count"] == 0
    # "discovered" = 0 new videos found after filtering; pipeline exits at the clean step
    assert result["status"] in {"discovery_failed", "discovered", "no_docs_to_clean", "no_docs_to_ingest", "end"}


# ──────────────────────────────────────────
# Integration test 4: full pipeline — new videos → ingested → persona built
# ──────────────────────────────────────────

def test_full_pipeline_new_videos_ingested_and_persona_built(monkeypatch, tmp_path):
    """End-to-end pipeline run: 2 videos fetched → cleaned → ingested → persona built."""
    from state_manager import PipelineState
    from youtube_fetcher import VideoInfo

    # Stub external APIs
    monkeypatch.setattr(pipeline, "get_video_infos", lambda **kwargs: [
        VideoInfo("v1", "Video 1", "@creator", "url", "transcript one " * 60),
        VideoInfo("v2", "Video 2", "@creator", "url", "transcript two " * 60),
    ])

    fake_client = FakeClient()
    import vector_store as vs

    # Patch chromadb.PersistentClient and the embedding function (via embedding_functions submodule)
    monkeypatch.setattr(vs.chromadb, "PersistentClient", lambda path: fake_client)

    class FakeEmbedFn:
        def __call__(self, input):
            return [[0.1] * 10 for _ in input]

    monkeypatch.setattr(vs.embedding_functions, "SentenceTransformerEmbeddingFunction", lambda model_name: FakeEmbedFn())

    monkeypatch.setattr(pipeline, "PipelineState", lambda: PipelineState(state_file=str(tmp_path / "s.json")))
    monkeypatch.setattr(pb.config, "personas_dir", str(tmp_path))

    # Stub OpenAI for persona building (build_and_save_persona is imported locally in the pipeline node)
    monkeypatch.setattr(pb, "OpenAI", lambda: _persona_openai_client("PipeCreator"))

    result = pipeline.run_pipeline(channel_input="@creator", limit=None)

    assert result["new_video_count"] == 2
    assert result.get("ingestion_stats", {}).get("total_chunks", 0) > 0
    assert result["status"] in {"persona_built", "ingested"}
