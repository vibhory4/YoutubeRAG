import pipeline
from youtube_fetcher import VideoInfo
from document_cleaner import CleanedDocument


def _base_state(**overrides):
    state = {
        "channel_input": "@abc",
        "limit": None,
        "video_infos": [],
        "new_video_count": 0,
        "cleaned_docs": [],
        "clean_count": 0,
        "ingestion_stats": {},
        "persona_built": False,
        "status": "starting",
        "errors": [],
    }
    state.update(overrides)
    return state


def test_discover_videos_success(monkeypatch):
    class DummyState:
        def get_indexed_video_ids(self, channel_input):
            return {"old"}

    vids = [
        VideoInfo("v1", "T1", "@abc", "url", "text"),
        VideoInfo("v2", "T2", "@abc", "url", None),
    ]

    monkeypatch.setattr(pipeline, "PipelineState", lambda: DummyState())
    monkeypatch.setattr(pipeline, "get_video_infos", lambda **kwargs: vids)

    out = pipeline.discover_videos(_base_state())
    assert out["new_video_count"] == 1
    assert len(out["video_infos"]) == 1
    assert "No transcript" in out["errors"][0]


def test_discover_videos_failure(monkeypatch):
    monkeypatch.setattr(pipeline, "PipelineState", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "get_video_infos",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    out = pipeline.discover_videos(_base_state())
    assert out["status"] == "discovery_failed"
    assert out["new_video_count"] == 0


def test_clean_documents_no_videos():
    out = pipeline.clean_documents(_base_state(video_infos=[]))
    assert out["status"] == "no_docs_to_clean"
    assert out["clean_count"] == 0


def test_clean_documents_filters_short_and_errors(monkeypatch):
    infos = [
        VideoInfo("ok", "T1", "@abc", "url", "text"),
        VideoInfo("short", "T2", "@abc", "url", "text"),
        VideoInfo("bad", "T3", "@abc", "url", "text"),
    ]

    def fake_process(**kwargs):
        if kwargs["video_id"] == "bad":
            raise RuntimeError("bad clean")
        wc = 20 if kwargs["video_id"] == "ok" else 5
        return CleanedDocument(
            kwargs["video_id"], kwargs["title"], kwargs["channel_name"], "url", "x", wc, 1, {}
        )

    monkeypatch.setattr(pipeline, "process_transcript", fake_process)
    out = pipeline.clean_documents(_base_state(video_infos=infos))

    assert out["clean_count"] == 1
    assert len(out["errors"]) == 2


def test_ingest_to_vectorstore_no_docs():
    out = pipeline.ingest_to_vectorstore(_base_state(cleaned_docs=[]))
    assert out["status"] == "no_docs_to_ingest"


def test_ingest_to_vectorstore_success(monkeypatch):
    class DummyStore:
        def add_documents(self, channel_input, docs):
            assert channel_input == "@abc"
            return {"documents_processed": 1, "documents_failed": 0, "total_chunks": 4}

    calls = {"add": 0, "mark": 0}

    class DummyState:
        def add_channel(self, channel):
            calls["add"] += 1

        def mark_videos_indexed(self, channel, video_ids, failed_ids):
            calls["mark"] += 1
            assert video_ids == ["v1"]

    monkeypatch.setattr(pipeline, "VectorStoreManager", lambda: DummyStore())
    monkeypatch.setattr(pipeline, "PipelineState", lambda: DummyState())

    docs = [CleanedDocument("v1", "T", "@abc", "u", "x", 10, 1, {})]
    out = pipeline.ingest_to_vectorstore(_base_state(cleaned_docs=docs))

    assert out["status"] == "ingested"
    assert out["ingestion_stats"]["total_chunks"] == 4
    assert calls == {"add": 1, "mark": 1}


def test_ingest_to_vectorstore_failure(monkeypatch):
    class BadStore:
        def add_documents(self, *_args, **_kwargs):
            raise RuntimeError("ingest err")

    monkeypatch.setattr(pipeline, "VectorStoreManager", lambda: BadStore())
    out = pipeline.ingest_to_vectorstore(
        _base_state(cleaned_docs=[CleanedDocument("v1", "T", "@abc", "u", "x", 10, 1, {})])
    )
    assert out["status"] == "ingestion_failed"


def test_build_persona_node_success_and_failure(monkeypatch):
    class FakeModule:
        @staticmethod
        def build_and_save_persona(channel_input, cleaned_docs):
            return {"ok": True}

    import sys

    sys.modules["persona_builder"] = FakeModule
    out = pipeline.build_persona_node(_base_state(cleaned_docs=[object()]))
    assert out["persona_built"] is True

    class BadModule:
        @staticmethod
        def build_and_save_persona(channel_input, cleaned_docs):
            raise RuntimeError("oops")

    sys.modules["persona_builder"] = BadModule
    out = pipeline.build_persona_node(_base_state(cleaned_docs=[object()]))
    assert out["persona_built"] is False


def test_conditional_edges_and_run_pipeline(monkeypatch):
    assert pipeline.should_continue_to_clean({"new_video_count": 0}) == "end"
    assert pipeline.should_continue_to_clean({"new_video_count": 1}) == "clean"
    assert pipeline.should_continue_to_ingest({"clean_count": 0}) == "end"
    assert pipeline.should_continue_to_ingest({"clean_count": 1}) == "ingest"

    class DummyApp:
        def invoke(self, initial_state):
            initial_state["status"] = "ok"
            return initial_state

    monkeypatch.setattr(pipeline, "compile_pipeline", lambda: DummyApp())
    out = pipeline.run_pipeline("@abc", limit=1)
    assert out["status"] == "ok"
    assert out["channel_input"] == "@abc"


def test_build_and_compile_pipeline_graph():
    graph = pipeline.build_pipeline_graph()
    assert graph is not None
    app = pipeline.compile_pipeline()
    assert app is not None
