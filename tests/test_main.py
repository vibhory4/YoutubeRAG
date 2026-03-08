from argparse import Namespace

import main


def test_cmd_index_prints_results(monkeypatch, capsys):
    monkeypatch.setitem(
        __import__("sys").modules,
        "pipeline",
        type(
            "M",
            (),
            {
                "run_pipeline": lambda **kwargs: {
                    "status": "ingested",
                    "new_video_count": 2,
                    "clean_count": 2,
                    "ingestion_stats": {"total_chunks": 5},
                    "errors": ["warn1"],
                }
            },
        ),
    )
    main.cmd_index(Namespace(channel="@abc", limit=2))
    out = capsys.readouterr().out
    assert "PIPELINE RESULTS" in out
    assert "Chunks added:  5" in out


def test_cmd_query_no_results(monkeypatch, capsys):
    class DummyStore:
        def query(self, **kwargs):
            return {"documents": [], "metadatas": [], "distances": [], "total_results": 0}

    monkeypatch.setitem(__import__("sys").modules, "vector_store", type("M", (), {"VectorStoreManager": DummyStore}))
    main.cmd_query(Namespace(channel="@abc", query="q", n=3))
    out = capsys.readouterr().out
    assert "No results found" in out


def test_cmd_query_prints_results(monkeypatch, capsys):
    class DummyStore:
        def query(self, **kwargs):
            return {
                "documents": ["A" * 10],
                "metadatas": [{"title": "T", "video_url": "u"}],
                "distances": [0.1],
                "total_results": 1,
            }

    monkeypatch.setitem(__import__("sys").modules, "vector_store", type("M", (), {"VectorStoreManager": DummyStore}))
    main.cmd_query(Namespace(channel="@abc", query="q", n=3))
    out = capsys.readouterr().out
    assert "--- Result 1" in out
    assert "Video: T" in out


def test_cmd_channels_empty(monkeypatch, capsys):
    class DummyState:
        def get_tracked_channels(self):
            return []

    monkeypatch.setitem(__import__("sys").modules, "state_manager", type("M", (), {"PipelineState": DummyState}))
    main.cmd_channels(Namespace())
    out = capsys.readouterr().out
    assert "No channels tracked yet" in out


def test_cmd_status_handles_vector_store_error(monkeypatch, capsys):
    class DummyState:
        def get_summary(self):
            return {"total_channels": 0, "total_videos_indexed": 0, "channels": {}, "last_updated": None}

    class BadStore:
        def __init__(self):
            raise RuntimeError("no store")

    monkeypatch.setitem(__import__("sys").modules, "state_manager", type("M", (), {"PipelineState": DummyState}))
    monkeypatch.setitem(__import__("sys").modules, "vector_store", type("M", (), {"VectorStoreManager": BadStore}))
    main.cmd_status(Namespace())
    out = capsys.readouterr().out
    assert "Vector store: Error" in out


def test_cmd_channels_lists_channels(monkeypatch, capsys):
    class DummyState:
        def get_tracked_channels(self):
            return ["@abc"]

        def get_channel_info(self, channel):
            return {"indexed_video_ids": ["v1", "v2"]}

    monkeypatch.setitem(__import__("sys").modules, "state_manager", type("M", (), {"PipelineState": DummyState}))
    main.cmd_channels(Namespace())
    out = capsys.readouterr().out
    assert "Tracked channels (1)" in out


def test_cmd_scheduler_and_mcp(monkeypatch):
    called = {"scheduler": False, "mcp": False}

    monkeypatch.setitem(
        __import__("sys").modules,
        "scheduler",
        type("M", (), {"start_scheduler": lambda interval_minutes=None: called.__setitem__("scheduler", True)}),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "mcp_server",
        type("M", (), {"mcp": type("S", (), {"run": staticmethod(lambda: called.__setitem__("mcp", True))})()}),
    )

    main.cmd_scheduler(Namespace(interval=5))
    main.cmd_mcp(Namespace())
    assert called["scheduler"] is True
    assert called["mcp"] is True


def test_main_without_command_prints_help(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["main.py"])
    main.main()
    out = capsys.readouterr().out
    assert "usage:" in out.lower()
