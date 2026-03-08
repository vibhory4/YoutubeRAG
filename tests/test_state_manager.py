import json

from state_manager import PipelineState


def test_load_missing_file_defaults(tmp_path):
    state_file = tmp_path / "state.json"
    state = PipelineState(state_file=str(state_file))
    assert state.state == {"channels": {}, "last_updated": None}


def test_load_invalid_json_defaults(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text("not-json")
    state = PipelineState(state_file=str(state_file))
    assert state.state == {"channels": {}, "last_updated": None}


def test_add_and_remove_channel(tmp_path):
    state = PipelineState(state_file=str(tmp_path / "state.json"))

    state.add_channel("@abc")
    state.add_channel("@abc")  # idempotent

    channels = state.get_tracked_channels()
    assert channels == ["@abc"]

    state.remove_channel("@abc")
    assert state.get_tracked_channels() == []


def test_mark_videos_indexed_creates_channel_and_dedupes(tmp_path):
    state = PipelineState(state_file=str(tmp_path / "state.json"))

    state.mark_videos_indexed("@abc", ["v1", "v2", "v1"], failed_ids=["v3"])
    info = state.get_channel_info("@abc")

    assert set(info["indexed_video_ids"]) == {"v1", "v2"}
    assert info["total_videos_indexed"] == 2
    assert info["total_videos_failed"] == 1
    assert info["last_checked"] is not None


def test_summary_aggregates_counts(tmp_path):
    state = PipelineState(state_file=str(tmp_path / "state.json"))
    state.mark_videos_indexed("@a", ["v1", "v2"])
    state.mark_videos_indexed("@b", ["v3"])

    summary = state.get_summary()
    assert summary["total_channels"] == 2
    assert summary["total_videos_indexed"] == 3
    assert summary["channels"]["@a"]["videos_indexed"] == 2

    raw = json.loads((tmp_path / "state.json").read_text())
    assert raw["last_updated"] is not None
