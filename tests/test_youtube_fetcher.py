"""Tests for youtube_fetcher.py (yt-dlp + Gemini implementation)."""

from unittest.mock import MagicMock, patch
import pytest
import youtube_fetcher as yf


# ──────────────────────────────────────────
# _normalize_channel_url
# ──────────────────────────────────────────

def test_normalize_bare_handle():
    url = yf._normalize_channel_url("@mkbhd")
    assert url == "https://www.youtube.com/@mkbhd/videos"


def test_normalize_full_url_no_videos_suffix():
    url = yf._normalize_channel_url("https://www.youtube.com/@mkbhd")
    assert url == "https://www.youtube.com/@mkbhd/videos"


def test_normalize_full_url_with_videos_suffix():
    url = yf._normalize_channel_url("https://www.youtube.com/@mkbhd/videos")
    assert url == "https://www.youtube.com/@mkbhd/videos"


def test_normalize_channel_id():
    url = yf._normalize_channel_url("UC1234567890123456789012")
    assert url == "https://www.youtube.com/channel/UC1234567890123456789012/videos"


def test_normalize_fallback_passthrough():
    url = yf._normalize_channel_url("some-other-value")
    assert url == "some-other-value"


# ──────────────────────────────────────────
# fetch_channel_videos
# ──────────────────────────────────────────

def _make_ydl_mock(entries):
    """Build a yt_dlp.YoutubeDL context-manager mock returning given entries."""
    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {"entries": entries}
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_ydl)
    cm.__exit__ = MagicMock(return_value=False)
    return cm, mock_ydl


def test_fetch_channel_videos_success(monkeypatch):
    entries = [
        {"id": "v1", "title": "Title 1", "upload_date": "20240101",
         "view_count": 1000, "duration": 600},
        {"id": "v2", "title": "Title 2", "upload_date": "20240201",
         "view_count": None, "duration": None},
    ]
    cm, _ = _make_ydl_mock(entries)
    monkeypatch.setattr(yf.yt_dlp, "YoutubeDL", lambda opts: cm)

    result = yf.fetch_channel_videos("@abc", limit=2)

    assert len(result) == 2
    assert result[0]["videoId"] == "v1"
    assert result[0]["title"] == "Title 1"
    assert result[0]["publishedTimeText"] == "20240101"
    assert result[0]["viewCountText"] == "1000"
    assert result[1]["viewCountText"] == ""
    assert result[1]["lengthText"] == ""


def test_fetch_channel_videos_skips_entries_without_id(monkeypatch):
    entries = [
        {"id": "v1", "title": "Valid"},
        {"title": "No ID"},
        None,
    ]
    cm, _ = _make_ydl_mock(entries)
    monkeypatch.setattr(yf.yt_dlp, "YoutubeDL", lambda opts: cm)

    result = yf.fetch_channel_videos("@abc")
    assert len(result) == 1
    assert result[0]["videoId"] == "v1"


def test_fetch_channel_videos_empty_channel(monkeypatch):
    cm, _ = _make_ydl_mock([])
    monkeypatch.setattr(yf.yt_dlp, "YoutubeDL", lambda opts: cm)

    result = yf.fetch_channel_videos("@empty")
    assert result == []


def test_fetch_channel_videos_raises_on_error(monkeypatch):
    mock_ydl = MagicMock()
    mock_ydl.extract_info.side_effect = RuntimeError("network error")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_ydl)
    cm.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(yf.yt_dlp, "YoutubeDL", lambda opts: cm)

    with pytest.raises(RuntimeError, match="network error"):
        yf.fetch_channel_videos("@abc")


def test_fetch_channel_videos_limit_passed_to_ydl(monkeypatch):
    captured_opts = {}

    def fake_ydl(opts):
        captured_opts.update(opts)
        cm, _ = _make_ydl_mock([])
        return cm

    monkeypatch.setattr(yf.yt_dlp, "YoutubeDL", fake_ydl)
    yf.fetch_channel_videos("@abc", limit=5)
    assert captured_opts.get("playlistend") == 5


def test_fetch_channel_videos_no_limit_no_playlistend(monkeypatch):
    captured_opts = {}

    def fake_ydl(opts):
        captured_opts.update(opts)
        cm, _ = _make_ydl_mock([])
        return cm

    monkeypatch.setattr(yf.yt_dlp, "YoutubeDL", fake_ydl)
    yf.fetch_channel_videos("@abc")
    assert "playlistend" not in captured_opts


# ──────────────────────────────────────────
# fetch_transcript
# ──────────────────────────────────────────

def _make_gemini_mock(text):
    """Build a mock for genai.GenerativeModel that returns the given text."""
    mock_response = MagicMock()
    mock_response.text = text
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    return mock_model


def test_fetch_transcript_success(monkeypatch):
    monkeypatch.setattr(yf.config, "gemini_api_key", "fake-key")
    mock_model = _make_gemini_mock("Hello world this is the transcript.")
    monkeypatch.setattr(yf.genai, "configure", MagicMock())
    monkeypatch.setattr(yf.genai, "GenerativeModel", lambda name: mock_model)

    result = yf.fetch_transcript("vid123")
    assert result == "Hello world this is the transcript."


def test_fetch_transcript_empty_response_returns_none(monkeypatch):
    monkeypatch.setattr(yf.config, "gemini_api_key", "fake-key")
    mock_model = _make_gemini_mock("")
    monkeypatch.setattr(yf.genai, "configure", MagicMock())
    monkeypatch.setattr(yf.genai, "GenerativeModel", lambda name: mock_model)

    result = yf.fetch_transcript("vid123")
    assert result is None


def test_fetch_transcript_none_response_returns_none(monkeypatch):
    monkeypatch.setattr(yf.config, "gemini_api_key", "fake-key")
    mock_model = _make_gemini_mock(None)
    monkeypatch.setattr(yf.genai, "configure", MagicMock())
    monkeypatch.setattr(yf.genai, "GenerativeModel", lambda name: mock_model)

    result = yf.fetch_transcript("vid123")
    assert result is None


def test_fetch_transcript_no_api_key_returns_none(monkeypatch):
    monkeypatch.setattr(yf.config, "gemini_api_key", "")

    result = yf.fetch_transcript("vid123")
    assert result is None


def test_fetch_transcript_gemini_exception_returns_none(monkeypatch):
    monkeypatch.setattr(yf.config, "gemini_api_key", "fake-key")
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = RuntimeError("quota exceeded")
    monkeypatch.setattr(yf.genai, "configure", MagicMock())
    monkeypatch.setattr(yf.genai, "GenerativeModel", lambda name: mock_model)

    result = yf.fetch_transcript("vid123")
    assert result is None


def test_fetch_transcript_passes_youtube_url(monkeypatch):
    monkeypatch.setattr(yf.config, "gemini_api_key", "fake-key")
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "transcript text"
    mock_model.generate_content.return_value = mock_response
    monkeypatch.setattr(yf.genai, "configure", MagicMock())
    monkeypatch.setattr(yf.genai, "GenerativeModel", lambda name: mock_model)

    yf.fetch_transcript("abc123")

    call_args = mock_model.generate_content.call_args[0][0]
    # Second element of the content list should contain the YouTube URL
    assert "https://www.youtube.com/watch?v=abc123" in str(call_args)


# ──────────────────────────────────────────
# get_video_infos
# ──────────────────────────────────────────

def test_get_video_infos_filters_existing_ids(monkeypatch):
    monkeypatch.setattr(
        yf, "fetch_channel_videos",
        lambda channel_input, limit=None: [
            {"videoId": "v1", "title": "A", "publishedTimeText": "p1", "viewCountText": "c1"},
            {"videoId": "v2", "title": "B", "publishedTimeText": "p2", "viewCountText": "c2"},
        ],
    )
    monkeypatch.setattr(yf, "fetch_transcript", lambda vid: "transcript text")

    infos = yf.get_video_infos("@abc", existing_video_ids={"v1"})

    assert len(infos) == 1
    assert infos[0].video_id == "v2"


def test_get_video_infos_transcript_available(monkeypatch):
    monkeypatch.setattr(
        yf, "fetch_channel_videos",
        lambda channel_input, limit=None: [
            {"videoId": "v1", "title": "A", "publishedTimeText": "20240101", "viewCountText": "500"},
        ],
    )
    monkeypatch.setattr(yf, "fetch_transcript", lambda vid: "spoken words here")

    infos = yf.get_video_infos("@abc")

    assert infos[0].transcript == "spoken words here"
    assert infos[0].error is None
    assert infos[0].published_text == "20240101"
    assert infos[0].view_count == "500"


def test_get_video_infos_transcript_unavailable(monkeypatch):
    monkeypatch.setattr(
        yf, "fetch_channel_videos",
        lambda channel_input, limit=None: [
            {"videoId": "v1", "title": "A", "publishedTimeText": "p1", "viewCountText": "c1"},
        ],
    )
    monkeypatch.setattr(yf, "fetch_transcript", lambda vid: None)

    infos = yf.get_video_infos("@abc")

    assert len(infos) == 1
    assert infos[0].transcript is None
    assert infos[0].error == "Transcript unavailable"


def test_get_video_infos_empty_channel(monkeypatch):
    monkeypatch.setattr(
        yf, "fetch_channel_videos",
        lambda channel_input, limit=None: [],
    )
    infos = yf.get_video_infos("@abc")
    assert infos == []


def test_get_video_infos_channel_url_set_correctly(monkeypatch):
    monkeypatch.setattr(
        yf, "fetch_channel_videos",
        lambda channel_input, limit=None: [
            {"videoId": "xyz", "title": "T", "publishedTimeText": "", "viewCountText": ""},
        ],
    )
    monkeypatch.setattr(yf, "fetch_transcript", lambda vid: "text")

    infos = yf.get_video_infos("@abc")
    assert infos[0].channel_url == "https://www.youtube.com/watch?v=xyz"
    assert infos[0].channel_name == "@abc"
