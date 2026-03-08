import youtube_fetcher as yf


class DummyYTT:
    def fetch(self, video_id, languages=None):
        return [{"text": "Hello"}, {"text": "[Music] world"}]


class DummyFormatter:
    def format_transcript(self, transcript):
        return "\n".join(item["text"] for item in transcript)


def test_extract_channel_identifier_formats():
    assert yf.extract_channel_identifier("https://www.youtube.com/@foo") == {
        "channel_url": "https://www.youtube.com/@foo"
    }
    assert yf.extract_channel_identifier("https://www.youtube.com/channel/UC1234567890123456789012") == {
        "channel_id": "UC1234567890123456789012"
    }
    assert yf.extract_channel_identifier("https://www.youtube.com/c/FooBar") == {
        "channel_url": "https://www.youtube.com/c/FooBar"
    }
    assert yf.extract_channel_identifier("@foo") == {"channel_url": "https://www.youtube.com/@foo"}
    assert yf.extract_channel_identifier("UC1234567890123456789012") == {
        "channel_id": "UC1234567890123456789012"
    }


def test_fetch_channel_videos_success(monkeypatch):
    def fake_get_channel(**kwargs):
        assert kwargs["sort_by"] == "newest"
        return [
            {
                "videoId": "v1",
                "title": {"runs": [{"text": "Title 1"}]},
                "publishedTimeText": {"simpleText": "1 day ago"},
                "viewCountText": {"simpleText": "1K views"},
                "lengthText": {"simpleText": "10:00"},
            },
            {"videoId": "v2"},
        ]

    monkeypatch.setattr(yf.scrapetube, "get_channel", fake_get_channel)
    out = yf.fetch_channel_videos("@abc", limit=2)

    assert out[0]["videoId"] == "v1"
    assert out[0]["title"] == "Title 1"
    assert out[1]["title"] == "Unknown"


def test_fetch_channel_videos_raises(monkeypatch):
    monkeypatch.setattr(yf.scrapetube, "get_channel", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    try:
        yf.fetch_channel_videos("@abc")
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_fetch_transcript_success(monkeypatch):
    monkeypatch.setattr(yf, "YouTubeTranscriptApi", DummyYTT)
    monkeypatch.setattr(yf, "TextFormatter", DummyFormatter)

    text = yf.fetch_transcript("vid")
    assert text == "Hello\n world"


def test_fetch_transcript_failure_returns_none(monkeypatch):
    class BadYTT:
        def fetch(self, video_id, languages=None):
            raise RuntimeError("no transcript")

    monkeypatch.setattr(yf, "YouTubeTranscriptApi", BadYTT)
    assert yf.fetch_transcript("vid") is None


def test_get_video_infos_filters_existing_and_marks_missing(monkeypatch):
    monkeypatch.setattr(
        yf,
        "fetch_channel_videos",
        lambda channel_input, limit=None: [
            {"videoId": "v1", "title": "A", "publishedTimeText": "p1", "viewCountText": "c1"},
            {"videoId": "v2", "title": "B", "publishedTimeText": "p2", "viewCountText": "c2"},
        ],
    )

    monkeypatch.setattr(yf, "fetch_transcript", lambda vid: "text" if vid == "v2" else None)

    infos = yf.get_video_infos("@abc", existing_video_ids={"v1"})

    assert len(infos) == 1
    assert infos[0].video_id == "v2"
    assert infos[0].transcript == "text"
    assert infos[0].error is None
