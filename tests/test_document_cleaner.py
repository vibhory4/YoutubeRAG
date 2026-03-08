import document_cleaner as dc


def test_clean_transcript_empty_returns_empty():
    assert dc.clean_transcript("") == ""
    assert dc.clean_transcript(None) == ""


def test_clean_transcript_removes_noise_and_normalizes():
    raw = """[Music]
    um um This is a test line.
    uh uh It has fillers.
    This is hyphen-\nated text and lower\ncase join.


    [Applause]
    """
    cleaned = dc.clean_transcript(raw)

    assert "[Music]" not in cleaned
    assert "[Applause]" not in cleaned
    assert "um" not in cleaned.lower()
    assert "uh" not in cleaned.lower()
    assert "hyphenated" in cleaned
    assert "lower case" in cleaned


def test_clean_transcript_paragraph_break_after_many_sentences():
    raw = "One. Two. Three. Four. Five. Six."
    cleaned = dc.clean_transcript(raw)
    # At least one paragraph exists and text survives.
    assert "One." in cleaned and "Six." in cleaned


def test_enrich_metadata_excludes_falsey_extras():
    meta = dc.enrich_metadata(
        video_id="vid1",
        title="Title",
        channel_name="@chan",
        published_text="",
        view_count=None,
        duration_seconds=0,
        region="IN",
    )
    assert meta["video_id"] == "vid1"
    assert meta["region"] == "IN"
    assert "published_text" not in meta
    assert "view_count" not in meta
    assert "duration_seconds" not in meta


def test_process_transcript_builds_cleaned_document():
    doc = dc.process_transcript(
        video_id="abc123",
        title="Video",
        channel_name="@creator",
        raw_transcript="This is sentence one. This is sentence two.",
        published_text="1 day ago",
    )

    assert doc.video_id == "abc123"
    assert doc.video_url.endswith("abc123")
    assert doc.word_count > 0
    assert doc.paragraph_count >= 1
    assert doc.metadata["channel_name"] == "@creator"
    assert doc.metadata["published_text"] == "1 day ago"
