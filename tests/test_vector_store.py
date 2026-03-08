import vector_store as vs
from document_cleaner import CleanedDocument
from tests.helpers import FakeClient, FakeCollection


class DummySplitter:
    def __init__(self, *args, **kwargs):
        self._chunks = None

    def split_text(self, text):
        return self._chunks if self._chunks is not None else [text]


def _make_manager(monkeypatch):
    fake_client = FakeClient()
    monkeypatch.setattr(vs.chromadb, "PersistentClient", lambda path: fake_client)
    monkeypatch.setattr(
        vs.embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        lambda model_name: object(),
    )
    monkeypatch.setattr(vs, "RecursiveCharacterTextSplitter", DummySplitter)
    mgr = vs.VectorStoreManager(persist_dir="/tmp/x")
    return mgr, fake_client


def test_get_collection_name_sanitization(monkeypatch):
    mgr, _ = _make_manager(monkeypatch)
    assert mgr._get_collection_name("https://www.youtube.com/@My Channel!").startswith("my_channel")
    assert mgr._get_collection_name("@@@").startswith("yt_")


def test_get_existing_video_ids_empty(monkeypatch):
    mgr, _ = _make_manager(monkeypatch)
    col = mgr.get_or_create_collection("@a")
    col._count = 0
    assert mgr.get_existing_video_ids("@a") == set()


def test_get_existing_video_ids_from_metadata(monkeypatch):
    mgr, _ = _make_manager(monkeypatch)
    col = mgr.get_or_create_collection("@a")
    col._count = 2
    col._get_result = {"metadatas": [{"video_id": "v1"}, {"video_id": "v2"}, {}]}
    assert mgr.get_existing_video_ids("@a") == {"v1", "v2"}


def test_add_document_no_chunks(monkeypatch):
    mgr, _ = _make_manager(monkeypatch)
    mgr.text_splitter._chunks = []
    doc = CleanedDocument("v1", "t", "@a", "url", "", 0, 0, {})
    assert mgr.add_document("@a", doc) == 0


def test_add_document_upserts_chunk_metadata(monkeypatch):
    mgr, _ = _make_manager(monkeypatch)
    mgr.text_splitter._chunks = ["chunk1", "chunk2"]
    doc = CleanedDocument(
        video_id="v1",
        title="Title",
        channel_name="@a",
        video_url="url",
        clean_text="x",
        word_count=1,
        paragraph_count=1,
        metadata={"video_id": "v1", "title": "Title"},
    )

    added = mgr.add_document("@a", doc)
    col = mgr.get_or_create_collection("@a")

    assert added == 2
    assert col.upserts[0]["ids"] == ["v1_chunk_0000", "v1_chunk_0001"]
    assert col.upserts[0]["metadatas"][0]["chunk_index"] == 0


def test_add_documents_handles_failures(monkeypatch):
    mgr, _ = _make_manager(monkeypatch)

    def fake_add(channel_name, doc):
        if doc.video_id == "bad":
            raise RuntimeError("fail")
        return 3

    mgr.add_document = fake_add
    docs = [
        CleanedDocument("ok", "t", "@a", "u", "x", 1, 1, {}),
        CleanedDocument("bad", "t", "@a", "u", "x", 1, 1, {}),
    ]
    stats = mgr.add_documents("@a", docs)
    assert stats == {"documents_processed": 1, "documents_failed": 1, "total_chunks": 3}


def test_query_uses_min_results_and_where(monkeypatch):
    mgr, _ = _make_manager(monkeypatch)
    col = mgr.get_or_create_collection("@a")
    col._count = 2
    col._query_result = {
        "documents": [["d1"]],
        "metadatas": [[{"title": "T"}]],
        "distances": [[0.2]],
    }

    out = mgr.query("@a", "q", n_results=5, where_filter={"video_id": "v1"})

    assert out["total_results"] == 1
    assert col.query_calls[0]["n_results"] == 2
    assert col.query_calls[0]["where"] == {"video_id": "v1"}


def test_list_collections_handles_str_and_obj(monkeypatch):
    mgr, fake_client = _make_manager(monkeypatch)
    fake_client.collections = {
        "col1": FakeCollection(count=1, metadata={"channel": "@a"}),
        "col2": FakeCollection(count=2, metadata={"channel": "@b"}),
    }
    mgr.client.list_collections = lambda: ["col1", type("C", (), {"name": "col2"})()]

    out = mgr.list_collections()
    assert len(out) == 2
    assert out[0]["name"] == "col1"


def test_delete_collection_success_and_failure(monkeypatch):
    mgr, _ = _make_manager(monkeypatch)
    mgr._get_collection_name = lambda channel: "ok"
    assert mgr.delete_collection("@a") is True

    mgr._get_collection_name = lambda channel: "raise_error"
    assert mgr.delete_collection("@a") is False


def test_get_stats(monkeypatch):
    mgr, _ = _make_manager(monkeypatch)
    col = mgr.get_or_create_collection("@a")
    col._count = 7
    mgr.get_existing_video_ids = lambda channel: {"v1", "v2"}

    stats = mgr.get_stats("@a")
    assert stats["total_chunks"] == 7
    assert stats["total_videos"] == 2


# ──────────────────────────────────────────
# Coverage gap tests (P0)
# ──────────────────────────────────────────

def test_list_collections_skips_falsy_name(monkeypatch):
    mgr, _ = _make_manager(monkeypatch)
    # Return a collection object whose .name is None — should be skipped
    mgr.client.list_collections = lambda: [type("C", (), {"name": None})()]
    assert mgr.list_collections() == []
