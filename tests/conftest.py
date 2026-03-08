from dataclasses import dataclass


@dataclass
class DummyDoc:
    video_id: str
    title: str
    channel_name: str
    clean_text: str
    metadata: dict


class FakeCollection:
    def __init__(self, count=0, get_result=None, query_result=None, metadata=None):
        self._count = count
        self._get_result = get_result if get_result is not None else {"metadatas": []}
        self._query_result = (
            query_result
            if query_result is not None
            else {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        )
        self.metadata = metadata or {}
        self.upserts = []
        self.query_calls = []

    def count(self):
        return self._count

    def get(self, include=None):
        return self._get_result

    def upsert(self, ids, documents, metadatas):
        self.upserts.append({"ids": ids, "documents": documents, "metadatas": metadatas})
        self._count += len(ids)

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return self._query_result


class FakeClient:
    def __init__(self):
        self.collections = {}
        self.deleted = []

    def get_or_create_collection(self, name, embedding_function=None, metadata=None):
        if name not in self.collections:
            self.collections[name] = FakeCollection(metadata=metadata)
        return self.collections[name]

    def list_collections(self):
        return list(self.collections.keys())

    def get_collection(self, name, embedding_function=None):
        return self.collections[name]

    def delete_collection(self, name):
        if name == "raise_error":
            raise RuntimeError("delete failed")
        self.deleted.append(name)
        self.collections.pop(name, None)


# ──────────────────────────────────────────
# Shared factory helpers
# ──────────────────────────────────────────

def make_cleaned_doc(video_id="v1", title="T", text="word ", repeat=50):
    """Return a CleanedDocument for use in persona/agent/integration tests."""
    from document_cleaner import CleanedDocument
    body = text * repeat
    return CleanedDocument(video_id, title, "@ch", f"https://youtu.be/{video_id}", body, repeat, 1, {})


def make_session_file(tmp_path, session_id, channel_input, messages=None):
    """Write a valid session JSON file and return its Path."""
    import json
    path = tmp_path / f"{session_id}.json"
    path.write_text(json.dumps({"channel_input": channel_input, "messages": messages or []}))
    return path


def make_raising_class(exc_type=RuntimeError, msg="boom"):
    """Return a class whose __init__ raises exc_type(msg). Useful for dependency-injection tests."""
    class _Raise:
        def __init__(self, *a, **kw):
            raise exc_type(msg)
    return _Raise
