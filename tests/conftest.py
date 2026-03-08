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
