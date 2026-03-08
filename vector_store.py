"""ChromaDB vector store management for YouTube transcript RAG."""

import logging
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config
from document_cleaner import CleanedDocument

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages ChromaDB collections for YouTube channel transcripts."""

    def __init__(
        self,
        persist_dir: str = None,
        embedding_model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.persist_dir = persist_dir or config.chroma_persist_dir
        self.embedding_model_name = embedding_model or config.embedding_model
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Initialize embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
            length_function=len,
        )

        logger.info(
            f"VectorStore initialized: persist={self.persist_dir}, "
            f"model={self.embedding_model_name}"
        )

    def _get_collection_name(self, channel_name: str) -> str:
        """Generate a safe collection name from channel input."""
        import re
        # Extract handle or channel name
        name = channel_name.strip().lower()
        name = re.sub(r"https?://.*youtube\.com/", "", name)
        name = re.sub(r"[^a-z0-9_-]", "_", name)
        name = name.strip("_")[:50]  # ChromaDB name length limit
        # Must start and end with alphanumeric
        if not name or not name[0].isalnum():
            name = "yt_" + name
        if not name[-1].isalnum():
            name = name + "_col"
        return name

    def get_or_create_collection(self, channel_name: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection for a channel."""
        col_name = self._get_collection_name(channel_name)
        collection = self.client.get_or_create_collection(
            name=col_name,
            embedding_function=self.embedding_fn,
            metadata={"channel": channel_name, "hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{col_name}': {collection.count()} documents")
        return collection

    def get_existing_video_ids(self, channel_name: str) -> set[str]:
        """Get all video IDs already indexed for a channel."""
        collection = self.get_or_create_collection(channel_name)

        if collection.count() == 0:
            return set()

        # Get all metadata to extract video_ids
        results = collection.get(include=["metadatas"])
        video_ids = set()
        for meta in results["metadatas"]:
            if "video_id" in meta:
                video_ids.add(meta["video_id"])

        return video_ids

    def add_document(self, channel_name: str, doc: CleanedDocument) -> int:
        """
        Chunk and add a cleaned document to the vector store.

        Returns the number of chunks added.
        """
        collection = self.get_or_create_collection(channel_name)

        # Split text into chunks
        chunks = self.text_splitter.split_text(doc.clean_text)

        if not chunks:
            logger.warning(f"No chunks generated for {doc.video_id}")
            return 0

        # Prepare batch for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc.video_id}_chunk_{i:04d}"
            ids.append(chunk_id)
            documents.append(chunk)

            # Per-chunk metadata
            chunk_meta = {
                **doc.metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
            }
            metadatas.append(chunk_meta)

        # Upsert (idempotent - safe for re-runs)
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(
            f"Added {len(chunks)} chunks for '{doc.title}' ({doc.video_id})"
        )
        return len(chunks)

    def add_documents(
        self, channel_name: str, docs: list[CleanedDocument]
    ) -> dict:
        """Add multiple documents. Returns summary stats."""
        total_chunks = 0
        processed = 0
        failed = 0

        for doc in docs:
            try:
                n = self.add_document(channel_name, doc)
                total_chunks += n
                processed += 1
            except Exception as e:
                logger.error(f"Failed to add {doc.video_id}: {e}")
                failed += 1

        return {
            "documents_processed": processed,
            "documents_failed": failed,
            "total_chunks": total_chunks,
        }

    def query(
        self,
        channel_name: str,
        query_text: str,
        n_results: int = 5,
        where_filter: Optional[dict] = None,
    ) -> dict:
        """
        Query the vector store for relevant transcript chunks.

        Returns dict with documents, metadatas, distances.
        """
        collection = self.get_or_create_collection(channel_name)

        kwargs = {
            "query_texts": [query_text],
            "n_results": min(n_results, collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = collection.query(**kwargs)

        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "total_results": len(results["documents"][0]) if results["documents"] else 0,
        }

    def list_collections(self) -> list[dict]:
        """List all channel collections with stats."""
        collections = self.client.list_collections()
        result = []
        for collection_item in collections:
            # ChromaDB versions differ: list_collections may return names or collection objects.
            if isinstance(collection_item, str):
                col_name = collection_item
                col = self.client.get_collection(
                    name=col_name,
                    embedding_function=self.embedding_fn,
                )
            else:
                col_name = getattr(collection_item, "name", None)
                if not col_name:
                    continue
                col = self.client.get_collection(
                    name=col_name,
                    embedding_function=self.embedding_fn,
                )
            meta = col.metadata or {}
            result.append({
                "name": col_name,
                "channel": meta.get("channel", "unknown"),
                "document_count": col.count(),
            })
        return result

    def delete_collection(self, channel_name: str) -> bool:
        """Delete an entire channel's collection."""
        col_name = self._get_collection_name(channel_name)
        try:
            self.client.delete_collection(col_name)
            logger.info(f"Deleted collection: {col_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {col_name}: {e}")
            return False

    def get_stats(self, channel_name: str) -> dict:
        """Get statistics for a channel's collection."""
        collection = self.get_or_create_collection(channel_name)
        video_ids = self.get_existing_video_ids(channel_name)

        return {
            "channel": channel_name,
            "total_chunks": collection.count(),
            "total_videos": len(video_ids),
            "video_ids": list(video_ids),
        }
