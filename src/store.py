from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401
            # TODO: initialize chromadb client + collection

            chroma_client = chromadb.Client()
            collection = chroma_client.get_or_create_collection(name="tuna_db")
            self._collection = collection

            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        # raise NotImplementedError("Implement EmbeddingStore._make_record")
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        # raise NotImplementedError("Implement EmbeddingStore._search_records")
        query_embedding = self._embedding_fn(query)
        
        scores = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            result_with_score = record.copy()
            result_with_score["score"] = score
            scores.append(result_with_score)
        
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # TODO: embed each doc and add to store
        # raise NotImplementedError("Implement EmbeddingStore.add_documents")
        if self._use_chroma and self._collection:
            ids = [doc.id for doc in docs]
            contents = [doc.content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            embeddings = [self._embedding_fn(doc.content) for doc in docs]
            self._collection.add(ids=ids, documents=contents, metadatas=metadatas, embeddings=embeddings)
        else:
            self._store.extend([self._make_record(doc) for doc in docs])

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        # raise NotImplementedError("Implement EmbeddingStore.search")
        if self._use_chroma and self._collection:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
            
            formatted_results = []
            if results['ids']:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "embedding": results['embeddings'][0][i] if results['embeddings'] else None,
                    })
            return formatted_results
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        # raise NotImplementedError("Implement EmbeddingStore.get_collection_size")
        if self._use_chroma and self._collection:
            return self._collection.count()
        else:
            return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        # raise NotImplementedError("Implement EmbeddingStore.search_with_filter")
        if self._use_chroma and self._collection:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter,
            )
            
            formatted_results = []
            if results['ids']:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "embedding": results['embeddings'][0][i] if results['embeddings'] else None,
                    })
            return formatted_results
        else:
            filtered_records = []
            if metadata_filter:
                for record in self._store:
                    # Check if all filter items are in record's metadata
                    if all(item in record["metadata"].items() for item in metadata_filter.items()):
                        filtered_records.append(record)
            else:
                filtered_records = self._store
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        # raise NotImplementedError("Implement EmbeddingStore.delete_document")
        if self._use_chroma and self._collection:
            initial_count = self.get_collection_size()
            self._collection.delete(ids=[doc_id])
            return self.get_collection_size() < initial_count
        else:
            initial_count = self.get_collection_size()
            self._store = [record for record in self._store if record.get("id") != doc_id]
            return self.get_collection_size() < initial_count