"""Persistent Chroma vector-store construction, indexing, and search."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreError(Exception):
    """Raised when documents cannot be validated or indexed."""


def create_embeddings(model_name: str) -> Embeddings:
    """Create normalized CPU Hugging Face embeddings."""
    return HuggingFaceEmbeddings(
        model=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(
    *,
    persist_directory: Path,
    collection_name: str,
    embeddings: Embeddings,
) -> Chroma:
    """Create or reopen a persistent local Chroma collection."""
    if not collection_name.strip():
        raise ValueError("collection_name must not be blank")
    if persist_directory.exists() and not persist_directory.is_dir():
        raise NotADirectoryError(
            f"Persistence path is not a directory: {persist_directory}"
        )

    persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def _validated_chunk_ids(documents: list[Document]) -> list[str]:
    """Return unique, non-blank chunk IDs or raise a clear error."""
    chunk_ids: list[str] = []
    seen: set[str] = set()

    for index, document in enumerate(documents):
        chunk_id = document.metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise VectorStoreError(
                f"Document at index {index} has an invalid chunk_id"
            )
        if chunk_id in seen:
            raise VectorStoreError(f"Duplicate chunk_id in input batch: {chunk_id}")
        seen.add(chunk_id)
        chunk_ids.append(chunk_id)

    return chunk_ids


def index_documents(vector_store: Chroma, documents: list[Document]) -> int:
    """Upsert documents into Chroma using deterministic chunk IDs."""
    if not documents:
        return 0

    chunk_ids = _validated_chunk_ids(documents)
    try:
        vector_store.add_documents(documents, ids=chunk_ids)
    except Exception as exc:
        raise VectorStoreError("Failed to index documents in Chroma") from exc
    return len(documents)


def _validate_search(query: str, k: int) -> None:
    """Validate common similarity-search arguments."""
    if not query.strip():
        raise ValueError("query must not be blank")
    if k < 1:
        raise ValueError("k must be at least 1")


def similarity_search(
    vector_store: Chroma,
    query: str,
    *,
    k: int = 4,
) -> list[Document]:
    """Return the nearest documents for a non-blank query."""
    _validate_search(query, k)
    return vector_store.similarity_search(query, k=k)


def similarity_search_with_scores(
    vector_store: Chroma,
    query: str,
    *,
    k: int = 4,
) -> list[tuple[Document, float]]:
    """Return nearest documents with Chroma distances; lower is closer."""
    _validate_search(query, k)
    return vector_store.similarity_search_with_score(query, k=k)