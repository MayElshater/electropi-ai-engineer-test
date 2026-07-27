"""Tests for persistent Chroma indexing and similarity search."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import app.vector_store as vector_store_module
from app.vector_store import (
    VectorStoreError,
    create_embeddings,
    create_vector_store,
    index_documents,
    similarity_search,
    similarity_search_with_scores,
)


class FakeEmbeddings(Embeddings):
    """Deterministic token-count embeddings for offline tests."""

    vocabulary = ("refund", "leave", "security", "password", "invoice")

    def _embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z]+", text.lower())
        values = [float(tokens.count(word)) for word in self.vocabulary]
        values.append(float(sum(token not in self.vocabulary for token in tokens)))
        magnitude = math.sqrt(sum(value * value for value in values))
        return [value / magnitude for value in values] if magnitude else [0.0] * 6

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def make_store(
    tmp_path: Path,
    collection_name: str = "test_collection",
) -> Chroma:
    """Create an isolated persistent store with fake embeddings."""
    return create_vector_store(
        persist_directory=tmp_path / "chroma",
        collection_name=collection_name,
        embeddings=FakeEmbeddings(),
    )


def make_document(
    content: str,
    chunk_id: str,
    *,
    source: str = "policy.md",
) -> Document:
    """Create a chunk with complete representative metadata."""
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "source_path": f"policies/{source}",
            "file_type": "markdown",
            "page": 2,
            "chunk_id": chunk_id,
            "chunk_index": 0,
            "total_chunks": 1,
        },
    )


def test_create_embeddings_configures_cpu_and_normalization(monkeypatch) -> None:
    created = object()
    received: dict[str, object] = {}

    def fake_embeddings(**kwargs: object) -> object:
        received.update(kwargs)
        return created

    monkeypatch.setattr(
        vector_store_module,
        "HuggingFaceEmbeddings",
        fake_embeddings,
    )

    result = create_embeddings("sentence-transformers/example")

    assert result is created
    assert received == {
        "model": "sentence-transformers/example",
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True},
    }


def test_create_vector_store_creates_persistence_directory(tmp_path: Path) -> None:
    persist_directory = tmp_path / "nested" / "chroma"

    store = create_vector_store(
        persist_directory=persist_directory,
        collection_name="create_test",
        embeddings=FakeEmbeddings(),
    )

    assert isinstance(store, Chroma)
    assert persist_directory.is_dir()


def test_empty_indexing_returns_zero(tmp_path: Path) -> None:
    assert index_documents(make_store(tmp_path), []) == 0


def test_indexes_documents_and_preserves_metadata(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    documents = [
        make_document("Annual leave policy", "policy.md:2:0"),
        make_document("Refund invoice policy", "billing.md:2:0", source="billing.md"),
    ]

    count = index_documents(store, documents)
    stored = store.get(ids=[document.metadata["chunk_id"] for document in documents])

    assert count == 2
    assert set(stored["ids"]) == {"policy.md:2:0", "billing.md:2:0"}
    assert {metadata["source"] for metadata in stored["metadatas"]} == {
        "policy.md",
        "billing.md",
    }


def test_missing_chunk_id_raises_clear_error(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    document = Document(page_content="Missing ID", metadata={"source": "bad.md"})

    with pytest.raises(VectorStoreError, match="index 0.*chunk_id"):
        index_documents(store, [document])


def test_empty_chunk_id_raises_error(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    document = Document(page_content="Empty ID", metadata={"chunk_id": "   "})

    with pytest.raises(VectorStoreError, match="invalid chunk_id"):
        index_documents(store, [document])


def test_duplicate_batch_ids_are_rejected_before_writing(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    documents = [
        make_document("First", "duplicate:na:0"),
        make_document("Second", "duplicate:na:0"),
    ]

    with pytest.raises(VectorStoreError, match="Duplicate chunk_id"):
        index_documents(store, documents)

    assert store.get()["ids"] == []


def test_similarity_search_ranks_semantic_match_first(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    index_documents(
        store,
        [
            make_document("Employees may request annual leave.", "leave.md:na:0"),
            make_document(
                "Refund requests must include an invoice.",
                "refund.md:na:0",
                source="refund.md",
            ),
            make_document(
                "Passwords must not be shared.",
                "security.md:na:0",
                source="security.md",
            ),
        ],
    )

    results = similarity_search(
        store,
        "How do I request a refund with my invoice?",
        k=3,
    )

    assert results[0].metadata["chunk_id"] == "refund.md:na:0"
    assert results[0].metadata["source"] == "refund.md"


def test_search_respects_k(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    index_documents(
        store,
        [make_document(f"Document {index}", f"doc.md:na:{index}") for index in range(3)],
    )

    assert len(similarity_search(store, "Document", k=2)) <= 2


@pytest.mark.parametrize("query", ["", " ", "\t\n"])
def test_blank_query_is_rejected(tmp_path: Path, query: str) -> None:
    with pytest.raises(ValueError, match="query"):
        similarity_search(make_store(tmp_path), query)


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_is_rejected(tmp_path: Path, k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        similarity_search(make_store(tmp_path), "refund", k=k)


def test_reindexing_same_id_upserts_latest_content(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    chunk_id = "policy.md:na:0"
    index_documents(store, [make_document("Annual leave rules", chunk_id)])
    index_documents(store, [make_document("Refund invoice rules", chunk_id)])

    stored = store.get(ids=[chunk_id], include=["documents"])

    assert stored["ids"] == [chunk_id]
    assert stored["documents"] == ["Refund invoice rules"]
    assert similarity_search(store, "refund invoice", k=1)[0].page_content == (
        "Refund invoice rules"
    )


def test_reopening_store_preserves_indexed_data(tmp_path: Path) -> None:
    persist_directory = tmp_path / "persistent"
    first = create_vector_store(
        persist_directory=persist_directory,
        collection_name="reopen_test",
        embeddings=FakeEmbeddings(),
    )
    index_documents(first, [make_document("Refund invoice policy", "refund.md:na:0")])

    reopened = create_vector_store(
        persist_directory=persist_directory,
        collection_name="reopen_test",
        embeddings=FakeEmbeddings(),
    )

    assert similarity_search(reopened, "refund invoice", k=1)[0].metadata[
        "chunk_id"
    ] == "refund.md:na:0"


def test_collections_are_isolated(tmp_path: Path) -> None:
    persist_directory = tmp_path / "shared"
    first = create_vector_store(
        persist_directory=persist_directory,
        collection_name="first_collection",
        embeddings=FakeEmbeddings(),
    )
    second = create_vector_store(
        persist_directory=persist_directory,
        collection_name="second_collection",
        embeddings=FakeEmbeddings(),
    )
    index_documents(first, [make_document("Refund invoice", "refund.md:na:0")])
    index_documents(second, [make_document("Annual leave", "leave.md:na:0")])

    assert first.get()["ids"] == ["refund.md:na:0"]
    assert second.get()["ids"] == ["leave.md:na:0"]


def test_existing_file_persistence_path_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "not-a-directory"
    path.write_text("file", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        create_vector_store(
            persist_directory=path,
            collection_name="invalid_path",
            embeddings=FakeEmbeddings(),
        )


def test_blank_collection_name_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="collection_name"):
        create_vector_store(
            persist_directory=tmp_path / "chroma",
            collection_name="  ",
            embeddings=FakeEmbeddings(),
        )


def test_similarity_search_with_scores_returns_ranked_distances(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    index_documents(
        store,
        [
            make_document("Refund invoice requirements", "refund.md:na:0"),
            make_document("Password security requirements", "security.md:na:0"),
        ],
    )

    results = similarity_search_with_scores(store, "refund invoice", k=2)

    assert all(isinstance(document, Document) for document, _ in results)
    assert all(isinstance(distance, float) for _, distance in results)
    assert results[0][0].metadata["chunk_id"] == "refund.md:na:0"
    assert results[0][1] <= results[1][1]


def test_all_chunk_metadata_survives_round_trip(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    document = make_document("Refund invoice policy", "policy.pdf:2:0")

    index_documents(store, [document])
    result = similarity_search(store, "refund invoice", k=1)[0]

    for key in (
        "source",
        "source_path",
        "file_type",
        "page",
        "chunk_id",
        "chunk_index",
        "total_chunks",
    ):
        assert result.metadata[key] == document.metadata[key]