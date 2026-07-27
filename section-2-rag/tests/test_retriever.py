"""Tests for distance-filtered retrieval and deterministic context building."""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import app.retriever as retriever_module
from app.retriever import (
    RetrievedChunk,
    RetrievalStatus,
    build_context,
    retrieve_documents,
)
from app.vector_store import create_vector_store, index_documents


class FakeEmbeddings(Embeddings):
    """Deterministic token-count embeddings for offline retrieval tests."""

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


def make_store(tmp_path: Path, name: str = "retriever_test") -> Chroma:
    """Create an isolated persistent Chroma store."""
    return create_vector_store(
        persist_directory=tmp_path / "chroma",
        collection_name=name,
        embeddings=FakeEmbeddings(),
    )


def make_document(content: str, chunk_id: str, source: str) -> Document:
    """Create an indexed chunk with representative metadata."""
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "source": source,
            "source_path": f"policies/{source}",
            "file_type": "markdown",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    )


def indexed_policy_store(tmp_path: Path) -> Chroma:
    """Create a store containing three semantically distinct policies."""
    store = make_store(tmp_path)
    index_documents(
        store,
        [
            make_document("Employees may request annual leave.", "leave.md:na:0", "leave.md"),
            make_document(
                "Refund requests must include an invoice.",
                "refund.md:na:0",
                "refund.md",
            ),
            make_document(
                "Passwords must not be shared.",
                "security.md:na:0",
                "security.md",
            ),
        ],
    )
    return store


def test_relevant_retrieval_without_threshold(tmp_path: Path) -> None:
    result = retrieve_documents(
        indexed_policy_store(tmp_path),
        "How can I request a refund using my invoice?",
        k=3,
    )

    assert result.status is RetrievalStatus.RELEVANT_CONTEXT
    assert result.query == "How can I request a refund using my invoice?"
    assert result.chunks[0].document.metadata["chunk_id"] == "refund.md:na:0"
    assert result.chunks
    assert result.context


def test_relevant_retrieval_with_threshold(tmp_path: Path) -> None:
    store = indexed_policy_store(tmp_path)
    unfiltered = retrieve_documents(store, "refund invoice", k=3)
    maximum_distance = unfiltered.chunks[0].distance

    result = retrieve_documents(
        store,
        "refund invoice",
        k=3,
        maximum_distance=maximum_distance,
    )

    assert result.status is RetrievalStatus.RELEVANT_CONTEXT
    assert all(chunk.distance <= maximum_distance for chunk in result.chunks)


def test_threshold_rejects_all_results(monkeypatch) -> None:
    document = make_document("Annual leave", "leave.md:na:0", "leave.md")
    monkeypatch.setattr(
        retriever_module,
        "similarity_search_with_scores",
        lambda *args, **kwargs: [(document, 0.5)],
    )

    result = retrieve_documents(object(), "unrelated query", maximum_distance=0.0)

    assert result.status is RetrievalStatus.BELOW_THRESHOLD
    assert result.chunks == []
    assert result.context == ""
    assert result.searched_count == 1


def test_empty_store_returns_no_results(tmp_path: Path) -> None:
    result = retrieve_documents(make_store(tmp_path), "refund", k=4)

    assert result.status is RetrievalStatus.NO_RESULTS
    assert result.chunks == []
    assert result.context == ""
    assert result.searched_count == 0


def test_k_limits_results(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    index_documents(
        store,
        [
            make_document(f"Policy document {index}", f"policy.md:na:{index}", "policy.md")
            for index in range(4)
        ],
    )

    result = retrieve_documents(store, "policy", k=2)

    assert result.searched_count <= 2
    assert len(result.chunks) <= 2


def test_ranking_is_preserved(monkeypatch) -> None:
    documents = [
        make_document("Second", "second.md:na:0", "second.md"),
        make_document("First", "first.md:na:0", "first.md"),
    ]
    monkeypatch.setattr(
        retriever_module,
        "similarity_search_with_scores",
        lambda *args, **kwargs: [(documents[0], 0.4), (documents[1], 0.2)],
    )

    result = retrieve_documents(object(), "query")

    assert [chunk.document for chunk in result.chunks] == documents


def test_filtering_preserves_original_ranking(monkeypatch) -> None:
    documents = [
        make_document(str(index), f"doc.md:na:{index}", "doc.md")
        for index in range(3)
    ]
    monkeypatch.setattr(
        retriever_module,
        "similarity_search_with_scores",
        lambda *args, **kwargs: list(zip(documents, [0.20, 0.60, 0.90])),
    )

    result = retrieve_documents(object(), "query", maximum_distance=0.70)

    assert [chunk.distance for chunk in result.chunks] == [0.20, 0.60]


def test_exact_threshold_boundary_is_accepted(monkeypatch) -> None:
    document = make_document("Boundary", "boundary.md:na:0", "boundary.md")
    monkeypatch.setattr(
        retriever_module,
        "similarity_search_with_scores",
        lambda *args, **kwargs: [(document, 0.7)],
    )

    result = retrieve_documents(object(), "query", maximum_distance=0.7)

    assert result.status is RetrievalStatus.RELEVANT_CONTEXT
    assert len(result.chunks) == 1


@pytest.mark.parametrize("query", ["", " ", "\t\n"])
def test_blank_query_is_rejected(query: str) -> None:
    with pytest.raises(ValueError, match="query"):
        retrieve_documents(object(), query)


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_is_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        retrieve_documents(object(), "query", k=k)


@pytest.mark.parametrize(
    "maximum_distance",
    [-0.1, float("nan"), float("inf"), float("-inf"), True, False],
)
def test_invalid_maximum_distance_is_rejected(maximum_distance) -> None:
    with pytest.raises(ValueError, match="maximum_distance"):
        retrieve_documents(
            object(),
            "query",
            maximum_distance=maximum_distance,
        )


def test_none_threshold_accepts_all_results(monkeypatch) -> None:
    documents = [
        make_document("Near", "near.md:na:0", "near.md"),
        make_document("Far", "far.md:na:0", "far.md"),
    ]
    monkeypatch.setattr(
        retriever_module,
        "similarity_search_with_scores",
        lambda *args, **kwargs: [(documents[0], 0.1), (documents[1], 9.0)],
    )

    result = retrieve_documents(object(), "query", maximum_distance=None)

    assert [chunk.document for chunk in result.chunks] == documents


def test_build_context_empty_input() -> None:
    assert build_context([]) == ""


def test_context_format_is_exact_and_deterministic() -> None:
    chunks = [
        RetrievedChunk(
            document=Document(
                page_content="Refund requests must include the original invoice.",
                metadata={"chunk_id": "refund.md:na:0", "source": "refund.md"},
            ),
            distance=0.1234564,
        ),
        RetrievedChunk(
            document=Document(
                page_content="Employees may request annual leave through the HR portal.",
                metadata={
                    "chunk_id": "employee.pdf:4:2",
                    "source": "employee.pdf",
                    "page": 4,
                },
            ),
            distance=0.456789,
        ),
    ]

    assert build_context(chunks) == (
        "[Chunk 1]\n"
        "chunk_id: refund.md:na:0\n"
        "source: refund.md\n"
        "page: N/A\n"
        "distance: 0.123456\n\n"
        "Refund requests must include the original invoice.\n\n"
        "[Chunk 2]\n"
        "chunk_id: employee.pdf:4:2\n"
        "source: employee.pdf\n"
        "page: 4\n"
        "distance: 0.456789\n\n"
        "Employees may request annual leave through the HR portal."
    )


def test_context_contains_required_fields() -> None:
    context = build_context(
        [
            RetrievedChunk(
                document=Document(
                    page_content="Content",
                    metadata={"chunk_id": "doc.md:na:0", "source": "doc.md"},
                ),
                distance=0.25,
            )
        ]
    )

    for expected in (
        "[Chunk 1]",
        "chunk_id: doc.md:na:0",
        "source: doc.md",
        "page: N/A",
        "distance: 0.250000",
        "Content",
    ):
        assert expected in context


def test_context_uses_missing_metadata_fallbacks() -> None:
    context = build_context(
        [RetrievedChunk(document=Document(page_content="Content"), distance=1.0)]
    )

    assert "chunk_id: unknown" in context
    assert "source: unknown" in context
    assert "page: N/A" in context


def test_page_zero_is_not_treated_as_missing() -> None:
    context = build_context(
        [
            RetrievedChunk(
                document=Document(page_content="Content", metadata={"page": 0}),
                distance=1.0,
            )
        ]
    )

    assert "page: 0" in context
    assert "page: N/A" not in context


def test_context_strips_only_outer_content_whitespace() -> None:
    context = build_context(
        [
            RetrievedChunk(
                document=Document(page_content="  First line.\nSecond line.  "),
                distance=1.0,
            )
        ]
    )

    assert context.endswith("First line.\nSecond line.")


def test_build_context_does_not_mutate_metadata() -> None:
    document = Document(
        page_content="Content",
        metadata={"source": "policy.md", "page": 0, "custom": "value"},
    )
    original = copy.deepcopy(document.metadata)

    build_context([RetrievedChunk(document=document, distance=0.5)])

    assert document.metadata == original


def test_searched_count_is_measured_before_filtering(monkeypatch) -> None:
    documents = [
        make_document(str(index), f"doc.md:na:{index}", "doc.md")
        for index in range(3)
    ]
    monkeypatch.setattr(
        retriever_module,
        "similarity_search_with_scores",
        lambda *args, **kwargs: list(zip(documents, [0.1, 0.5, 0.9])),
    )

    result = retrieve_documents(object(), "query", maximum_distance=0.2)

    assert result.searched_count == 3
    assert len(result.chunks) == 1


def test_distances_are_normalized_to_float(monkeypatch) -> None:
    document = make_document("Content", "doc.md:na:0", "doc.md")
    monkeypatch.setattr(
        retriever_module,
        "similarity_search_with_scores",
        lambda *args, **kwargs: [(document, 1)],
    )

    result = retrieve_documents(object(), "query")

    assert result.chunks[0].distance == 1.0
    assert isinstance(result.chunks[0].distance, float)