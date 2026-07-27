"""Tests for deterministic LangChain document splitting."""

from itertools import pairwise

from langchain_core.documents import Document

from app.splitter import split_documents


def test_empty_input_returns_empty_list() -> None:
    assert split_documents([], chunk_size=100, chunk_overlap=20) == []


def test_short_document_returns_one_metadata_rich_chunk() -> None:
    metadata = {
        "source": "policy.md",
        "source_path": "policies/policy.md",
        "file_type": "markdown",
        "department": "people",
    }
    document = Document(page_content="Short policy text.", metadata=metadata)

    chunks = split_documents([document], chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 1
    assert chunks[0].page_content == "Short policy text."
    assert chunks[0].metadata == {
        **metadata,
        "chunk_index": 0,
        "total_chunks": 1,
        "chunk_id": "policy.md:na:0",
    }


def test_large_document_has_incrementing_indexes_and_total() -> None:
    document = Document(
        page_content="ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 5,
        metadata={"source": "large.md"},
    )

    chunks = split_documents([document], chunk_size=30, chunk_overlap=5)

    assert len(chunks) > 1
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert {chunk.metadata["total_chunks"] for chunk in chunks} == {len(chunks)}


def test_preserves_required_metadata() -> None:
    metadata = {
        "source": "employee.pdf",
        "source_path": "handbooks/employee.pdf",
        "file_type": "pdf",
        "page": 4,
    }

    chunks = split_documents(
        [Document(page_content="A" * 80, metadata=metadata)],
        chunk_size=25,
        chunk_overlap=5,
    )

    for chunk in chunks:
        for key, value in metadata.items():
            assert chunk.metadata[key] == value


def test_chunks_preserve_original_document_order() -> None:
    documents = [
        Document(page_content="A" * 45, metadata={"source": "first.md"}),
        Document(page_content="B" * 45, metadata={"source": "second.md"}),
    ]

    chunks = split_documents(documents, chunk_size=20, chunk_overlap=5)

    sources = [chunk.metadata["source"] for chunk in chunks]
    first_second = sources.index("second.md")
    assert sources[:first_second] == ["first.md"] * first_second
    assert sources[first_second:] == ["second.md"] * (len(sources) - first_second)


def test_chunk_numbering_restarts_for_each_document() -> None:
    documents = [
        Document(page_content="A" * 35, metadata={"source": "first.md"}),
        Document(page_content="B" * 35, metadata={"source": "second.md"}),
    ]

    chunks = split_documents(documents, chunk_size=20, chunk_overlap=5)

    first_indexes = [
        chunk.metadata["chunk_index"]
        for chunk in chunks
        if chunk.metadata["source"] == "first.md"
    ]
    second_indexes = [
        chunk.metadata["chunk_index"]
        for chunk in chunks
        if chunk.metadata["source"] == "second.md"
    ]
    assert first_indexes == list(range(len(first_indexes)))
    assert second_indexes == list(range(len(second_indexes)))


def test_chunk_ids_match_required_format() -> None:
    document = Document(
        page_content="A" * 50,
        metadata={"source": "employee.pdf", "page": 4},
    )

    chunks = split_documents([document], chunk_size=20, chunk_overlap=5)

    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [
        f"employee.pdf:4:{index}" for index in range(len(chunks))
    ]


def test_neighboring_chunks_contain_overlap() -> None:
    document = Document(
        page_content="ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 3,
        metadata={"source": "letters.md"},
    )

    chunks = split_documents([document], chunk_size=20, chunk_overlap=5)

    assert len(chunks) > 1
    for current, following in pairwise(chunks):
        assert current.page_content[-5:] == following.page_content[:5]


def test_page_less_document_uses_na_in_chunk_id() -> None:
    document = Document(
        page_content="Page-less content",
        metadata={"source": "policy.md"},
    )

    chunks = split_documents([document], chunk_size=100, chunk_overlap=10)

    assert chunks[0].metadata["chunk_id"] == "policy.md:na:0"


def test_output_is_deterministic() -> None:
    documents = [
        Document(
            page_content="Deterministic content " * 10,
            metadata={"source": "stable.md"},
        )
    ]

    first = split_documents(documents, chunk_size=45, chunk_overlap=10)
    second = split_documents(documents, chunk_size=45, chunk_overlap=10)

    assert [
        (chunk.page_content, chunk.metadata["chunk_id"]) for chunk in first
    ] == [
        (chunk.page_content, chunk.metadata["chunk_id"]) for chunk in second
    ]