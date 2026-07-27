"""Tests for recursive PDF and Markdown document loading."""

from pathlib import Path

import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.loader import DocumentLoadError, load_documents


def write_pdf(path: Path, page_texts: list[str]) -> None:
    """Create a small searchable PDF using pypdf."""
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)

    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_reference}
                )
            }
        )
        content = DecodedStreamObject()
        escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content.set_data(
            f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("ascii")
        )
        page[NameObject("/Contents")] = writer._add_object(content)

    with path.open("wb") as output:
        writer.write(output)


def test_loads_markdown_file(tmp_path: Path) -> None:
    markdown = tmp_path / "guide.md"
    markdown.write_text("# Guide\n\nHello world.", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "# Guide\n\nHello world."
    assert documents[0].metadata == {
        "source": "guide.md",
        "source_path": "guide.md",
        "file_type": "markdown",
    }


def test_loads_one_document_per_pdf_page(tmp_path: Path) -> None:
    pdf = tmp_path / "handbook.pdf"
    write_pdf(pdf, ["First page policy", "Second page procedure"])

    documents = load_documents(tmp_path)

    assert len(documents) == 2
    assert [document.metadata["page"] for document in documents] == [0, 1]
    assert all(document.metadata["source"] == "handbook.pdf" for document in documents)
    assert all(document.metadata["source_path"] == "handbook.pdf" for document in documents)
    assert all(document.metadata["file_type"] == "pdf" for document in documents)
    assert "First page policy" in documents[0].page_content
    assert "Second page procedure" in documents[1].page_content


def test_discovers_files_recursively(tmp_path: Path) -> None:
    nested = tmp_path / "policies" / "regional"
    nested.mkdir(parents=True)
    (nested / "leave.markdown").write_text("Nested policy", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].metadata["source_path"] == "policies/regional/leave.markdown"


def test_ignores_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("Ignore me", encoding="utf-8")
    (tmp_path / "guide.md").write_text("Load me", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [document.metadata["source"] for document in documents] == ["guide.md"]


def test_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    assert load_documents(tmp_path) == []


def test_missing_directory_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")


def test_file_path_raises_not_a_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file.md"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        load_documents(file_path)


def test_results_follow_sorted_paths_and_pdf_page_order(tmp_path: Path) -> None:
    (tmp_path / "z-last.md").write_text("Last", encoding="utf-8")
    write_pdf(tmp_path / "b-middle.pdf", ["PDF page one", "PDF page two"])
    (tmp_path / "a-first.markdown").write_text("First", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [
        (document.metadata["source"], document.metadata.get("page"))
        for document in documents
    ] == [
        ("a-first.markdown", None),
        ("b-middle.pdf", 0),
        ("b-middle.pdf", 1),
        ("z-last.md", None),
    ]


def test_supports_uppercase_extensions(tmp_path: Path) -> None:
    (tmp_path / "POLICY.MD").write_text("Uppercase Markdown", encoding="utf-8")
    write_pdf(tmp_path / "GUIDE.PDF", ["Uppercase PDF"])

    documents = load_documents(tmp_path)

    assert [document.metadata["source"] for document in documents] == [
        "GUIDE.PDF",
        "POLICY.MD",
    ]
    assert "Uppercase PDF" in documents[0].page_content
    assert documents[1].page_content == "Uppercase Markdown"


def test_broken_pdf_raises_document_load_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a valid PDF")

    with pytest.raises(DocumentLoadError, match="broken.pdf") as error:
        load_documents(tmp_path)

    assert error.value.__cause__ is not None