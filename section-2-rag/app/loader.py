"""Load supported knowledge-base documents into LangChain documents."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown"}


class DocumentLoadError(Exception):
    """Raised when a supported document cannot be loaded."""


def _discover_supported_files(documents_dir: Path) -> list[Path]:
    """Return supported files recursively in deterministic path order."""
    return sorted(
        path
        for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _load_pdf(path: Path, documents_dir: Path) -> list[Document]:
    """Load one LangChain document per PDF page."""
    pages = PyPDFLoader(str(path)).load()
    source_path = path.relative_to(documents_dir).as_posix()

    for page_number, page in enumerate(pages):
        page.metadata.update(
            {
                "source": path.name,
                "source_path": source_path,
                "file_type": "pdf",
                "page": page_number,
            }
        )
    return pages


def _load_markdown(path: Path, documents_dir: Path) -> list[Document]:
    """Load an entire UTF-8 Markdown file as one document."""
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={
                "source": path.name,
                "source_path": path.relative_to(documents_dir).as_posix(),
                "file_type": "markdown",
            },
        )
    ]


def load_documents(documents_dir: Path) -> list[Document]:
    """Load supported files recursively from ``documents_dir``."""
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {documents_dir}")
    if not documents_dir.is_dir():
        raise NotADirectoryError(f"Documents path is not a directory: {documents_dir}")

    documents: list[Document] = []
    for path in _discover_supported_files(documents_dir):
        try:
            if path.suffix.lower() == ".pdf":
                documents.extend(_load_pdf(path, documents_dir))
            else:
                documents.extend(_load_markdown(path, documents_dir))
        except Exception as exc:
            raise DocumentLoadError(f"Failed to load document: {path.name}") from exc

    return documents