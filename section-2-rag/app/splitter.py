"""Split loaded LangChain documents into metadata-rich chunks."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _chunk_id(metadata: dict[str, object], chunk_index: int) -> str:
    """Build a deterministic chunk identifier from source metadata."""
    source = str(metadata.get("source", "unknown"))
    page = metadata.get("page")
    page_part = "na" if page is None else str(page)
    return f"{source}:{page_part}:{chunk_index}"


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split documents in order while preserving and extending metadata."""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    output: list[Document] = []

    for document in documents:
        chunks = splitter.split_documents([document])
        total_chunks = len(chunks)
        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "chunk_id": _chunk_id(document.metadata, chunk_index),
                }
            )
            output.append(chunk)

    return output