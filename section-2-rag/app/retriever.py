"""Distance-filtered document retrieval and deterministic context formatting."""

import math
from enum import Enum
from numbers import Real

from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict

from app.vector_store import similarity_search_with_scores


class RetrievalStatus(str, Enum):
    """Outcome of a vector-store retrieval attempt."""

    RELEVANT_CONTEXT = "relevant_context"
    NO_RESULTS = "no_results"
    BELOW_THRESHOLD = "below_threshold"


class RetrievedChunk(BaseModel):
    """A retrieved LangChain document and its Chroma distance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    document: Document
    distance: float


class RetrievalResult(BaseModel):
    """Structured result of distance-filtered document retrieval."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str
    chunks: list[RetrievedChunk]
    context: str
    searched_count: int
    status: RetrievalStatus


def _validate_retrieval(
    query: str,
    k: int,
    maximum_distance: float | None,
) -> float | None:
    """Validate retrieval arguments and normalize the optional distance."""
    if not query.strip():
        raise ValueError("query must not be blank")
    if k < 1:
        raise ValueError("k must be at least 1")
    if maximum_distance is None:
        return None
    if (
        isinstance(maximum_distance, bool)
        or not isinstance(maximum_distance, Real)
        or not math.isfinite(maximum_distance)
        or maximum_distance < 0
    ):
        raise ValueError("maximum_distance must be a finite number at least 0")
    return float(maximum_distance)


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as deterministic plain-text context."""
    blocks: list[str] = []
    for display_index, chunk in enumerate(chunks, start=1):
        metadata = chunk.document.metadata
        chunk_id = metadata.get("chunk_id") or "unknown"
        source = metadata.get("source") or "unknown"
        page = metadata.get("page")
        page_display = "N/A" if page is None else str(page)
        content = chunk.document.page_content.strip()
        blocks.append(
            "\n".join(
                [
                    f"[Chunk {display_index}]",
                    f"chunk_id: {chunk_id}",
                    f"source: {source}",
                    f"page: {page_display}",
                    f"distance: {chunk.distance:.6f}",
                    "",
                    content,
                ]
            )
        )
    return "\n\n".join(blocks)


def retrieve_documents(
    vector_store: Chroma,
    query: str,
    *,
    k: int = 4,
    maximum_distance: float | None = None,
) -> RetrievalResult:
    """Retrieve, optionally distance-filter, and format matching chunks."""
    threshold = _validate_retrieval(query, k, maximum_distance)
    search_results = similarity_search_with_scores(vector_store, query, k=k)
    retrieved = [
        RetrievedChunk(document=document, distance=float(distance))
        for document, distance in search_results[:k]
    ]
    searched_count = len(retrieved)

    if not retrieved:
        return RetrievalResult(
            query=query,
            chunks=[],
            context="",
            searched_count=0,
            status=RetrievalStatus.NO_RESULTS,
        )

    accepted = (
        retrieved
        if threshold is None
        else [chunk for chunk in retrieved if chunk.distance <= threshold]
    )
    if not accepted:
        return RetrievalResult(
            query=query,
            chunks=[],
            context="",
            searched_count=searched_count,
            status=RetrievalStatus.BELOW_THRESHOLD,
        )

    return RetrievalResult(
        query=query,
        chunks=accepted,
        context=build_context(accepted),
        searched_count=searched_count,
        status=RetrievalStatus.RELEVANT_CONTEXT,
    )