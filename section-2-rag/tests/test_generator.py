"""Tests for grounded structured answer generation."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.generator import (
    INSUFFICIENT_CONTEXT_ANSWER,
    GeneratorOutputError,
    GeneratorResult,
    build_generator_prompt,
    generate_answer,
)
from app.retriever import RetrievedChunk, RetrievalResult, RetrievalStatus, build_context


class FakeStructuredRunnable:
    """Structured runnable that records invocations and returns prepared output."""

    def __init__(self, owner: FakeChatModel) -> None:
        self.owner = owner

    def invoke(self, messages: list[Any]) -> Any:
        self.owner.invocation_count += 1
        self.owner.messages = messages
        if self.owner.error is not None:
            raise self.owner.error
        return self.owner.response


class FakeChatModel:
    """Minimal structured-output chat-model test double."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.schema: type[Any] | None = None
        self.messages: list[Any] = []
        self.invocation_count = 0

    def with_structured_output(self, schema: type[Any]) -> FakeStructuredRunnable:
        self.schema = schema
        return FakeStructuredRunnable(self)


def make_chunk(
    chunk_id: object = "refund.md:na:0",
    *,
    content: str = "Refund requests require an invoice.",
    source: str = "refund.md",
) -> RetrievedChunk:
    """Create a retrieved chunk with configurable ID metadata."""
    metadata: dict[str, object] = {"source": source}
    if chunk_id is not None:
        metadata["chunk_id"] = chunk_id
    return RetrievedChunk(
        document=Document(page_content=content, metadata=metadata),
        distance=0.2,
    )


def make_retrieval(
    *,
    status: RetrievalStatus = RetrievalStatus.RELEVANT_CONTEXT,
    chunks: list[RetrievedChunk] | None = None,
    context: str | None = None,
) -> RetrievalResult:
    """Create a consistent retrieval result for generator tests."""
    resolved_chunks = [make_chunk()] if chunks is None else chunks
    resolved_context = build_context(resolved_chunks) if context is None else context
    return RetrievalResult(
        query="How do refunds work?",
        chunks=resolved_chunks,
        context=resolved_context,
        searched_count=len(resolved_chunks),
        status=status,
    )


def supported_result(
    *,
    answer: str = "Refund requests require an invoice.",
    chunk_ids: list[str] | None = None,
) -> GeneratorResult:
    """Create a supported structured model response."""
    return GeneratorResult(
        answer=answer,
        supported=True,
        chunk_ids=["refund.md:na:0"] if chunk_ids is None else chunk_ids,
        insufficient_context=False,
    )


def assert_fallback(result: GeneratorResult) -> None:
    """Assert the deterministic insufficient-context invariant."""
    assert result == GeneratorResult(
        answer=INSUFFICIENT_CONTEXT_ANSWER,
        supported=False,
        chunk_ids=[],
        insufficient_context=True,
    )


def test_successful_supported_generation() -> None:
    model = FakeChatModel(supported_result())

    result = generate_answer(model, "How do refunds work?", make_retrieval())

    assert isinstance(result, GeneratorResult)
    assert result.answer == "Refund requests require an invoice."
    assert result.supported is True
    assert result.insufficient_context is False
    assert result.chunk_ids == ["refund.md:na:0"]
    assert model.invocation_count == 1


@pytest.mark.parametrize(
    "status",
    [RetrievalStatus.NO_RESULTS, RetrievalStatus.BELOW_THRESHOLD],
)
def test_unusable_retrieval_status_bypasses_llm(status: RetrievalStatus) -> None:
    model = FakeChatModel(supported_result())
    retrieval = make_retrieval(status=status, chunks=[], context="")

    result = generate_answer(model, "Question", retrieval)

    assert_fallback(result)
    assert model.invocation_count == 0


def test_relevant_status_with_empty_chunks_bypasses_llm() -> None:
    model = FakeChatModel(supported_result())
    retrieval = make_retrieval(chunks=[], context="Context without chunks")

    assert_fallback(generate_answer(model, "Question", retrieval))
    assert model.invocation_count == 0


def test_relevant_status_with_blank_context_bypasses_llm() -> None:
    model = FakeChatModel(supported_result())
    retrieval = make_retrieval(context="   ")

    assert_fallback(generate_answer(model, "Question", retrieval))
    assert model.invocation_count == 0


@pytest.mark.parametrize("question", ["", " ", "\t\n"])
def test_blank_question_is_rejected_without_llm_call(question: str) -> None:
    model = FakeChatModel(supported_result())

    with pytest.raises(ValueError, match="question"):
        generate_answer(model, question, make_retrieval())

    assert model.invocation_count == 0


def test_prompt_contains_normalized_question() -> None:
    messages = build_generator_prompt("  How do refunds work?  ", make_retrieval())

    assert isinstance(messages[1], HumanMessage)
    assert "Question:\nHow do refunds work?" in messages[1].content


def test_prompt_contains_exact_retrieval_context() -> None:
    retrieval = make_retrieval()

    messages = build_generator_prompt("Question", retrieval)

    assert retrieval.context in messages[1].content


def test_prompt_lists_authoritative_chunk_ids() -> None:
    chunks = [
        make_chunk("refund.md:na:0"),
        make_chunk("leave.md:na:1", source="leave.md"),
    ]

    messages = build_generator_prompt("Question", make_retrieval(chunks=chunks))

    assert "- refund.md:na:0" in messages[1].content
    assert "- leave.md:na:1" in messages[1].content


def test_prompt_omits_missing_non_string_and_blank_ids() -> None:
    chunks = [make_chunk(None), make_chunk(123), make_chunk("   ")]

    messages = build_generator_prompt(
        "Question",
        make_retrieval(chunks=chunks, context="Context"),
    )

    human = messages[1].content
    assert "- 123" not in human
    assert "Available chunk IDs (authoritative):\n- (none)" in human


def test_system_prompt_contains_grounding_and_injection_protections() -> None:
    messages = build_generator_prompt("Question", make_retrieval())

    assert isinstance(messages[0], SystemMessage)
    system = " ".join(messages[0].content.lower().split())
    for phrase in (
        "answer only from",
        "outside knowledge",
        "never invent",
        "untrusted source material",
        "must not override",
        "hidden prompts",
    ):
        assert phrase in system


def test_structured_output_schema_is_passed() -> None:
    model = FakeChatModel(supported_result())

    generate_answer(model, "Question", make_retrieval())

    assert model.schema is GeneratorResult


def test_answer_outer_whitespace_is_stripped() -> None:
    result = generate_answer(
        FakeChatModel(supported_result(answer="  Supported answer.  ")),
        "Question",
        make_retrieval(),
    )

    assert result.answer == "Supported answer."


def test_chunk_id_whitespace_is_stripped() -> None:
    result = generate_answer(
        FakeChatModel(supported_result(chunk_ids=["  refund.md:na:0  "])),
        "Question",
        make_retrieval(),
    )

    assert result.chunk_ids == ["refund.md:na:0"]


def test_duplicate_chunk_ids_are_removed_in_first_occurrence_order() -> None:
    retrieval = make_retrieval(
        chunks=[make_chunk("refund.md:na:0"), make_chunk("leave.md:na:0")]
    )
    response = supported_result(
        chunk_ids=["refund.md:na:0", "leave.md:na:0", "refund.md:na:0"]
    )

    result = generate_answer(FakeChatModel(response), "Question", retrieval)

    assert result.chunk_ids == ["refund.md:na:0", "leave.md:na:0"]


def test_unknown_chunk_id_raises_generator_output_error() -> None:
    response = supported_result(chunk_ids=["invented.md:99:3"])

    with pytest.raises(GeneratorOutputError, match="unknown chunk ID"):
        generate_answer(FakeChatModel(response), "Question", make_retrieval())


def test_supported_output_without_chunk_ids_is_rejected() -> None:
    with pytest.raises(GeneratorOutputError, match="at least one chunk ID"):
        generate_answer(
            FakeChatModel(supported_result(chunk_ids=[])),
            "Question",
            make_retrieval(),
        )


def test_supported_output_with_blank_answer_is_rejected() -> None:
    with pytest.raises(GeneratorOutputError, match="non-blank answer"):
        generate_answer(
            FakeChatModel(supported_result(answer="  ")),
            "Question",
            make_retrieval(),
        )


def test_insufficient_context_output_takes_precedence_over_invented_id() -> None:
    response = GeneratorResult(
        answer="Custom fallback",
        supported=False,
        chunk_ids=["invented.md:99:3"],
        insufficient_context=True,
    )

    result = generate_answer(FakeChatModel(response), "Question", make_retrieval())

    assert_fallback(result)


def test_unsupported_output_normalizes_to_fallback() -> None:
    response = GeneratorResult(
        answer="I am not sure.",
        supported=False,
        chunk_ids=[],
        insufficient_context=False,
    )

    assert_fallback(generate_answer(FakeChatModel(response), "Question", make_retrieval()))


def test_unsupported_output_cannot_retain_citations() -> None:
    response = GeneratorResult(
        answer="Unsupported",
        supported=False,
        chunk_ids=["refund.md:na:0"],
        insufficient_context=False,
    )

    assert_fallback(generate_answer(FakeChatModel(response), "Question", make_retrieval()))


def test_all_available_ids_are_accepted() -> None:
    retrieval = make_retrieval(
        chunks=[make_chunk("refund.md:na:0"), make_chunk("leave.md:na:0")]
    )

    result = generate_answer(
        FakeChatModel(
            supported_result(chunk_ids=["refund.md:na:0", "leave.md:na:0"])
        ),
        "Question",
        retrieval,
    )

    assert result.chunk_ids == ["refund.md:na:0", "leave.md:na:0"]


def test_available_id_prompt_order_follows_retrieval_order() -> None:
    retrieval = make_retrieval(
        chunks=[make_chunk("leave.md:na:0"), make_chunk("refund.md:na:0")]
    )

    human = build_generator_prompt("Question", retrieval)[1].content

    assert human.index("- leave.md:na:0") < human.index("- refund.md:na:0")


def test_available_ids_are_deduplicated_in_prompt() -> None:
    retrieval = make_retrieval(
        chunks=[make_chunk("refund.md:na:0"), make_chunk("refund.md:na:0")]
    )

    human = build_generator_prompt("Question", retrieval)[1].content

    assert human.count("- refund.md:na:0") == 1


def test_retrieval_result_and_metadata_are_not_mutated() -> None:
    retrieval = make_retrieval()
    original = copy.deepcopy(retrieval)

    generate_answer(FakeChatModel(supported_result()), "Question", retrieval)

    assert retrieval == original
    assert retrieval.chunks[0].document.metadata == original.chunks[0].document.metadata


def test_prompt_preserves_arabic_question() -> None:
    question = "ما هي سياسة استرداد المبلغ؟"

    human = build_generator_prompt(question, make_retrieval())[1].content

    assert question in human


def test_provider_error_propagates_unchanged() -> None:
    model = FakeChatModel(error=RuntimeError("provider unavailable"))

    with pytest.raises(RuntimeError, match="provider unavailable"):
        generate_answer(model, "Question", make_retrieval())


def test_mapping_output_is_accepted() -> None:
    response = {
        "answer": "Refund requests require an invoice.",
        "supported": True,
        "chunk_ids": ["refund.md:na:0"],
        "insufficient_context": False,
    }

    result = generate_answer(FakeChatModel(response), "Question", make_retrieval())

    assert result == supported_result()


def test_malformed_output_raises_generator_output_error() -> None:
    with pytest.raises(GeneratorOutputError, match="malformed structured output"):
        generate_answer(FakeChatModel(object()), "Question", make_retrieval())