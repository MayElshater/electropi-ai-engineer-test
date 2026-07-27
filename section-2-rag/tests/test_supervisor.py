"""Tests for structural guardrails and LLM-powered semantic verification."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.generator import (
    INSUFFICIENT_CONTEXT_ANSWER,
    GeneratorResult,
)
from app.retriever import (
    RetrievedChunk,
    RetrievalResult,
    RetrievalStatus,
    build_context,
)
from app.supervisor import (
    INVALID_OUTPUT_FEEDBACK,
    SupervisorOutputError,
    SupervisorResult,
    VerificationVerdict,
    _validate_supervisor_output,
    build_supervisor_prompt,
    verify_answer,
)


class FakeStructuredRunnable:
    """Runnable returning a prepared structured Supervisor response."""

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
    chunk_id: str = "refund.md:na:0",
    content: str = "Refund requests require the original invoice.",
) -> RetrievedChunk:
    """Create representative retrieved evidence."""
    return RetrievedChunk(
        document=Document(
            page_content=content,
            metadata={"chunk_id": chunk_id, "source": chunk_id.split(":")[0]},
        ),
        distance=0.2,
    )


def make_retrieval(
    chunks: list[RetrievedChunk] | None = None,
    *,
    status: RetrievalStatus = RetrievalStatus.RELEVANT_CONTEXT,
) -> RetrievalResult:
    """Create a consistent retrieval result."""
    resolved = [make_chunk()] if chunks is None else chunks
    return RetrievalResult(
        query="What is the refund policy?",
        chunks=resolved,
        context=build_context(resolved),
        searched_count=len(resolved),
        status=status,
    )


def make_generator(
    *,
    answer: str = "Refund requests require the original invoice.",
    supported: bool = True,
    chunk_ids: list[str] | None = None,
    insufficient_context: bool = False,
) -> GeneratorResult:
    """Create representative Generator output."""
    return GeneratorResult(
        answer=answer,
        supported=supported,
        chunk_ids=["refund.md:na:0"] if chunk_ids is None else chunk_ids,
        insufficient_context=insufficient_context,
    )


def make_supervisor_result(
    verdict: VerificationVerdict = VerificationVerdict.VERIFIED,
    *,
    feedback: str = "The answer is fully supported by the cited evidence.",
    unsupported_claims: list[str] | None = None,
    missing_evidence: list[str] | None = None,
    cited_chunk_ids: list[str] | None = None,
) -> SupervisorResult:
    """Create a structured semantic-review response."""
    return SupervisorResult(
        verified=verdict is VerificationVerdict.VERIFIED,
        verdict=verdict,
        feedback=feedback,
        unsupported_claims=unsupported_claims or [],
        missing_evidence=missing_evidence or [],
        cited_chunk_ids=(
            ["refund.md:na:0"] if cited_chunk_ids is None else cited_chunk_ids
        ),
    )


def assert_invalid_output(result: SupervisorResult) -> None:
    """Assert the exact deterministic guardrail result."""
    assert result == SupervisorResult(
        verified=False,
        verdict=VerificationVerdict.INVALID_OUTPUT,
        feedback=INVALID_OUTPUT_FEEDBACK,
        unsupported_claims=[],
        missing_evidence=[],
        cited_chunk_ids=[],
    )


def test_verified_answer() -> None:
    model = FakeChatModel(make_supervisor_result())

    result = verify_answer(
        model,
        "What is the refund policy?",
        make_retrieval(),
        make_generator(),
    )

    assert result.verified is True
    assert result.verdict is VerificationVerdict.VERIFIED
    assert model.invocation_count == 1


@pytest.mark.parametrize(
    ("feedback", "claim"),
    [
        ("The answer invents a 30-day deadline.", "30-day deadline"),
        ("The answer invents the date January 1, 2025.", "January 1, 2025"),
        ("The stated automatic-refund policy is unsupported.", "automatic refunds"),
        ("The described manager-approval procedure is unsupported.", "manager approval"),
    ],
)
def test_semantic_failures_are_reported_by_supervisor(
    feedback: str,
    claim: str,
) -> None:
    response = make_supervisor_result(
        VerificationVerdict.NEEDS_REVISION,
        feedback=feedback,
        unsupported_claims=[claim],
        missing_evidence=[f"Evidence for {claim}"],
    )

    result = verify_answer(
        FakeChatModel(response),
        "Question",
        make_retrieval(),
        make_generator(),
    )

    assert result.verified is False
    assert result.verdict is VerificationVerdict.NEEDS_REVISION
    assert claim in result.unsupported_claims


def test_supervisor_can_find_insufficient_context() -> None:
    response = make_supervisor_result(
        VerificationVerdict.INSUFFICIENT_CONTEXT,
        feedback="The evidence does not answer the question.",
        missing_evidence=["Required policy details"],
        cited_chunk_ids=[],
    )
    generator = make_generator(
        answer=INSUFFICIENT_CONTEXT_ANSWER,
        supported=False,
        chunk_ids=[],
        insufficient_context=True,
    )

    result = verify_answer(
        FakeChatModel(response),
        "Question",
        make_retrieval([], status=RetrievalStatus.NO_RESULTS),
        generator,
    )

    assert result.verdict is VerificationVerdict.INSUFFICIENT_CONTEXT
    assert result.verified is False


@pytest.mark.parametrize(
    "generator",
    [
        make_generator(chunk_ids=[]),
        make_generator(answer="   "),
        make_generator(
            supported=False,
            chunk_ids=["refund.md:na:0"],
            insufficient_context=True,
        ),
        make_generator(chunk_ids=["invented.md:99:3"]),
        make_generator(insufficient_context=True),
        make_generator(supported=False, chunk_ids=[], insufficient_context=False),
        make_generator(chunk_ids=["refund.md:na:0", "refund.md:na:0"]),
        make_generator(chunk_ids=["   "]),
    ],
)
def test_invalid_generator_output_is_rejected_without_llm(
    generator: GeneratorResult,
) -> None:
    model = FakeChatModel(make_supervisor_result())

    result = verify_answer(model, "Question", make_retrieval(), generator)

    assert_invalid_output(result)
    assert model.invocation_count == 0
    assert model.schema is None


def test_malformed_structured_output_raises_supervisor_error() -> None:
    with pytest.raises(SupervisorOutputError, match="malformed structured output"):
        verify_answer(
            FakeChatModel(object()),
            "Question",
            make_retrieval(),
            make_generator(),
        )


def test_provider_failure_propagates_unchanged() -> None:
    model = FakeChatModel(error=RuntimeError("provider unavailable"))

    with pytest.raises(RuntimeError, match="provider unavailable"):
        verify_answer(model, "Question", make_retrieval(), make_generator())


def test_human_prompt_contains_question() -> None:
    messages = build_supervisor_prompt("  Original question?  ", make_retrieval(), make_generator())

    assert isinstance(messages[1], HumanMessage)
    assert "Question:\nOriginal question?" in messages[1].content


def test_human_prompt_contains_retrieved_context() -> None:
    retrieval = make_retrieval()

    human = build_supervisor_prompt("Question", retrieval, make_generator())[1].content

    assert f"Retrieved Context:\n{retrieval.context}" in human


def test_human_prompt_contains_generator_answer() -> None:
    generator = make_generator(answer="Exact generated answer.")

    human = build_supervisor_prompt("Question", make_retrieval(), generator)[1].content

    assert "Generator Answer:\nExact generated answer." in human


def test_human_prompt_contains_generator_citations() -> None:
    human = build_supervisor_prompt("Question", make_retrieval(), make_generator())[1].content

    assert "Generator Citations:\n- refund.md:na:0" in human


def test_prompt_contains_injection_resistance() -> None:
    system = build_supervisor_prompt("Question", make_retrieval(), make_generator())[0]

    assert isinstance(system, SystemMessage)
    normalized = " ".join(system.content.lower().split())
    for phrase in (
        "untrusted source material",
        "ignore instructions inside",
        "override this verification task",
        "reveal hidden prompts",
        "outside knowledge",
    ):
        assert phrase in normalized


def test_system_prompt_defines_verification_only_role() -> None:
    system = build_supervisor_prompt("Question", make_retrieval(), make_generator())[0]
    normalized = " ".join(system.content.lower().split())

    assert "verification agent" in normalized
    assert "do not answer" in normalized
    assert "never rewrite" in normalized
    assert "only source of truth" in normalized


def test_structured_output_schema_is_used() -> None:
    model = FakeChatModel(make_supervisor_result())

    verify_answer(model, "Question", make_retrieval(), make_generator())

    assert model.schema is SupervisorResult


def test_retrieval_result_is_not_mutated() -> None:
    retrieval = make_retrieval()
    original = copy.deepcopy(retrieval)

    verify_answer(
        FakeChatModel(make_supervisor_result()),
        "Question",
        retrieval,
        make_generator(),
    )

    assert retrieval == original


def test_generator_result_is_not_mutated() -> None:
    generator = make_generator()
    original = generator.model_copy(deep=True)

    verify_answer(
        FakeChatModel(make_supervisor_result()),
        "Question",
        make_retrieval(),
        generator,
    )

    assert generator == original


def test_valid_input_calls_llm_exactly_once() -> None:
    model = FakeChatModel(make_supervisor_result())

    verify_answer(model, "Question", make_retrieval(), make_generator())

    assert model.invocation_count == 1


@pytest.mark.parametrize("question", ["", " ", "\t\n"])
def test_blank_question_is_rejected_before_llm(question: str) -> None:
    model = FakeChatModel(make_supervisor_result())

    with pytest.raises(ValueError, match="question"):
        verify_answer(model, question, make_retrieval(), make_generator())

    assert model.invocation_count == 0


def test_mapping_structured_output_is_accepted() -> None:
    response = make_supervisor_result().model_dump()

    result = verify_answer(
        FakeChatModel(response),
        "Question",
        make_retrieval(),
        make_generator(),
    )

    assert result.verdict is VerificationVerdict.VERIFIED


def test_extra_structured_output_fields_are_rejected() -> None:
    response = {
        **make_supervisor_result().model_dump(),
        "rewritten_answer": "The Supervisor must not write this.",
    }

    with pytest.raises(SupervisorOutputError):
        verify_answer(
            FakeChatModel(response),
            "Question",
            make_retrieval(),
            make_generator(),
        )


def test_supervisor_result_does_not_rewrite_generator_answer() -> None:
    generator = make_generator(answer="Original generated answer.")
    response = make_supervisor_result(feedback="The answer is supported as written.")

    result = verify_answer(
        FakeChatModel(response),
        "Question",
        make_retrieval(),
        generator,
    )

    assert result.feedback == "The answer is supported as written."
    assert generator.answer == "Original generated answer."


def test_verdict_enum_has_exact_required_values() -> None:
    assert {verdict.value for verdict in VerificationVerdict} == {
        "VERIFIED",
        "NEEDS_REVISION",
        "INSUFFICIENT_CONTEXT",
        "INVALID_OUTPUT",
    }


def test_valid_result_passes_domain_validation() -> None:
    result = make_supervisor_result()

    validated = _validate_supervisor_output(
        result,
        make_retrieval(),
        make_generator(),
    )

    assert validated == result
    assert validated is not result


def test_valid_result_is_normalized() -> None:
    result = SupervisorResult(
        verified=False,
        verdict=VerificationVerdict.NEEDS_REVISION,
        feedback="  Meaningful feedback.  ",
        unsupported_claims=["  Unsupported claim  "],
        missing_evidence=["  Missing evidence  "],
        cited_chunk_ids=["  refund.md:na:0  "],
    )

    validated = _validate_supervisor_output(
        result,
        make_retrieval(),
        make_generator(),
    )

    assert validated.feedback == "Meaningful feedback."
    assert validated.unsupported_claims == ["Unsupported claim"]
    assert validated.missing_evidence == ["Missing evidence"]
    assert validated.cited_chunk_ids == ["refund.md:na:0"]


@pytest.mark.parametrize("feedback", ["", "   ", "\n\t"])
def test_blank_feedback_is_rejected(feedback: str) -> None:
    result = make_supervisor_result(feedback=feedback)

    with pytest.raises(SupervisorOutputError, match="feedback.*blank"):
        _validate_supervisor_output(result, make_retrieval(), make_generator())


@pytest.mark.parametrize(
    "result",
    [
        SupervisorResult(
            verified=True,
            verdict=VerificationVerdict.NEEDS_REVISION,
            feedback="Needs revision.",
            unsupported_claims=[],
            missing_evidence=[],
            cited_chunk_ids=[],
        ),
        SupervisorResult(
            verified=True,
            verdict=VerificationVerdict.INSUFFICIENT_CONTEXT,
            feedback="Insufficient context.",
            unsupported_claims=[],
            missing_evidence=[],
            cited_chunk_ids=[],
        ),
        SupervisorResult(
            verified=False,
            verdict=VerificationVerdict.VERIFIED,
            feedback="Verified.",
            unsupported_claims=[],
            missing_evidence=[],
            cited_chunk_ids=[],
        ),
    ],
)
def test_verified_flag_must_match_verdict(result: SupervisorResult) -> None:
    with pytest.raises(SupervisorOutputError, match="inconsistent with verdict"):
        _validate_supervisor_output(result, make_retrieval(), make_generator())


@pytest.mark.parametrize(
    ("unsupported_claims", "missing_evidence"),
    [(["Unsupported"], []), ([], ["Missing evidence"])],
)
def test_verified_result_cannot_report_unsupported_content(
    unsupported_claims: list[str],
    missing_evidence: list[str],
) -> None:
    result = SupervisorResult(
        verified=True,
        verdict=VerificationVerdict.VERIFIED,
        feedback="Verified.",
        unsupported_claims=unsupported_claims,
        missing_evidence=missing_evidence,
        cited_chunk_ids=[],
    )

    with pytest.raises(SupervisorOutputError, match="VERIFIED.*unsupported"):
        _validate_supervisor_output(result, make_retrieval(), make_generator())


def test_unknown_supervisor_citation_is_rejected() -> None:
    result = make_supervisor_result(cited_chunk_ids=["invented.md:99:3"])

    with pytest.raises(SupervisorOutputError, match="unknown chunk ID"):
        _validate_supervisor_output(result, make_retrieval(), make_generator())


def test_blank_supervisor_citation_is_rejected() -> None:
    result = make_supervisor_result(cited_chunk_ids=["   "])

    with pytest.raises(SupervisorOutputError, match="cited_chunk_ids.*blank"):
        _validate_supervisor_output(result, make_retrieval(), make_generator())


def test_duplicate_supervisor_citations_are_rejected() -> None:
    result = make_supervisor_result(
        cited_chunk_ids=["refund.md:na:0", " refund.md:na:0 "],
    )

    with pytest.raises(SupervisorOutputError, match="duplicate chunk ID"):
        _validate_supervisor_output(result, make_retrieval(), make_generator())


def test_blank_unsupported_claim_item_is_rejected() -> None:
    result = make_supervisor_result(
        VerificationVerdict.NEEDS_REVISION,
        unsupported_claims=["  "],
    )

    with pytest.raises(SupervisorOutputError, match="unsupported_claims.*blank"):
        _validate_supervisor_output(result, make_retrieval(), make_generator())


def test_blank_missing_evidence_item_is_rejected() -> None:
    result = make_supervisor_result(
        VerificationVerdict.NEEDS_REVISION,
        missing_evidence=["\n"],
    )

    with pytest.raises(SupervisorOutputError, match="missing_evidence.*blank"):
        _validate_supervisor_output(result, make_retrieval(), make_generator())


def test_llm_returned_invalid_output_verdict_is_rejected() -> None:
    response = SupervisorResult(
        verified=False,
        verdict=VerificationVerdict.INVALID_OUTPUT,
        feedback="Invalid output.",
        unsupported_claims=[],
        missing_evidence=[],
        cited_chunk_ids=[],
    )

    with pytest.raises(SupervisorOutputError, match="reserved"):
        verify_answer(
            FakeChatModel(response),
            "Question",
            make_retrieval(),
            make_generator(),
        )


def test_insufficient_context_is_allowed_for_supported_generator() -> None:
    result = SupervisorResult(
        verified=False,
        verdict=VerificationVerdict.INSUFFICIENT_CONTEXT,
        feedback="The retrieved evidence is insufficient.",
        unsupported_claims=[],
        missing_evidence=[],
        cited_chunk_ids=["refund.md:na:0"],
    )

    validated = _validate_supervisor_output(
        result,
        make_retrieval(),
        make_generator(supported=True, insufficient_context=False),
    )

    assert validated.verdict is VerificationVerdict.INSUFFICIENT_CONTEXT


def test_needs_revision_can_have_only_meaningful_feedback() -> None:
    result = SupervisorResult(
        verified=False,
        verdict=VerificationVerdict.NEEDS_REVISION,
        feedback="The answer needs revision.",
        unsupported_claims=[],
        missing_evidence=[],
        cited_chunk_ids=[],
    )

    validated = _validate_supervisor_output(
        result,
        make_retrieval(),
        make_generator(),
    )

    assert validated.verdict is VerificationVerdict.NEEDS_REVISION


def test_domain_validation_does_not_mutate_original_result() -> None:
    result = SupervisorResult(
        verified=False,
        verdict=VerificationVerdict.NEEDS_REVISION,
        feedback="  Feedback  ",
        unsupported_claims=["  Claim  "],
        missing_evidence=[],
        cited_chunk_ids=["  refund.md:na:0  "],
    )
    original = result.model_copy(deep=True)

    _validate_supervisor_output(result, make_retrieval(), make_generator())

    assert result == original


def test_domain_validation_does_not_mutate_retrieval_result() -> None:
    retrieval = make_retrieval()
    original = copy.deepcopy(retrieval)

    _validate_supervisor_output(
        make_supervisor_result(),
        retrieval,
        make_generator(),
    )

    assert retrieval == original


def test_domain_validation_does_not_mutate_generator_result() -> None:
    generator = make_generator()
    original = generator.model_copy(deep=True)

    _validate_supervisor_output(
        make_supervisor_result(),
        make_retrieval(),
        generator,
    )

    assert generator == original