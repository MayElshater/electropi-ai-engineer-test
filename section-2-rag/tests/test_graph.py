"""Offline tests for LangGraph RAG orchestration."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from langchain_core.documents import Document

import app.graph as graph_module
from app.generator import INSUFFICIENT_CONTEXT_ANSWER, GeneratorResult
from app.graph import (
    INVALID_OUTPUT_ANSWER,
    GraphStatus,
    RAGGraphResult,
    build_rag_graph,
    build_revision_feedback,
    route_after_supervision,
    run_rag_graph,
)
from app.retriever import RetrievedChunk, RetrievalResult, RetrievalStatus
from app.supervisor import SupervisorResult, VerificationVerdict


class DummyModel:
    """Opaque dependency used when orchestration functions are monkeypatched."""


def make_retrieval() -> RetrievalResult:
    chunk = RetrievedChunk(
        document=Document(
            page_content="Refunds require an invoice.",
            metadata={"chunk_id": "policy.md:na:0", "source": "policy.md"},
        ),
        distance=0.1,
    )
    return RetrievalResult(
        query="What is required?",
        chunks=[chunk],
        context="Refunds require an invoice.",
        searched_count=1,
        status=RetrievalStatus.RELEVANT_CONTEXT,
    )


def generated(answer: str = "An invoice is required.") -> GeneratorResult:
    return GeneratorResult(
        answer=answer,
        supported=True,
        chunk_ids=["policy.md:na:0"],
        insufficient_context=False,
    )


def supervised(
    verdict: VerificationVerdict,
    *,
    feedback: str = "Review complete.",
) -> SupervisorResult:
    return SupervisorResult(
        verified=verdict is VerificationVerdict.VERIFIED,
        verdict=verdict,
        feedback=feedback,
        unsupported_claims=(
            ["Unsupported deadline"]
            if verdict is VerificationVerdict.NEEDS_REVISION
            else []
        ),
        missing_evidence=(
            ["Deadline evidence"]
            if verdict is VerificationVerdict.NEEDS_REVISION
            else []
        ),
        cited_chunk_ids=["policy.md:na:0"],
    )


def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    generator_results: list[GeneratorResult],
    supervisor_results: list[SupervisorResult],
) -> tuple[RetrievalResult, dict[str, list[Any]]]:
    retrieval = make_retrieval()
    calls: dict[str, list[Any]] = {"retrieve": [], "generate": [], "supervise": []}

    def fake_retrieve(*args: Any, **kwargs: Any) -> RetrievalResult:
        calls["retrieve"].append((args, kwargs))
        return retrieval

    def fake_generate(*args: Any, **kwargs: Any) -> GeneratorResult:
        calls["generate"].append((args, kwargs))
        return generator_results[len(calls["generate"]) - 1]

    def fake_supervise(*args: Any, **kwargs: Any) -> SupervisorResult:
        calls["supervise"].append((args, kwargs))
        return supervisor_results[len(calls["supervise"]) - 1]

    monkeypatch.setattr(graph_module, "retrieve_documents", fake_retrieve)
    monkeypatch.setattr(graph_module, "generate_answer", fake_generate)
    monkeypatch.setattr(graph_module, "verify_answer", fake_supervise)
    return retrieval, calls


def make_graph(max_retries: int = 1) -> Any:
    return build_rag_graph(
        vector_store=object(),
        generator_model=DummyModel(),
        supervisor_model=DummyModel(),
        max_retries=max_retries,
    )


def test_verified_answer_completes_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = install_fakes(
        monkeypatch,
        [generated()],
        [supervised(VerificationVerdict.VERIFIED)],
    )

    result = run_rag_graph(make_graph(), "What is required?")

    assert isinstance(result, RAGGraphResult)
    assert result.status is GraphStatus.COMPLETED
    assert result.verified is True
    assert result.attempts == 1
    assert [len(calls[name]) for name in calls] == [1, 1, 1]


def test_revision_then_verified_reuses_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    retrieval, calls = install_fakes(
        monkeypatch,
        [generated("First"), generated("Revised")],
        [
            supervised(VerificationVerdict.NEEDS_REVISION, feedback="Remove claim."),
            supervised(VerificationVerdict.VERIFIED),
        ],
    )

    result = run_rag_graph(make_graph(), "  What is required?  ")

    assert result.status is GraphStatus.COMPLETED
    assert result.answer == "Revised"
    assert result.attempts == 2
    assert len(calls["retrieve"]) == 1
    assert len(calls["generate"]) == len(calls["supervise"]) == 2
    first_args, first_kwargs = calls["generate"][0]
    second_args, second_kwargs = calls["generate"][1]
    assert first_args[1] == second_args[1] == "What is required?"
    assert first_args[2] is second_args[2] is retrieval
    assert first_kwargs["revision_feedback"] == ""
    assert "Remove claim." in second_kwargs["revision_feedback"]
    assert "Unsupported deadline" in second_kwargs["revision_feedback"]
    assert "Deadline evidence" in second_kwargs["revision_feedback"]


def test_retries_are_bounded_and_latest_result_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, calls = install_fakes(
        monkeypatch,
        [generated("First"), generated("Latest unverified")],
        [
            supervised(VerificationVerdict.NEEDS_REVISION, feedback="First review"),
            supervised(VerificationVerdict.NEEDS_REVISION, feedback="Latest review"),
        ],
    )

    result = run_rag_graph(make_graph(max_retries=1), "Question")

    assert result.status is GraphStatus.MAX_RETRIES_EXCEEDED
    assert result.verdict is VerificationVerdict.NEEDS_REVISION
    assert result.verified is False
    assert result.answer == "Latest unverified"
    assert result.feedback == "Latest review"
    assert result.attempts == 2
    assert len(calls["generate"]) == 2


def test_zero_retries_disables_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = install_fakes(
        monkeypatch,
        [generated()],
        [supervised(VerificationVerdict.NEEDS_REVISION)],
    )

    result = run_rag_graph(make_graph(max_retries=0), "Question")

    assert result.status is GraphStatus.MAX_RETRIES_EXCEEDED
    assert result.attempts == 1
    assert len(calls["generate"]) == 1


@pytest.mark.parametrize(
    ("verdict", "status", "answer"),
    [
        (
            VerificationVerdict.INSUFFICIENT_CONTEXT,
            GraphStatus.INSUFFICIENT_CONTEXT,
            INSUFFICIENT_CONTEXT_ANSWER,
        ),
        (
            VerificationVerdict.INVALID_OUTPUT,
            GraphStatus.INVALID_OUTPUT,
            INVALID_OUTPUT_ANSWER,
        ),
    ],
)
def test_terminal_verdicts_do_not_retry(
    monkeypatch: pytest.MonkeyPatch,
    verdict: VerificationVerdict,
    status: GraphStatus,
    answer: str,
) -> None:
    _, calls = install_fakes(monkeypatch, [generated()], [supervised(verdict)])

    result = run_rag_graph(make_graph(), "Question")

    assert result.status is status
    assert result.answer == answer
    assert result.verified is False
    assert result.attempts == 1
    assert len(calls["generate"]) == 1


def test_latest_generator_citations_are_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    revised = generated("Latest")
    revised.chunk_ids = ["policy.md:na:0"]
    install_fakes(
        monkeypatch,
        [generated("First"), revised],
        [
            supervised(VerificationVerdict.NEEDS_REVISION),
            supervised(VerificationVerdict.VERIFIED),
        ],
    )

    result = run_rag_graph(make_graph(), "Question")

    assert result.answer == "Latest"
    assert result.chunk_ids == revised.chunk_ids


@pytest.mark.parametrize("value", [-1, -5])
def test_negative_factory_max_retries_is_rejected(value: int) -> None:
    with pytest.raises(ValueError, match="max_retries"):
        make_graph(max_retries=value)


def test_negative_invocation_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        run_rag_graph(object(), "Question", max_retries=-1)


@pytest.mark.parametrize("question", ["", " ", "\t\n"])
def test_blank_question_is_rejected_before_execution(question: str) -> None:
    class NeverGraph:
        def invoke(self, state: Any) -> Any:
            raise AssertionError("graph must not run")

    with pytest.raises(ValueError, match="question"):
        run_rag_graph(NeverGraph(), question)


def test_revision_feedback_builder_contains_all_review_details() -> None:
    feedback = build_revision_feedback(
        supervised(VerificationVerdict.NEEDS_REVISION, feedback="Revise deadline.")
    )

    assert "Revise deadline." in feedback
    assert "Unsupported deadline" in feedback
    assert "Deadline evidence" in feedback
    assert "only the retrieved context" in feedback


@pytest.mark.parametrize(
    ("verdict", "retry_count", "max_retries", "expected"),
    [
        (VerificationVerdict.VERIFIED, 0, 1, "finalize_verified"),
        (
            VerificationVerdict.INSUFFICIENT_CONTEXT,
            0,
            1,
            "finalize_insufficient_context",
        ),
        (
            VerificationVerdict.INVALID_OUTPUT,
            0,
            1,
            "finalize_invalid_output",
        ),
        (VerificationVerdict.NEEDS_REVISION, 0, 1, "prepare_retry"),
        (VerificationVerdict.NEEDS_REVISION, 1, 1, "finalize_max_retries"),
    ],
)
def test_router_is_deterministic(
    verdict: VerificationVerdict,
    retry_count: int,
    max_retries: int,
    expected: str,
) -> None:
    state = {
        "supervisor_result": supervised(verdict),
        "retry_count": retry_count,
        "max_retries": max_retries,
    }

    assert route_after_supervision(state) == expected
    assert route_after_supervision(state) == expected


@pytest.mark.parametrize("stage", ["retrieve", "generate", "supervise"])
def test_provider_exceptions_propagate_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    error = RuntimeError(f"{stage} failed")
    retrieval = make_retrieval()

    def retrieve(*args: Any, **kwargs: Any) -> RetrievalResult:
        if stage == "retrieve":
            raise error
        return retrieval

    def generate(*args: Any, **kwargs: Any) -> GeneratorResult:
        if stage == "generate":
            raise error
        return generated()

    def supervise(*args: Any, **kwargs: Any) -> SupervisorResult:
        if stage == "supervise":
            raise error
        return supervised(VerificationVerdict.VERIFIED)

    monkeypatch.setattr(graph_module, "retrieve_documents", retrieve)
    monkeypatch.setattr(graph_module, "generate_answer", generate)
    monkeypatch.setattr(graph_module, "verify_answer", supervise)

    with pytest.raises(RuntimeError) as caught:
        run_rag_graph(make_graph(), "Question")

    assert caught.value is error


def test_pipeline_does_not_mutate_typed_results(monkeypatch: pytest.MonkeyPatch) -> None:
    retrieval = make_retrieval()
    generator_result = generated()
    supervisor_result = supervised(VerificationVerdict.VERIFIED)
    originals = tuple(copy.deepcopy(item) for item in (retrieval, generator_result, supervisor_result))

    monkeypatch.setattr(graph_module, "retrieve_documents", lambda *a, **k: retrieval)
    monkeypatch.setattr(graph_module, "generate_answer", lambda *a, **k: generator_result)
    monkeypatch.setattr(graph_module, "verify_answer", lambda *a, **k: supervisor_result)

    run_rag_graph(make_graph(), "Question")

    assert retrieval == originals[0]
    assert generator_result == originals[1]
    assert supervisor_result == originals[2]


def test_existing_generator_guardrail_routes_to_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval = make_retrieval()
    invalid = GeneratorResult(
        answer="Invented",
        supported=True,
        chunk_ids=["unknown"],
        insufficient_context=False,
    )
    monkeypatch.setattr(graph_module, "retrieve_documents", lambda *a, **k: retrieval)
    monkeypatch.setattr(graph_module, "generate_answer", lambda *a, **k: invalid)

    class NeverSupervisor:
        def with_structured_output(self, schema: Any) -> Any:
            raise AssertionError("guardrail must bypass Supervisor LLM")

    graph = build_rag_graph(
        vector_store=object(),
        generator_model=DummyModel(),
        supervisor_model=NeverSupervisor(),
    )

    result = run_rag_graph(graph, "Question")

    assert result.status is GraphStatus.INVALID_OUTPUT
    assert result.attempts == 1
