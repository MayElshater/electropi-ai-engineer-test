"""LangGraph orchestration for the grounded RAG pipeline."""

from enum import Enum
from typing import Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from app.generator import INSUFFICIENT_CONTEXT_ANSWER, GeneratorResult, generate_answer
from app.retriever import RetrievalResult, retrieve_documents
from app.supervisor import SupervisorResult, VerificationVerdict, verify_answer


INVALID_OUTPUT_ANSWER = "The generated answer could not be verified."


class GraphStatus(str, Enum):
    """Terminal outcomes of the RAG graph."""

    COMPLETED = "COMPLETED"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    MAX_RETRIES_EXCEEDED = "MAX_RETRIES_EXCEEDED"
    INVALID_OUTPUT = "INVALID_OUTPUT"


class RAGGraphResult(BaseModel):
    """Structured public result returned by the RAG graph."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    status: GraphStatus
    verified: bool
    verdict: VerificationVerdict
    chunk_ids: list[str] = Field(default_factory=list)
    attempts: int
    feedback: str = ""


class RAGGraphState(TypedDict, total=False):
    """Typed internal state shared by graph nodes."""

    question: str
    retrieval_result: RetrievalResult
    generator_result: GeneratorResult
    supervisor_result: SupervisorResult
    revision_feedback: str
    retry_count: int
    max_retries: int
    final_result: RAGGraphResult


def build_revision_feedback(result: SupervisorResult) -> str:
    """Build deterministic, non-evidentiary revision guidance."""
    sections = [f"Supervisor feedback:\n{result.feedback}"]
    if result.unsupported_claims:
        items = "\n".join(f"- {claim}" for claim in result.unsupported_claims)
        sections.append(f"Unsupported claims:\n{items}")
    if result.missing_evidence:
        items = "\n".join(f"- {item}" for item in result.missing_evidence)
        sections.append(f"Missing evidence:\n{items}")
    sections.append(
        "Revision instruction:\nRevise the answer using only the retrieved "
        "context. Remove unsupported claims. If the context cannot support "
        "the answer, return the existing insufficient-context response."
    )
    return "\n\n".join(sections)


def route_after_supervision(state: RAGGraphState) -> str:
    """Select the next node deterministically from the Supervisor verdict."""
    result = state["supervisor_result"]
    if result.verdict is VerificationVerdict.VERIFIED:
        return "finalize_verified"
    if result.verdict is VerificationVerdict.INSUFFICIENT_CONTEXT:
        return "finalize_insufficient_context"
    if result.verdict is VerificationVerdict.INVALID_OUTPUT:
        return "finalize_invalid_output"
    if state["retry_count"] < state["max_retries"]:
        return "prepare_retry"
    return "finalize_max_retries"


def _attempts(state: RAGGraphState) -> int:
    """Return the total number of Generator calls represented by state."""
    return state["retry_count"] + 1


def build_rag_graph(
    *,
    vector_store: Any,
    generator_model: BaseChatModel,
    supervisor_model: BaseChatModel,
    retrieval_k: int = 4,
    maximum_distance: float | None = None,
    max_retries: int = 1,
) -> Any:
    """Build a dependency-injected synchronous RAG workflow."""
    if max_retries < 0:
        raise ValueError("max_retries must be at least 0")

    def retrieve_node(state: RAGGraphState) -> dict[str, RetrievalResult]:
        return {
            "retrieval_result": retrieve_documents(
                vector_store,
                state["question"],
                k=retrieval_k,
                maximum_distance=maximum_distance,
            )
        }

    def generate_node(state: RAGGraphState) -> dict[str, GeneratorResult]:
        result = generate_answer(
            generator_model,
            state["question"],
            state["retrieval_result"],
            revision_feedback=state.get("revision_feedback"),
        )
        return {"generator_result": result}

    def supervise_node(state: RAGGraphState) -> dict[str, SupervisorResult]:
        return {
            "supervisor_result": verify_answer(
                supervisor_model,
                state["question"],
                state["retrieval_result"],
                state["generator_result"],
            )
        }

    def prepare_retry_node(state: RAGGraphState) -> dict[str, int | str]:
        return {
            "retry_count": state["retry_count"] + 1,
            "revision_feedback": build_revision_feedback(
                state["supervisor_result"]
            ),
        }

    def finalize_verified_node(state: RAGGraphState) -> dict[str, RAGGraphResult]:
        generated = state["generator_result"]
        supervised = state["supervisor_result"]
        return {
            "final_result": RAGGraphResult(
                answer=generated.answer,
                status=GraphStatus.COMPLETED,
                verified=True,
                verdict=VerificationVerdict.VERIFIED,
                chunk_ids=list(generated.chunk_ids),
                attempts=_attempts(state),
                feedback=supervised.feedback,
            )
        }

    def finalize_insufficient_node(
        state: RAGGraphState,
    ) -> dict[str, RAGGraphResult]:
        return {
            "final_result": RAGGraphResult(
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                status=GraphStatus.INSUFFICIENT_CONTEXT,
                verified=False,
                verdict=VerificationVerdict.INSUFFICIENT_CONTEXT,
                chunk_ids=[],
                attempts=_attempts(state),
                feedback=state["supervisor_result"].feedback,
            )
        }

    def finalize_invalid_node(state: RAGGraphState) -> dict[str, RAGGraphResult]:
        return {
            "final_result": RAGGraphResult(
                answer=INVALID_OUTPUT_ANSWER,
                status=GraphStatus.INVALID_OUTPUT,
                verified=False,
                verdict=VerificationVerdict.INVALID_OUTPUT,
                chunk_ids=[],
                attempts=_attempts(state),
                feedback=state["supervisor_result"].feedback,
            )
        }

    def finalize_max_retries_node(
        state: RAGGraphState,
    ) -> dict[str, RAGGraphResult]:
        generated = state["generator_result"]
        supervised = state["supervisor_result"]
        return {
            "final_result": RAGGraphResult(
                answer=generated.answer,
                status=GraphStatus.MAX_RETRIES_EXCEEDED,
                verified=False,
                verdict=VerificationVerdict.NEEDS_REVISION,
                chunk_ids=list(generated.chunk_ids),
                attempts=_attempts(state),
                feedback=supervised.feedback,
            )
        }

    workflow = StateGraph(RAGGraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("supervise", supervise_node)
    workflow.add_node("prepare_retry", prepare_retry_node)
    workflow.add_node("finalize_verified", finalize_verified_node)
    workflow.add_node(
        "finalize_insufficient_context",
        finalize_insufficient_node,
    )
    workflow.add_node("finalize_invalid_output", finalize_invalid_node)
    workflow.add_node("finalize_max_retries", finalize_max_retries_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "supervise")
    workflow.add_conditional_edges(
        "supervise",
        route_after_supervision,
        {
            "finalize_verified": "finalize_verified",
            "finalize_insufficient_context": "finalize_insufficient_context",
            "finalize_invalid_output": "finalize_invalid_output",
            "prepare_retry": "prepare_retry",
            "finalize_max_retries": "finalize_max_retries",
        },
    )
    workflow.add_edge("prepare_retry", "generate")
    for final_node in (
        "finalize_verified",
        "finalize_insufficient_context",
        "finalize_invalid_output",
        "finalize_max_retries",
    ):
        workflow.add_edge(final_node, END)

    graph = workflow.compile()
    graph._rag_max_retries = max_retries
    return graph


def run_rag_graph(
    graph: Any,
    question: str,
    *,
    max_retries: int | None = None,
) -> RAGGraphResult:
    """Validate a question, invoke the graph, and return its typed result."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be blank")
    resolved_retries = (
        getattr(graph, "_rag_max_retries", 1)
        if max_retries is None
        else max_retries
    )
    if resolved_retries < 0:
        raise ValueError("max_retries must be at least 0")
    final_state = graph.invoke(
        {
            "question": normalized_question,
            "retry_count": 0,
            "max_retries": resolved_retries,
            "revision_feedback": "",
        }
    )
    return RAGGraphResult.model_validate(final_state["final_result"])
