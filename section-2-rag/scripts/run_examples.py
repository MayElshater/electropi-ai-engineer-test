"""Run interactive examples through the real LangGraph RAG pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.graph import RAGGraphResult, build_rag_graph, run_rag_graph
from app.loader import load_documents
from app.splitter import split_documents
from app.vector_store import (
    create_embeddings,
    create_vector_store,
    index_documents,
)


OUTPUT_PATH = PROJECT_ROOT / "examples" / "example_outputs.md"
SEPARATOR = "=" * 50
RULE = "-" * 50


def build_pipeline() -> Any:
    """Build the existing persisted RAG pipeline from project settings."""
    settings = get_settings()
    if not settings.google_api_key.strip():
        raise ValueError("GOOGLE_API_KEY is required to run the demo")

    embeddings = create_embeddings(settings.embedding_model)
    vector_store = create_vector_store(
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
        embeddings=embeddings,
    )
    documents = load_documents(settings.documents_dir)
    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    index_documents(vector_store, chunks)

    generator_model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
    )
    supervisor_model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
    )
    return build_rag_graph(
        vector_store=vector_store,
        generator_model=generator_model,
        supervisor_model=supervisor_model,
        retrieval_k=settings.retrieval_k,
        maximum_distance=settings.retrieval_maximum_distance,
    )


def ask_example_count() -> int:
    """Prompt until the user enters a positive number of examples."""
    while True:
        raw_value = input(
            "How many example questions would you like to run?\n> "
        ).strip()
        try:
            count = int(raw_value)
        except ValueError:
            print("Please enter a positive integer.")
            continue
        if count < 1:
            print("Please enter a positive integer.")
            continue
        return count


def ask_question(example_number: int) -> str:
    """Prompt until the user enters a non-blank question."""
    print(f"\nQuestion #{example_number}\n")
    while True:
        question = input("Enter your question:\n> ").strip()
        if question:
            return question
        print("Question must not be blank.")


def run_example(graph: Any, question: str) -> RAGGraphResult:
    """Execute one question through the existing compiled graph."""
    return run_rag_graph(graph, question)


def _citation_lines(chunk_ids: list[str]) -> str:
    """Format citations consistently for terminal and Markdown output."""
    if not chunk_ids:
        return "- None"
    return "\n".join(f"- {chunk_id}" for chunk_id in chunk_ids)


def format_terminal_result(
    example_number: int,
    question: str,
    result: RAGGraphResult,
) -> str:
    """Format one successful graph result for terminal display."""
    return "\n".join(
        [
            SEPARATOR,
            f"Example {example_number}",
            SEPARATOR,
            "",
            "Question",
            "",
            question,
            "",
            RULE,
            "",
            "Answer",
            "",
            result.answer,
            "",
            RULE,
            "",
            "Status",
            "",
            result.status.value,
            "",
            "Supervisor Verdict",
            "",
            result.verdict.value,
            "",
            "Verified",
            "",
            str(result.verified),
            "",
            "Attempts",
            "",
            str(result.attempts),
            "",
            "Citations",
            "",
            _citation_lines(result.chunk_ids),
            "",
            RULE,
        ]
    )


def format_markdown_result(
    example_number: int,
    question: str,
    result: RAGGraphResult,
) -> str:
    """Format one successful graph result as Markdown."""
    return "\n".join(
        [
            f"## Example {example_number}",
            "",
            "### Question",
            "",
            question,
            "",
            "### Answer",
            "",
            result.answer,
            "",
            "### Graph Status",
            "",
            result.status.value,
            "",
            "### Supervisor Verdict",
            "",
            result.verdict.value,
            "",
            "### Verified",
            "",
            str(result.verified),
            "",
            "### Attempts",
            "",
            str(result.attempts),
            "",
            "### Citations",
            "",
            _citation_lines(result.chunk_ids),
            "",
            "---",
            "",
        ]
    )


def format_error_result(
    example_number: int,
    question: str,
    error: Exception,
) -> str:
    """Format a failed example for the generated Markdown report."""
    return "\n".join(
        [
            f"## Example {example_number}",
            "",
            "### Question",
            "",
            question,
            "",
            "### Status",
            "",
            "ERROR",
            "",
            "### Error",
            "",
            str(error),
            "",
            "---",
            "",
        ]
    )


def write_header(output_path: Path = OUTPUT_PATH) -> None:
    """Create or overwrite the report with its generated header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat()
    header = "\n".join(
        [
            "# RAG Pipeline Example Outputs",
            "",
            "Generated automatically by scripts/run_examples.py.",
            "",
            "Do not edit manually.",
            "",
            "Generation timestamp:",
            timestamp,
            "",
        ]
    )
    output_path.write_text(header, encoding="utf-8")


def save_result(markdown: str, output_path: Path = OUTPUT_PATH) -> None:
    """Save a current-run result after the report has been overwritten."""
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(markdown)


def main() -> int:
    """Run the interactive acceptance demonstration."""
    print("LangGraph RAG Demo\n")
    try:
        graph = build_pipeline()
    except Exception as error:
        print(f"Unable to initialize the RAG pipeline: {error}")
        return 1

    write_header()
    example_count = ask_example_count()
    for example_number in range(1, example_count + 1):
        question = ask_question(example_number)
        try:
            result = run_example(graph, question)
        except Exception as error:
            print(f"\nExample {example_number} failed: {error}")
            save_result(format_error_result(example_number, question, error))
            continue

        print(f"\n{format_terminal_result(example_number, question, result)}")
        print("\nSaved to example_outputs.md")
        save_result(format_markdown_result(example_number, question, result))

    print(
        f"\n{SEPARATOR}\n\n"
        "Demo completed successfully.\n\n"
        "Results saved to:\n\n"
        "examples/example_outputs.md\n\n"
        f"{SEPARATOR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
