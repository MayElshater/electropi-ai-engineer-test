"""FastAPI application for local LLM text generation."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from fastapi import FastAPI, HTTPException, Request, status

from app.model_service import DEFAULT_SYSTEM_PROMPT, ModelService
from app.schemas import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"


def _resolve_model_path(value: str) -> Path:
    """Resolve a model path relative to the project root when necessary."""
    path = Path(value.strip()).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _parse_int(name: str, default: str) -> int:
    """Parse an integer environment variable with a clear error."""
    value = os.getenv(name, default).strip()
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid integer") from exc


def _parse_optional_int(name: str) -> int | None:
    """Parse a blankable integer environment variable."""
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid integer") from exc


def _parse_bool(name: str, default: str) -> bool:
    """Parse a case-insensitive boolean environment variable."""
    value = os.getenv(name, default).strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of: true, 1, yes, on, false, 0, no, off"
    )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load one configured model service for the application lifetime."""
    logger.info("Starting Local LLM Inference API")
    application.state.model_service = ModelService(
        model_path=str(
            _resolve_model_path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
        ),
        n_ctx=_parse_int("MODEL_N_CTX", "4096"),
        n_threads=_parse_optional_int("MODEL_N_THREADS"),
        n_gpu_layers=_parse_int("MODEL_N_GPU_LAYERS", "0"),
        verbose=_parse_bool("MODEL_VERBOSE", "false"),
        system_prompt=os.getenv("MODEL_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
    )
    try:
        yield
    finally:
        logger.info("Shutting down Local LLM Inference API")


app = FastAPI(
    title="Local LLM Inference API",
    description="Local text generation backed by llama.cpp and a GGUF model.",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_model_service(request: Request) -> ModelService:
    """Retrieve the lifespan-managed model service."""
    return cast(ModelService, request.app.state.model_service)


@app.get("/", summary="Service information", description="Return the service name and current API status.")
def service_info() -> dict[str, str]:
    """Return basic service information."""
    return {"service": "Local LLM Inference API", "status": "running"}


@app.get("/health", summary="Health check", description="Confirm that the API process is alive without running inference.")
def health_check() -> dict[str, str]:
    """Return the API liveness status."""
    return {"status": "healthy"}


@app.post("/generate", response_model=GenerateResponse, summary="Generate text", description="Generate a model response from validated sampling parameters.")
def generate_text(request: GenerateRequest, http_request: Request) -> GenerateResponse:
    """Generate text using the application-scoped model service."""
    logger.info("Received generation request")
    service = _get_model_service(http_request)
    try:
        return service.generate(request)
    except RuntimeError:
        logger.exception("Generation request failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model inference failed") from None
