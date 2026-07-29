"""Framework-independent service for local llama.cpp text generation."""

import logging
import time
from pathlib import Path
from typing import Any

import llama_cpp

from app.schemas import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a helpful, accurate, and concise AI assistant."


class ModelService:
    """Load one GGUF model and provide validated text generation."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        """Validate and load a GGUF model using llama.cpp."""
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

        normalized_system_prompt = system_prompt.strip()
        if not normalized_system_prompt:
            raise ValueError("system_prompt must not be empty")

        self.model_name = path.stem
        self.system_prompt = normalized_system_prompt
        logger.info("Loading model from %s", path)
        self._model = llama_cpp.Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        logger.info("Model loaded successfully")

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text and return validated output and timing metadata."""
        started_at = time.perf_counter()
        try:
            result = self._model.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": request.prompt},
                ],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                stream=False,
            )
        except Exception as exc:
            logger.exception("Model inference failed")
            raise RuntimeError("Model inference failed") from exc

        duration = time.perf_counter() - started_at
        response = self._build_response(result, duration)
        logger.info("Inference completed successfully in %.6f seconds", duration)
        return response

    def _build_response(
        self,
        result: dict[str, Any],
        duration: float,
    ) -> GenerateResponse:
        """Validate a llama.cpp chat result and build the public response."""
        try:
            content = result["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("generated content is empty")

            usage = result["usage"]
            prompt_tokens = self._get_token_count(usage, "prompt_tokens")
            completion_tokens = self._get_token_count(
                usage, "completion_tokens"
            )
            total_tokens = self._get_token_count(usage, "total_tokens")
            throughput = completion_tokens / duration if duration > 0.0 else 0.0

            return GenerateResponse(
                response=content.strip(),
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                generation_time_seconds=duration,
                tokens_per_second=throughput,
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("Invalid model response") from exc

    @staticmethod
    def _get_token_count(usage: dict[str, Any], field: str) -> int:
        """Extract a non-negative integer token count from usage metadata."""
        value = usage[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        return value
