"""Pydantic schemas for local text-generation requests and responses."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GenerateRequest(BaseModel):
    """Validated parameters for a local text-generation request."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        allow_inf_nan=False,
    )

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4_000,
        description="Prompt text to send to the local language model.",
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=2_048,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature; higher values increase randomness.",
    )
    top_p: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Nucleus-sampling probability threshold.",
    )
    stop: list[str] | None = Field(
        default=None,
        max_length=10,
        description="Optional normalized sequences that stop generation.",
    )

    @field_validator("prompt")
    @classmethod
    def normalize_prompt(cls, value: str) -> str:
        """Strip prompt whitespace and reject an empty normalized prompt."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("prompt must not be empty")
        return normalized

    @field_validator("stop")
    @classmethod
    def normalize_stop_sequences(
        cls,
        value: list[str] | None,
    ) -> list[str] | None:
        """Strip stop sequences and reject empty or duplicate values."""
        if value is None:
            return None

        normalized: list[str] = []
        seen: set[str] = set()
        for sequence in value:
            stripped = sequence.strip()
            if not stripped:
                raise ValueError("stop sequences must not be empty")
            if stripped in seen:
                raise ValueError("stop sequences must be unique")
            normalized.append(stripped)
            seen.add(stripped)
        return normalized


class GenerateResponse(BaseModel):
    """Validated output and performance metadata for a generation request."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        allow_inf_nan=False,
    )

    response: str = Field(
        ...,
        min_length=1,
        description="Generated response text.",
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Identifier of the model that generated the response.",
    )
    prompt_tokens: int = Field(
        ...,
        ge=0,
        description="Number of input prompt tokens.",
    )
    completion_tokens: int = Field(
        ...,
        ge=0,
        description="Number of generated completion tokens.",
    )
    total_tokens: int = Field(
        ...,
        ge=0,
        description="Sum of prompt and completion tokens.",
    )
    generation_time_seconds: float = Field(
        ...,
        ge=0.0,
        description="Generation duration in seconds.",
    )
    tokens_per_second: float = Field(
        ...,
        ge=0.0,
        description="Generation throughput in tokens per second.",
    )

    @field_validator("response", "model")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        """Strip required text fields and reject empty normalized values."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_token_total(self) -> Self:
        """Require total tokens to match prompt plus completion tokens."""
        expected_total = self.prompt_tokens + self.completion_tokens
        if self.total_tokens != expected_total:
            raise ValueError(
                "total_tokens must equal prompt_tokens + completion_tokens"
            )
        return self
