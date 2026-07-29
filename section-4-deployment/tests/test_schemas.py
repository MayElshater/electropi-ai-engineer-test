"""Unit tests for the generation request and response schemas."""

import math

import pytest
from pydantic import ValidationError

from app.schemas import GenerateRequest, GenerateResponse


def valid_response_data() -> dict[str, object]:
    """Return a valid response payload for focused mutation in tests."""
    return {
        "response": "Generated text",
        "model": "Qwen2.5-1.5B-Instruct",
        "prompt_tokens": 8,
        "completion_tokens": 12,
        "total_tokens": 20,
        "generation_time_seconds": 0.5,
        "tokens_per_second": 24.0,
    }


def test_request_with_prompt_uses_expected_defaults() -> None:
    request = GenerateRequest(prompt="Hello")

    assert request.prompt == "Hello"
    assert request.max_tokens == 256
    assert request.temperature == 0.7
    assert request.top_p == 0.9
    assert request.stop is None


def test_request_accepts_custom_generation_settings() -> None:
    request = GenerateRequest(
        prompt="Hello",
        max_tokens=512,
        temperature=1.2,
        top_p=0.8,
        stop=["END"],
    )

    assert request.max_tokens == 512
    assert request.temperature == 1.2
    assert request.top_p == 0.8
    assert request.stop == ["END"]


def test_request_strips_prompt_whitespace() -> None:
    assert GenerateRequest(prompt="  Hello  ").prompt == "Hello"


def test_request_rejects_missing_prompt() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest()


def test_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="")


def test_request_rejects_whitespace_only_prompt() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt=" \t\n ")


def test_request_rejects_prompt_longer_than_limit() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="x" * 4_001)


@pytest.mark.parametrize("max_tokens", [0, 2_049])
def test_request_rejects_max_tokens_outside_limits(max_tokens: int) -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Hello", max_tokens=max_tokens)


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_request_rejects_temperature_outside_limits(temperature: float) -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Hello", temperature=temperature)


@pytest.mark.parametrize("temperature", [math.nan, math.inf, -math.inf])
def test_request_rejects_non_finite_temperature(temperature: float) -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Hello", temperature=temperature)


@pytest.mark.parametrize("top_p", [0.0, 1.1])
def test_request_rejects_top_p_outside_limits(top_p: float) -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Hello", top_p=top_p)


def test_request_rejects_more_than_ten_stop_sequences() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Hello", stop=[str(index) for index in range(11)])


def test_request_rejects_empty_stop_sequence() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Hello", stop=[""])


def test_request_rejects_whitespace_only_stop_sequence() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Hello", stop=[" \t "])


def test_request_normalizes_stop_sequences() -> None:
    request = GenerateRequest(prompt="Hello", stop=[" END ", "\nDONE\t"])

    assert request.stop == ["END", "DONE"]


def test_request_rejects_duplicate_normalized_stop_sequences() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Hello", stop=["END", " END "])


def test_request_rejects_unexpected_fields() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Hello", unknown=True)


def test_valid_response_is_accepted() -> None:
    response = GenerateResponse(**valid_response_data())

    assert response.response == "Generated text"
    assert response.total_tokens == 20


def test_response_strips_response_and_model_whitespace() -> None:
    data = valid_response_data()
    data["response"] = "  Generated text\n"
    data["model"] = "\tQwen model "

    response = GenerateResponse(**data)

    assert response.response == "Generated text"
    assert response.model == "Qwen model"


@pytest.mark.parametrize("value", ["", " \t\n "])
def test_response_rejects_empty_response(value: str) -> None:
    data = valid_response_data()
    data["response"] = value

    with pytest.raises(ValidationError):
        GenerateResponse(**data)


@pytest.mark.parametrize("value", ["", " \t\n "])
def test_response_rejects_empty_model(value: str) -> None:
    data = valid_response_data()
    data["model"] = value

    with pytest.raises(ValidationError):
        GenerateResponse(**data)


@pytest.mark.parametrize(
    "field",
    ["prompt_tokens", "completion_tokens", "total_tokens"],
)
def test_response_rejects_negative_token_counts(field: str) -> None:
    data = valid_response_data()
    data[field] = -1

    with pytest.raises(ValidationError):
        GenerateResponse(**data)


def test_response_rejects_inconsistent_total_tokens() -> None:
    data = valid_response_data()
    data["total_tokens"] = 21

    with pytest.raises(ValidationError):
        GenerateResponse(**data)


def test_response_rejects_negative_generation_time() -> None:
    data = valid_response_data()
    data["generation_time_seconds"] = -0.1

    with pytest.raises(ValidationError):
        GenerateResponse(**data)


def test_response_rejects_negative_throughput() -> None:
    data = valid_response_data()
    data["tokens_per_second"] = -0.1

    with pytest.raises(ValidationError):
        GenerateResponse(**data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation_time_seconds", math.nan),
        ("generation_time_seconds", math.inf),
        ("generation_time_seconds", -math.inf),
        ("tokens_per_second", math.nan),
        ("tokens_per_second", math.inf),
        ("tokens_per_second", -math.inf),
    ],
)
def test_response_rejects_non_finite_metrics(field: str, value: float) -> None:
    data = valid_response_data()
    data[field] = value

    with pytest.raises(ValidationError):
        GenerateResponse(**data)


def test_response_rejects_unexpected_fields() -> None:
    data = valid_response_data()
    data["unknown"] = True

    with pytest.raises(ValidationError):
        GenerateResponse(**data)
