"""Unit tests for the framework-independent model service."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.model_service import DEFAULT_SYSTEM_PROMPT, ModelService
from app.schemas import GenerateRequest


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    path.touch()
    return path


@pytest.fixture
def llama_class() -> Iterator[MagicMock]:
    with patch("app.model_service.llama_cpp.Llama") as mocked:
        yield mocked


@pytest.fixture
def chat_result() -> dict[str, object]:
    return {
        "choices": [{"message": {"content": " Generated answer "}}],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15,
        },
    }


def create_service(model_file: Path, llama_class: MagicMock) -> ModelService:
    return ModelService(str(model_file))


def test_constructor_loads_model_once(model_file: Path, llama_class: MagicMock) -> None:
    create_service(model_file, llama_class)
    llama_class.assert_called_once()


def test_missing_model_raises_file_not_found(tmp_path: Path, llama_class: MagicMock) -> None:
    with pytest.raises(FileNotFoundError):
        ModelService(str(tmp_path / "missing.gguf"))
    llama_class.assert_not_called()


def test_clean_model_name_uses_path_stem(model_file: Path, llama_class: MagicMock) -> None:
    service = create_service(model_file, llama_class)
    assert service.model_name == "qwen2.5-1.5b-instruct-q4_k_m"


def test_empty_system_prompt_is_rejected(model_file: Path, llama_class: MagicMock) -> None:
    with pytest.raises(ValueError, match="system_prompt must not be empty"):
        ModelService(str(model_file), system_prompt="  \t ")
    llama_class.assert_not_called()


def test_generate_calls_chat_completion_with_messages_and_parameters(
    model_file: Path, llama_class: MagicMock, chat_result: dict[str, object]
) -> None:
    model = llama_class.return_value
    model.create_chat_completion.return_value = chat_result
    service = create_service(model_file, llama_class)
    request = GenerateRequest(
        prompt="Explain quantization",
        max_tokens=128,
        temperature=0.4,
        top_p=0.85,
        stop=["END"],
    )

    with patch("app.model_service.time.perf_counter", side_effect=[1.0, 2.0]):
        service.generate(request)

    model.create_chat_completion.assert_called_once_with(
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "Explain quantization"},
        ],
        max_tokens=128,
        temperature=0.4,
        top_p=0.85,
        stop=["END"],
        stream=False,
    )


def test_custom_system_prompt_is_stripped_and_used(
    model_file: Path, llama_class: MagicMock, chat_result: dict[str, object]
) -> None:
    llama_class.return_value.create_chat_completion.return_value = chat_result
    service = ModelService(str(model_file), system_prompt="  Be precise.  ")

    with patch("app.model_service.time.perf_counter", side_effect=[1.0, 2.0]):
        service.generate(GenerateRequest(prompt="Hello"))

    messages = llama_class.return_value.create_chat_completion.call_args.kwargs["messages"]
    assert service.system_prompt == "Be precise."
    assert messages[0] == {"role": "system", "content": "Be precise."}


def test_response_content_is_extracted_and_stripped(
    model_file: Path, llama_class: MagicMock, chat_result: dict[str, object]
) -> None:
    llama_class.return_value.create_chat_completion.return_value = chat_result
    service = create_service(model_file, llama_class)

    with patch("app.model_service.time.perf_counter", side_effect=[1.0, 2.0]):
        response = service.generate(GenerateRequest(prompt="Hello"))

    assert response.response == "Generated answer"


def test_response_uses_clean_model_identifier(
    model_file: Path, llama_class: MagicMock, chat_result: dict[str, object]
) -> None:
    llama_class.return_value.create_chat_completion.return_value = chat_result
    service = create_service(model_file, llama_class)

    with patch("app.model_service.time.perf_counter", side_effect=[1.0, 2.0]):
        response = service.generate(GenerateRequest(prompt="Hello"))

    assert response.model == "qwen2.5-1.5b-instruct-q4_k_m"
    assert str(model_file.parent) not in response.model


def test_usage_values_are_extracted(
    model_file: Path, llama_class: MagicMock, chat_result: dict[str, object]
) -> None:
    llama_class.return_value.create_chat_completion.return_value = chat_result
    service = create_service(model_file, llama_class)

    with patch("app.model_service.time.perf_counter", side_effect=[1.0, 3.0]):
        response = service.generate(GenerateRequest(prompt="Hello"))

    assert response.prompt_tokens == 5
    assert response.completion_tokens == 10
    assert response.total_tokens == 15
    assert response.generation_time_seconds == 2.0
    assert response.tokens_per_second == 5.0


def test_malformed_choices_raise_invalid_model_response(
    model_file: Path, llama_class: MagicMock
) -> None:
    llama_class.return_value.create_chat_completion.return_value = {"choices": []}
    service = create_service(model_file, llama_class)

    with (
        patch("app.model_service.time.perf_counter", side_effect=[1.0, 2.0]),
        pytest.raises(RuntimeError, match="^Invalid model response$") as error,
    ):
        service.generate(GenerateRequest(prompt="Hello"))

    assert isinstance(error.value.__cause__, IndexError)


def test_missing_usage_raises_invalid_model_response(
    model_file: Path, llama_class: MagicMock
) -> None:
    llama_class.return_value.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Answer"}}]
    }
    service = create_service(model_file, llama_class)

    with (
        patch("app.model_service.time.perf_counter", side_effect=[1.0, 2.0]),
        pytest.raises(RuntimeError, match="^Invalid model response$") as error,
    ):
        service.generate(GenerateRequest(prompt="Hello"))

    assert isinstance(error.value.__cause__, KeyError)


def test_empty_content_raises_invalid_model_response(
    model_file: Path, llama_class: MagicMock, chat_result: dict[str, object]
) -> None:
    chat_result["choices"] = [{"message": {"content": "   "}}]
    llama_class.return_value.create_chat_completion.return_value = chat_result
    service = create_service(model_file, llama_class)

    with (
        patch("app.model_service.time.perf_counter", side_effect=[1.0, 2.0]),
        pytest.raises(RuntimeError, match="^Invalid model response$"),
    ):
        service.generate(GenerateRequest(prompt="Hello"))


def test_llama_exception_is_wrapped_as_inference_failure(
    model_file: Path, llama_class: MagicMock
) -> None:
    original = ValueError("backend details")
    llama_class.return_value.create_chat_completion.side_effect = original
    service = create_service(model_file, llama_class)

    with (
        patch("app.model_service.time.perf_counter", return_value=1.0),
        pytest.raises(RuntimeError, match="^Model inference failed$") as error,
    ):
        service.generate(GenerateRequest(prompt="Hello"))

    assert error.value.__cause__ is original


def test_zero_duration_produces_zero_throughput(
    model_file: Path, llama_class: MagicMock, chat_result: dict[str, object]
) -> None:
    llama_class.return_value.create_chat_completion.return_value = chat_result
    service = create_service(model_file, llama_class)
    with patch("app.model_service.time.perf_counter", side_effect=[1.0, 1.0]):
        response = service.generate(GenerateRequest(prompt="Hello"))
    assert response.tokens_per_second == 0.0


def test_logging_covers_loading_success_and_failure(
    model_file: Path, llama_class: MagicMock, chat_result: dict[str, object]
) -> None:
    with patch("app.model_service.logger.info") as info:
        llama_class.return_value.create_chat_completion.return_value = chat_result
        service = create_service(model_file, llama_class)
        with patch("app.model_service.time.perf_counter", side_effect=[1.0, 2.0]):
            service.generate(GenerateRequest(prompt="Hello"))
    assert info.call_count == 3
