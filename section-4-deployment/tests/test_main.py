"""API and runtime configuration tests for the FastAPI application."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import main
from app.model_service import DEFAULT_SYSTEM_PROMPT, ModelService
from app.schemas import GenerateRequest, GenerateResponse


@pytest.fixture
def model_service() -> MagicMock:
    return MagicMock(spec=ModelService)


@pytest.fixture
def client(model_service: MagicMock) -> Iterator[TestClient]:
    @asynccontextmanager
    async def test_lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.model_service = model_service
        yield

    with (
        patch.object(main.app.router, "lifespan_context", test_lifespan),
        patch("app.main.ModelService") as model_service_class,
        TestClient(main.app) as test_client,
    ):
        yield test_client
    model_service_class.assert_not_called()


def test_root_returns_expected_response(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "service": "Local LLM Inference API",
        "status": "running",
    }


def test_health_returns_expected_response(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_generate_returns_clean_model_and_calls_service(
    client: TestClient, model_service: MagicMock
) -> None:
    model_service.generate.return_value = GenerateResponse(
        response="Generated answer",
        model="qwen2.5-1.5b-instruct-q4_k_m",
        prompt_tokens=4,
        completion_tokens=6,
        total_tokens=10,
        generation_time_seconds=0.5,
        tokens_per_second=12.0,
    )

    response = client.post("/generate", json={"prompt": "Hello"})

    assert response.status_code == 200
    assert response.json()["model"] == "qwen2.5-1.5b-instruct-q4_k_m"
    model_service.generate.assert_called_once_with(GenerateRequest(prompt="Hello"))


def test_generate_returns_safe_inference_error(
    client: TestClient, model_service: MagicMock
) -> None:
    model_service.generate.side_effect = RuntimeError("internal details")
    response = client.post("/generate", json={"prompt": "Hello"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Model inference failed"}


def test_default_model_path_resolves_from_project_root() -> None:
    expected = (main.PROJECT_ROOT / main.DEFAULT_MODEL_PATH).resolve()
    assert main._resolve_model_path(main.DEFAULT_MODEL_PATH) == expected


def test_absolute_model_path_remains_absolute(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    assert main._resolve_model_path(str(path)) == path


def test_model_n_ctx_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_N_CTX", " 8192 ")
    assert main._parse_int("MODEL_N_CTX", "4096") == 8192


def test_blank_model_n_threads_becomes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_N_THREADS", "  ")
    assert main._parse_optional_int("MODEL_N_THREADS") is None


def test_numeric_model_n_threads_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_N_THREADS", " 8 ")
    assert main._parse_optional_int("MODEL_N_THREADS") == 8


def test_model_n_gpu_layers_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_N_GPU_LAYERS", " 12 ")
    assert main._parse_int("MODEL_N_GPU_LAYERS", "0") == 12


@pytest.mark.parametrize("value", ["true", "1", "yes", "on", " TRUE ", "YeS"])
def test_accepted_true_values_are_parsed(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_VERBOSE", value)
    assert main._parse_bool("MODEL_VERBOSE", "false") is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", " FALSE ", "OfF"])
def test_accepted_false_values_are_parsed(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_VERBOSE", value)
    assert main._parse_bool("MODEL_VERBOSE", "true") is False


def test_invalid_boolean_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_VERBOSE", "sometimes")
    with pytest.raises(ValueError, match="MODEL_VERBOSE"):
        main._parse_bool("MODEL_VERBOSE", "false")


@pytest.mark.parametrize("name", ["MODEL_N_CTX", "MODEL_N_GPU_LAYERS"])
def test_invalid_required_integer_names_variable(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(name, "invalid")
    with pytest.raises(ValueError, match=name):
        main._parse_int(name, "0")


def test_invalid_optional_integer_names_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_N_THREADS", "invalid")
    with pytest.raises(ValueError, match="MODEL_N_THREADS"):
        main._parse_optional_int("MODEL_N_THREADS")


def test_lifespan_passes_environment_configuration_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "configured.gguf"
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_N_CTX", "8192")
    monkeypatch.setenv("MODEL_N_THREADS", "6")
    monkeypatch.setenv("MODEL_N_GPU_LAYERS", "10")
    monkeypatch.setenv("MODEL_VERBOSE", "yes")
    monkeypatch.setenv("MODEL_SYSTEM_PROMPT", "  Custom assistant.  ")
    application = FastAPI()

    async def run_lifespan() -> None:
        with patch("app.main.ModelService") as service_class:
            async with main.lifespan(application):
                assert application.state.model_service is service_class.return_value
            service_class.assert_called_once_with(
                model_path=str(model_path),
                n_ctx=8192,
                n_threads=6,
                n_gpu_layers=10,
                verbose=True,
                system_prompt="  Custom assistant.  ",
            )

    asyncio.run(run_lifespan())


def test_lifespan_uses_configuration_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "MODEL_PATH",
        "MODEL_N_CTX",
        "MODEL_N_THREADS",
        "MODEL_N_GPU_LAYERS",
        "MODEL_VERBOSE",
        "MODEL_SYSTEM_PROMPT",
    ):
        monkeypatch.delenv(name, raising=False)
    application = FastAPI()

    async def run_lifespan() -> None:
        with patch("app.main.ModelService") as service_class:
            async with main.lifespan(application):
                pass
            service_class.assert_called_once_with(
                model_path=str((main.PROJECT_ROOT / main.DEFAULT_MODEL_PATH).resolve()),
                n_ctx=4096,
                n_threads=None,
                n_gpu_layers=0,
                verbose=False,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
            )

    asyncio.run(run_lifespan())
