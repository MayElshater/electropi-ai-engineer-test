# Local LLM Inference API using FastAPI + llama.cpp

## Overview

This project exposes a local GGUF language model through a small HTTP API. It
uses FastAPI for request handling and OpenAPI documentation, while
`llama-cpp-python` runs inference locally through llama.cpp. The service loads
the model once when the application starts and returns generated text together
with token usage, generation duration, and throughput metadata.

The included configuration targets the
`qwen2.5-1.5b-instruct-q4_k_m.gguf` model, but another compatible GGUF model can
be selected through environment variables.

## Demo
### Before Containerization
<video controls src="https://github.com/MayElshater/electropi-ai-engineer-test/blob/main/section-4-deployment/assests/LLM_Without_Containerization.mp4"></video>

### After Containerization
<video controls src="https://github.com/MayElshater/electropi-ai-engineer-test/blob/main/section-4-deployment/assests/LLM_With_Containerization.mp4" title="/home/mayrashad/projects/electropi-ai-engineer-test/section-4-deployment/assests/LLM_Without_Containerization.mp4"></video>

## Features

- FastAPI application with generated OpenAPI documentation
- Local inference with GGUF models
- llama.cpp integration through `llama-cpp-python`
- Model-aware chat completion using system and user messages
- Environment-based model and runtime configuration
- Pydantic v2 request and response validation
- Unit tests that do not load the real model
- Docker and Docker Compose support
- Non-root Docker runtime

## Project structure

```text
section-4-deployment/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model_service.py
│   └── schemas.py
├── models/
│   └── qwen2.5-1.5b-instruct-q4_k_m.gguf
├── tests/
│   ├── __init__.py
│   ├── test_main.py
│   ├── test_model_service.py
│   └── test_schemas.py
├── assests/
│   ├── LLM_Without_Containerization.mp4
│   ├── LLM_With_Containerization.mp4
├── .dockerignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11 or later
- A llama.cpp-compatible GGUF model
- Sufficient memory for the selected model and context size
- Docker with Docker Compose, if using the containerized workflow

The Docker image uses Python 3.11. The project is also compatible with Python
3.12.

## Installation

From the repository root:

```bash
cd section-4-deployment

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Place the GGUF model in `models/`, or set `MODEL_PATH` to another local GGUF
file.

## Environment Variables

Copy the provided example when custom configuration is needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_PATH` | `models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | GGUF model path. Relative paths are resolved from the project root; absolute paths are also supported. |
| `MODEL_N_CTX` | `4096` | Maximum llama.cpp context size. |
| `MODEL_N_THREADS` | Empty | CPU thread count. When empty or unset, llama.cpp chooses its default. |
| `MODEL_N_GPU_LAYERS` | `0` | Number of model layers offloaded to the GPU. `0` selects CPU-only execution. |
| `MODEL_VERBOSE` | `false` | Enables verbose llama.cpp output. Accepted values are `true`, `1`, `yes`, `on`, `false`, `0`, `no`, and `off`. |
| `MODEL_SYSTEM_PROMPT` | `You are a helpful, accurate, and concise AI assistant.` | System instruction supplied to every chat completion. |

The application reads normal operating-system environment variables directly;
it does not load `.env` itself. For a local shell, export the example values
before starting the API:

```bash
set -a
source .env
set +a
```

Docker Compose loads `.env` through its `env_file` configuration.

## Running locally

Activate the virtual environment, export any desired configuration, and run:

```bash
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive API documentation
is available at `http://127.0.0.1:8000/docs`.

`--reload` is intended for local development. The model is reloaded whenever
the development server restarts.

## Running with Docker

Ensure the model is present in `models/` and create `.env` from
`.env.example`. Then run:

```bash
docker compose up --build
```

The Compose service publishes port `8000` and mounts the local `models/`
directory at `/app/models`. Application source is copied into the image and is
not bind-mounted.

## API

### `GET /`

Returns basic service information:

```json
{
  "service": "Local LLM Inference API",
  "status": "running"
}
```

### `GET /health`

Provides a lightweight liveness check without running inference:

```json
{
  "status": "healthy"
}
```

### `POST /generate`

Generates a chat completion using the configured local model.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain model quantization in two sentences.",
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "stop": null
  }'
```

Example response:

```json
{
  "response": "Model quantization reduces the precision of model weights to lower memory and compute requirements. It can improve inference efficiency while preserving most of the model's quality.",
  "model": "qwen2.5-1.5b-instruct-q4_k_m",
  "prompt_tokens": 18,
  "completion_tokens": 31,
  "total_tokens": 49,
  "generation_time_seconds": 1.42,
  "tokens_per_second": 21.83
}
```

Generation parameters are optional apart from `prompt`. Defaults are 256
maximum tokens, a temperature of `0.7`, top-p of `0.9`, and no stop sequences.
Validation errors return FastAPI's standard HTTP 422 response. Inference
failures return HTTP 500 without exposing internal backend details.

## Testing

Run the complete test suite from `section-4-deployment`:

```bash
pytest -v
```

The tests mock `llama_cpp.Llama`, so they run without loading the GGUF model or
performing inference.

## Design decisions

- **Single application-scoped `ModelService`:** The lifespan creates one
  service and loads the model once per API process. This avoids expensive model
  initialization on every request without introducing a global singleton
  pattern.
- **FastAPI lifespan startup:** Model loading occurs before the application
  accepts traffic, and shutdown logging is handled through the same lifecycle
  boundary. Deprecated startup event handlers are not used.
- **Chat completion:** Requests use the model's chat template with explicit
  system and user messages rather than sending an unstructured completion
  prompt.
- **Environment configuration:** Model location and llama.cpp runtime settings
  can change between deployments without modifying application code.
- **Defensive validation:** Pydantic validates public input and output schemas,
  while the model service validates llama.cpp response structure and token
  metadata. Malformed backend responses and inference failures are converted
  into controlled errors without leaking filesystem paths or internal exception
  details.


## Half-Page Technical Write-up

The current implementation is intentionally simple: a single FastAPI application loads one GGUF model during startup and serves inference requests through llama-cpp-python. This architecture is suitable for development, demonstrations, and small-scale deployments because it avoids repeatedly loading the model while keeping the implementation straightforward.

If the service needed to support approximately 50 concurrent users, I would introduce several architectural improvements.

First, I would enable streaming responses so users begin receiving generated tokens immediately instead of waiting for the full completion. This significantly improves perceived responsiveness, particularly for longer generations.

Second, I would add request queueing and batching. Rather than processing each request independently, incoming requests could be queued briefly and grouped into batches when supported by the inference backend. This increases GPU utilization and improves overall throughput under sustained load.

Third, I would run multiple inference workers behind a load balancer. Instead of relying on a single FastAPI process, several replicas could be deployed, with requests distributed across them using a reverse proxy or Kubernetes Service. Horizontal scaling would allow the system to handle higher traffic while improving availability.

To reduce unnecessary computation, I would also introduce response caching for repeated prompts and common system requests. For conversational workloads, session state could be stored in Redis while inference results for identical prompts could be cached with a configurable expiration policy.

Finally, I would improve observability by collecting metrics such as request latency, time-to-first-token (TTFT), throughput, GPU utilization, and queue length using Prometheus and Grafana. These metrics would guide autoscaling decisions and help identify performance bottlenecks before they affect users.

With these additions-streaming, batching, multiple workers, caching, monitoring, and autoscaling-the service would evolve from a single-instance inference API into a production-ready architecture capable of serving dozens of concurrent users efficiently while maintaining low latency and high availability.
