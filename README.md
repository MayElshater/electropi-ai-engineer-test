# ElectroPi AI Engineer Technical Assessment

This repository contains my submission for the ElectroPi AI Engineer Technical Assessment.

The submission is organized into four independent sections, each focusing on a different area of AI engineering, from real-time voice agents to Retrieval-Augmented Generation (RAG), LLM optimization, and production deployment.

---

## Repository Structure

```text
.
├── 
├── section-1-livekit/
├── section-2-rag/
├── section-3-quantization/
└── section-4-deployment/
```

---

## Sections

| Section | Description | Status |
|----------|-------------|--------|
| Section 1 | LiveKit Structured Voice Intake Assistant | ✅ |
| Section 2 | Grounded RAG Pipeline with LangGraph Verification | ✅ |
| Section 3 | LLM Quantization Benchmark | ✅ |
| Section 4 | Production Deployment & Load Testing | ✅ |

Each section is self-contained and includes:

- Detailed README
- Installation instructions
- Design decisions
- Assumptions
- Known limitations
- Half-page technical write-up
- Tests (where applicable)

---

### Section Highlights

**Section 1 — Real-Time Voice AI**
- LiveKit voice agent
- Multi-provider STT/LLM/TTS fallback
- Structured information extraction
- Production-ready testing

**Section 2 — Grounded RAG**
- LangGraph workflow
- Chroma vector database
- Grounded answer verification
- End-to-end automated tests

**Section 3 — LLM Quantization**
- Qwen2.5-1.5B-Instruct benchmarking
- BF16 vs 4-bit NF4 (BitsAndBytes)
- VRAM, throughput, latency, and quality comparison
- Production deployment trade-off analysis

**Section 4 — Model Deployment**
- GGUF model served through FastAPI
- llama-cpp-python local inference
- Dockerized deployment
- Streaming endpoint
- Concurrent load testing

---

# Quick Start

Clone the repository:

```bash
git clone https://github.com/<username>/electropi-ai-engineer-test.git

cd electropi-ai-engineer-test
```

Each section can be executed independently by following the instructions inside its own README.

---

## Running the Sections

### Section 1

```bash
cd section-1-livekit
```

Follow:

```
section-1-livekit/README.md
```

---

### Section 2

```bash
cd section-2-rag
```

Follow:

```
section-2-rag/README.md
```

---

### Section 3

```bash
cd section-3-quantization
```

Follow:

```
section-3-quantization/README.md
```

---

### Section 4

```bash
cd section-4-deployment
```

Follow:

```
section-4-deployment/README.md
```

---

## Demo Videos

### Section 1 — LiveKit Voice Assistant

- 🎥 End-to-End Voice Intake Workflow
- 🎥 Provider Fallback & Production Testing

### Section 2 — LangGraph RAG

- 🎥 Grounded RAG Pipeline Demonstration

Demo videos are hosted externally to keep the repository lightweight.

### Section 4 — Model Deployment
- 🎥 Local GGUF API (Before Docker)
- 🎥 Dockerized Deployment
---

## Technologies

- Python 3.12
- LiveKit Agents
- LangGraph
- FastAPI
- Docker
- llama-cpp-python
- Hugging Face Transformers
- BitsAndBytes
- GGUF
- Google Gemini
- Groq
- Deepgram
- AssemblyAI
- Cartesia
- ElevenLabs
- ChromaDB
- Pytest

---

## Assumptions

- API credentials are supplied through `.env` files.
- External AI services require valid provider accounts.
- Section 3 benchmarks were executed on Google Colab using an NVIDIA Tesla T4 GPU.
- Section 4 supports local GGUF inference and Docker deployment.
- API credentials are supplied through `.env` files..

---

## Known Limitations

- Large demo videos are hosted externally instead of inside the repository.
- Some sections depend on third-party APIs and cloud services.
- Benchmark results may vary depending on hardware and provider availability.

---

## Submission Checklist

- ✅ Section 1 – LiveKit Voice Intake Assistant
- ✅ Section 2 – LangGraph RAG Pipeline
- ✅ Section 3 – LLM Quantization Benchmark
- ✅ Section 4 – Production Deployment

---

## Key Results

- Implemented a production-style LiveKit voice assistant with provider fallback.
- Built a grounded LangGraph RAG pipeline with verification.
- Benchmarked BF16 and 4-bit NF4 quantization on Qwen2.5-1.5B-Instruct.
- Deployed a GGUF model behind a FastAPI REST API using llama-cpp-python.
- Containerized the inference service with Docker.
- Added automated tests across all sections.

---

## Author

**May Mohamed Rashad**

AI Engineer