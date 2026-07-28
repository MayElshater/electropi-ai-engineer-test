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
| Section 3 | LLM Quantization Benchmark | 🚧 |
| Section 4 | Production Deployment & Load Testing | 🚧 |

Each section is self-contained and includes:

- Detailed README
- Installation instructions
- Design decisions
- Assumptions
- Known limitations
- Half-page technical write-up
- Tests (where applicable)

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

---

## Technologies

- Python 3.12
- LiveKit Agents
- LangGraph
- FastAPI
- Docker
- Google Gemini
- Groq
- Deepgram
- AssemblyAI
- Cartesia
- ElevenLabs
- ChromaDB
- Hugging Face Embeddings
- Pytest

---

## Assumptions

- API credentials are supplied through `.env` files.
- External AI services require valid provider accounts.
- Docker is required for the deployment section.
- Quantization benchmarks require a CUDA-capable GPU (Google Colab T4 was used).

---

## Known Limitations

- Large demo videos are hosted externally instead of inside the repository.
- Some sections depend on third-party APIs and cloud services.
- Benchmark results may vary depending on hardware and provider availability.

---

## Submission Checklist

- ✅ Section 1 – LiveKit Voice Intake Assistant
- ✅ Section 2 – LangGraph RAG Pipeline
- ⏳ Section 3 – LLM Quantization Benchmark
- ⏳ Section 4 – Production Deployment

---

## Author

**May Mohamed Rashad**

AI Engineer