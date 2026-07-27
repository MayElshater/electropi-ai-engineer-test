from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    gemini_model: str
    embedding_model: str

    documents_dir: Path
    chroma_persist_directory: Path
    chroma_collection_name: str

    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    min_relevance_score: float


def get_settings() -> Settings:
    settings = Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        documents_dir=BASE_DIR / "documents",
        chroma_persist_directory=Path(
            os.getenv(
                "CHROMA_PERSIST_DIRECTORY",
                str(BASE_DIR / "data" / "chroma"),
            )
        ),
        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME",
            "supportiq_knowledge_base",
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "700")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "4")),
        min_relevance_score=float(
            os.getenv("MIN_RELEVANCE_SCORE", "0.45")
        ),
    )

    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    if not 0.0 <= settings.min_relevance_score <= 1.0:
        raise ValueError(
            "MIN_RELEVANCE_SCORE must be between 0.0 and 1.0."
        )

    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_persist_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return settings