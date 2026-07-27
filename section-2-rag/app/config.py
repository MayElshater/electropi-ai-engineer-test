from dataclasses import dataclass
import math
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
    retrieval_maximum_distance: float
    


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
        retrieval_maximum_distance=float(
            os.getenv("RETRIEVAL_MAXIMUM_DISTANCE", "1.0")
        ),
       
    )

    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    

    if (
        not math.isfinite(settings.retrieval_maximum_distance)
        or settings.retrieval_maximum_distance < 0
    ):
        raise ValueError(
            "RETRIEVAL_MAXIMUM_DISTANCE must be finite and at least 0."
        )

    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_persist_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return settings