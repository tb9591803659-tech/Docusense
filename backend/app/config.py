"""Runtime configuration. Secrets live only in environment / .env (never sent to the frontend)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


@dataclass
class Settings:
    data_dir: Path = Path("./data")
    max_upload_mb: int = 25
    embedding_backend: str = "auto"  # auto | sentence-transformers | hashing
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 6
    min_score: float | None = None  # None -> use embedder default
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])


def load_settings() -> Settings:
    ms = os.getenv("MIN_SCORE")
    return Settings(
        data_dir=Path(os.getenv("DATA_DIR", "./data")),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "25")),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "auto"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        top_k=int(os.getenv("TOP_K", "6")),
        min_score=float(ms) if ms else None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        cors_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()],
    )
