"""
Configuration – all secrets via environment variables.
Copy .env.example to .env and fill in real values.
"""

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # ── Gemini API keys (failover pool) ────────────────────────────────────
    gemini_api_keys: List[str] = field(default_factory=lambda: [
        k for k in [
            os.getenv("GEMINI_API_KEY_1", ""),
            os.getenv("GEMINI_API_KEY_2", ""),
            os.getenv("GEMINI_API_KEY_3", ""),
        ] if k
    ])

    # ── Model names ─────────────────────────────────────────────────────────
    embedding_model: str = "gemini-embedding-001"
    primary_model: str = os.getenv("PRIMARY_MODEL", "gemini-3.1-flash-lite")
    fallback_models: List[str] = field(default_factory=lambda: [
        "gemini-flash-lite-latest",
        "gemini-robotics-er-1.6-preview",
    ])

    # ── RAG settings ───────────────────────────────────────────────────────
    top_k: int = 3
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))

    # ── Paths ────────────────────────────────────────────────────────────────
    knowledge_base_path: str = "embeddings/knowledge_base.json"
    faiss_index_path: str = "vector_store/faiss.index"
    metadata_path: str = "vector_store/metadata.json"

    # ── Twilio ───────────────────────────────────────────────────────────────
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twiml_voice: str = "Polly.Amy"

    # ── App ──────────────────────────────────────────────────────────────────
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"


settings = Settings()
