import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class Settings(BaseModel):
    openai_api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    embeddings_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-4.1"
    max_chunk_chars: int = 900
    chunk_overlap: int = 120
    storage_dir: Path = Field(default_factory=lambda: Path("storage") / "vector_store")
    default_source: Path = Field(default_factory=lambda: Path(".history") / "data.txt")

    def ensure_storage(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def validate_api_key(self) -> None:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")


settings = Settings()
settings.ensure_storage()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

