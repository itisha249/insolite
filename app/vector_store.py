import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict


class VectorStore:
    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.meta_file = store_path / "chunks.json"
        self.embedding_file = store_path / "embeddings.npy"
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def load(self) -> None:
        if not self.meta_file.exists() or not self.embedding_file.exists():
            logger.warning("Vector store not found at %s", self.store_path)
            self.chunks = []
            self.embeddings = None
            return
        with self.meta_file.open("r", encoding="utf-8") as fh:
            entries = json.load(fh)
        self.chunks = [Chunk(**entry) for entry in entries]
        self.embeddings = np.load(self.embedding_file)
        logger.info("Loaded %s chunks from vector store", len(self.chunks))

    def save(self) -> None:
        self.store_path.mkdir(parents=True, exist_ok=True)
        serializable = [chunk.__dict__ for chunk in self.chunks]
        with self.meta_file.open("w", encoding="utf-8") as fh:
            json.dump(serializable, fh, ensure_ascii=False, indent=2)
        np.save(self.embedding_file, self.embeddings)
        logger.info("Persisted %s chunks to vector store", len(self.chunks))

    def rebuild(self, texts: Sequence[str], embeddings: Sequence[Sequence[float]]) -> None:
        self.chunks = [
            Chunk(chunk_id=f"chunk-{idx}", text=texts[idx], metadata={"source": "data.txt"})
            for idx in range(len(texts))
        ]
        self.embeddings = np.array(embeddings, dtype=np.float32)
        self.save()

    def search(self, query_embedding: Sequence[float], top_k: int = 4) -> List[Chunk]:
        if self.embeddings is None or not len(self.chunks):
            raise ValueError("Vector store is empty. Run ingestion first.")
        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        sims = cosine_similarity(query, self.embeddings)[0]
        top_indices = sims.argsort()[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices]

