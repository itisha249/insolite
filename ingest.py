import argparse
import logging
from pathlib import Path
from typing import List

from openai import OpenAI

from app.config import settings
from app.text_utils import normalize_whitespace, paragraph_chunks, sliding_window
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)


def build_chunks(raw_text: str) -> List[str]:
    clean = normalize_whitespace(raw_text)
    paras = paragraph_chunks(clean)
    return sliding_window(paras, settings.max_chunk_chars, settings.chunk_overlap)


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    client = OpenAI(api_key=settings.openai_api_key)
    vectors: List[List[float]] = []
    for chunk in chunks:
        response = client.embeddings.create(
            model=settings.embeddings_model,
            input=chunk,
        )
        vectors.append(response.data[0].embedding)
    return vectors


def ingest_text(text: str) -> int:
    chunks = build_chunks(text)
    logger.info("Prepared %s text chunks", len(chunks))
    embeddings = embed_chunks(chunks)
    store = VectorStore(settings.storage_dir)
    store.rebuild(chunks, embeddings)
    return len(chunks)


def read_source(path: Path) -> str:
    with path.open("r", encoding="utf-8") as fh:
        return fh.read()


def cli() -> None:
    parser = argparse.ArgumentParser(description="Ingest brochure text into vector store.")
    parser.add_argument("--source", type=str, default=str(settings.default_source))
    args = parser.parse_args()
    settings.validate_api_key()
    text = read_source(Path(args.source))
    total = ingest_text(text)
    logger.info("Ingested %s chunks from %s", total, args.source)


if __name__ == "__main__":
    cli()

