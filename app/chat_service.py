import logging
from typing import List

from openai import OpenAI

from app.config import settings
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, store: VectorStore):
        self.store = store
        self.client = OpenAI(api_key=settings.openai_api_key)

    def answer(self, question: str, history: List[str] | None = None) -> tuple[str, List[str]]:
        history = history or []
        embeddings = self.client.embeddings.create(
            model=settings.embeddings_model,
            input=question,
        )
        query_vector = embeddings.data[0].embedding
        top_chunks = self.store.search(query_vector)
        context = "\n\n".join(
            f"[{chunk.chunk_id}]\n{chunk.text}" for chunk in top_chunks
        )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for CHEMBULLS INDIA LLP. Use only provided context.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]
        completion = self.client.responses.create(
            model=settings.chat_model,
            input=messages,
            max_output_tokens=600,
        )
        answer = completion.output_text.strip()
        sources = [chunk.chunk_id for chunk in top_chunks]
        return answer, sources

