from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3)
    history: Optional[List[str]] = Field(default=None)


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


class IngestRequest(BaseModel):
    text: Optional[str] = None
    source_path: Optional[str] = None

