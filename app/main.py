from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import ingest
from app.chat_service import ChatService
from app.config import settings
from app.schemas import ChatRequest, ChatResponse, IngestRequest
from app.vector_store import VectorStore

app = FastAPI(title="CHEMBULLS Chatbot", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
vector_store = VectorStore(settings.storage_dir)
chat_service: ChatService | None = None


@app.on_event("startup")
async def startup() -> None:
    global chat_service
    try:
        settings.validate_api_key()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    vector_store.load()
    chat_service = ChatService(vector_store)


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    if chat_service is None:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    try:
        answer, sources = chat_service.answer(payload.question, payload.history)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ChatResponse(answer=answer, sources=sources)


@app.post("/ingest")
async def reingest(payload: IngestRequest) -> dict:
    try:
        text = payload.text
        if not text:
            source_path = settings.default_source
            if payload.source_path:
                source_path = source_path.parent / payload.source_path
            if not source_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Source file not found: {source_path}. Make sure data.txt is in the repository."
                )
            text = ingest.read_source(source_path)
        count = ingest.ingest_text(text)
        vector_store.load()
        return {"chunks": count}
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {exc}. Make sure data.txt is committed to the repository."
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error during ingestion: {str(exc)}"
        ) from exc

