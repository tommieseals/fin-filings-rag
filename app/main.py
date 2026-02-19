"""FastAPI application for SEC filing RAG."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import AskRequest, AskResponse
from .rag import get_engine

app = FastAPI(
    title="SEC Filing RAG API",
    description="RAG for SEC 10-K filings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        engine = get_engine()
        response = engine.synthesize(request.question)
        return response
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Index not initialized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def stats():
    try:
        engine = get_engine()
        return {
            "indexed_chunks": len(engine.documents),
            "vocabulary_size": len(engine.vectorizer.vocabulary_),
        }
    except FileNotFoundError:
        return {"error": "Index not initialized"}
