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

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {"name": "SEC Filing RAG API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        engine = get_engine()
        response = engine.synthesize(request.question)
        return AskResponse(
            answer=response["answer"],
            citations=response["citations"],
            confidence=response["confidence"],
            abstained=response["abstained"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
