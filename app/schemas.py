"""Request and response schemas for the RAG API."""
from pydantic import BaseModel, Field
from typing import Optional


class AskRequest(BaseModel):
    """Request model for the /ask endpoint."""
    question: str = Field(..., min_length=1, max_length=1000, description="Question about financial filings")


class Citation(BaseModel):
    """A citation from a source document."""
    source: str = Field(..., description="Source document filename")
    chunk_id: int = Field(..., description="Chunk index within the document")
    text: str = Field(..., description="Relevant text excerpt")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class AskResponse(BaseModel):
    """Response model for the /ask endpoint."""
    answer: str = Field(..., description="Generated answer or abstention message")
    citations: list[Citation] = Field(default_factory=list, description="Supporting citations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    abstained: bool = Field(default=False, description="Whether the system abstained from answering")
