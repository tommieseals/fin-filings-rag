"""Pydantic models for request/response validation."""
from pydantic import BaseModel, Field
from typing import Optional

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)

class Citation(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_file: str

class AskResponse(BaseModel):
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    citations: list[Citation]
    abstained: bool = False
    message: Optional[str] = None
