from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    question: str

class SourceReference(BaseModel):
    timestamp: str  # Format HH:MM:SS or MM:SS extracted from chunk metadata
    excerpt: str    # Short text excerpt from the source chunk

class QuestionResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
