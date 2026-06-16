from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    question: str

class SourceReference(BaseModel):
    timestamp: str  # Formát HH:MM:SS nebo MM:SS vytažený z metadat
    excerpt: str    # Krátký výsek textu, ze kterého se čerpalo

class QuestionResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
