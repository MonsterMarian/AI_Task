import os
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.models import QuestionRequest, QuestionResponse, SourceReference
from app.config import settings
from app.database import get_chroma_client, ingest_data
from app.services import answer_question

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Documentary Q&A Backend",
    description="RAG-based backend for answering questions about historical domestic dangers",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing ChromaDB database on startup...")
    try:
        client = get_chroma_client()
        ingest_data(client)
    except Exception as e:
        logger.error(f"Error during startup ingestion: {e}")
        logger.error("Startup ingestion failed, but the server will remain running. "
                     "Please ensure your API key or Ollama connection is correct.")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question_endpoint(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    try:
        answer, sources = answer_question(request.question)
        source_refs = [
            SourceReference(timestamp=src["timestamp"], excerpt=src["excerpt"])
            for src in sources
        ]
        return QuestionResponse(answer=answer, sources=source_refs)
    except ValueError as ve:
        logger.error(f"Configuration or validation error: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except ConnectionError as ce:
        logger.error(f"Connection error to provider: {ce}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(ce)
        )
    except Exception as e:
        logger.error(f"Unexpected error processing request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        )

# Ensure static files directory exists and mount it
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Welcome to the Documentary Q&A Backend API. Please create the index.html file in app/static/ to view the Web UI."}
