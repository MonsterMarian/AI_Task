# Documentary Q&A Backend (RAG)

A modular, high-performance RAG-based backend service built with FastAPI and ChromaDB. It allows users to ask natural language questions (in Czech or English) about a long historical documentary and receive accurate answers grounded in the text, complete with timestamped source references.

It includes a premium responsive Web UI served directly by the app.

---

## Quick Start (Docker Compose)

The easiest way to run the service is using Docker Compose.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MonsterMarian/AI_Task.git
   cd AI_Task
   ```

2. **Start the application** (`.env` is already included with a working Gemini API key):
   ```bash
   docker compose up --build
   ```

3. **Access the application**:
   - **Web UI**: Open [http://localhost:8000](http://localhost:8000) in your browser.
   - **Interactive API Docs**: Go to [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Local Installation & Development

If you prefer to run the application directly on your host machine:

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   Create a `.env` file (see `.env.example`).

4. **Start the FastAPI server**:
   ```bash
   uvicorn app.main:app --reload
   ```

---

## Verification & Testing

### 1. Automated Verification Script (`verify_rag.py`)
An automated validation script that runs the 5 required evaluation scenarios directly against the Python backend. It checks factual accuracy, multi-era synthesis, person name matching, and out-of-scope query behaviour.

```bash
# Windows
.venv\Scripts\python verify_rag.py

# Linux/macOS
python verify_rag.py
```

### 2. Unit Tests
```bash
pytest
```

---

## Project Structure

```
app/
    __init__.py
    main.py          - FastAPI application, endpoint POST /ask
    config.py        - Config loader via Pydantic-Settings
    database.py      - ChromaDB setup and data ingestion
    services.py      - Core RAG logic (embeddings, search, prompts)
    models.py        - Pydantic request/response schemas
    static/
        index.html   - Web UI interface
        styles.css   - Responsive design styles
data/
    transcript.txt   - Ingest source document
DESIGN.md            - Design decisions (chunking, retrieval, prompts)
Dockerfile
docker-compose.yml
requirements.txt
verify_rag.py        - CLI verification test runner
.env.example         - Template configuration file
```

---

## Key Design Decisions & Traps

1. **Cross-Lingual Search**: Because the transcript is in English but user questions are in Czech, we implement an LLM-based query translation step before querying ChromaDB. This yields significantly higher cosine similarity matches.
2. **Era Synthesis**: Standard vector search tends to cluster documents in a single era. For comparison queries, the LLM splits the question into Victorian and Edwardian sub-queries, queries ChromaDB for both, and merges the results to build a balanced context.
3. **Out-of-Scope Protection**: Out-of-scope queries (e.g. *Ancient Rome*) return the exact required message `"I am sorry, but the provided material does not contain an answer to this question."` and empty sources `[]`.
