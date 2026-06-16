# Documentary Q&A Backend

A RAG-based backend that lets you ask natural language questions about a historical documentary transcript and get accurate, timestamped answers grounded strictly in the source material. Includes a full Web UI served directly from the API.

---

## Getting Started

```bash
git clone https://github.com/MonsterMarian/AI_Task.git
cd AI_Task
docker compose up --build
```

That's it. The `.env` with a working Gemini API key is already included.

Once running:

| Interface | URL |
|---|---|
| Web UI | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| API Endpoint | `POST http://localhost:8000/ask` |

---

## API Reference

**Request**

```json
POST /ask
{
  "question": "What was the result of the borax experiment in milk?"
}
```

**Response**

```json
{
  "answer": "The borax experiment neutralised the acid in the milk...",
  "sources": [
    {
      "timestamp": "00:10:33",
      "excerpt": "Of a product called borax, an alkali..."
    }
  ]
}
```

---

## Running Locally (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## Configuration

All configuration is done via environment variables. Copy `.env.example` to `.env` to customise.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `ollama` |
| `GEMINI_API_KEY` | — | Get one free at [aistudio.google.com](https://aistudio.google.com) |
| `LLM_MODEL` | `gemini-2.5-flash` | Any model from `gemini models list` |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | For fully local offline setup |
| `CHROMA_DB_DIR` | `./chroma_db` | Vector DB persistence path |

---

## Verification

Run all 5 acceptance scenarios against the live backend:

```bash
python verify_rag.py
```

Run unit tests:

```bash
pytest
```

---

## Project Structure

```
app/
    main.py          — FastAPI app, POST /ask endpoint
    services.py      — RAG pipeline (embed, retrieve, prompt, generate)
    database.py      — ChromaDB client and transcript ingestion
    config.py        — Settings loaded from .env via Pydantic
    models.py        — Request / response schemas
    static/          — Web UI (HTML + CSS)
data/
    transcript.txt   — Source transcript (~190 KB)
DESIGN.md            — Architecture and design decisions
Dockerfile
docker-compose.yml
requirements.txt
verify_rag.py        — End-to-end acceptance test runner
paragraph.txt        — Submission summary paragraph
```

---

## Design Highlights

**Cross-lingual retrieval** — Queries arrive in Czech, transcript is in English. Before hitting the vector database, the LLM translates the query to English, yielding significantly better cosine similarity matches.

**Query decomposition** — Comparison questions (e.g. Victorian vs. Edwardian) are split by the LLM into separate sub-queries. Results from each are merged and re-ranked, ensuring context from both parts of the document is retrieved.

**Model cascade** — If the primary Gemini model hits its free-tier quota or is temporarily unavailable, the service automatically falls back through a list of alternative models. No crashes, no manual intervention needed.

**Out-of-scope detection** — Questions with no answer in the transcript return a fixed message and an empty `sources` array rather than a hallucinated answer.

For full implementation details see [DESIGN.md](DESIGN.md).
