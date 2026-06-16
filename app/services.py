import json
import logging
import time
import httpx
from google import genai
from google.genai import types
from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini client — created once per process.
# ---------------------------------------------------------------------------

def _get_gemini_client() -> genai.Client:
    """Returns a configured Gemini API client.

    Raises:
        ValueError: If ``GEMINI_API_KEY`` is not set.
    """
    if not settings.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please add it to your .env file or change LLM_PROVIDER to 'ollama'."
        )
    return genai.Client(api_key=settings.gemini_api_key)


def _is_quota_error(exc: Exception) -> bool:
    """Returns ``True`` if the exception is a rate-limit, quota, or transient
    availability error that warrants trying the next model in the cascade."""
    msg = str(exc).lower()
    return any(kw in msg for kw in ("429", "quota", "exhausted", "rate", "resource_exhausted", "503", "unavailable"))


# ---------------------------------------------------------------------------
# Model cascade helper
# ---------------------------------------------------------------------------

def _gemini_generate_with_fallback(
    client: genai.Client,
    contents: str,
    gen_config: types.GenerateContentConfig,
) -> str:
    """Tries each model in ``settings.gemini_fallback_models`` in order.

    For every model it retries up to ``max_retries`` times on transient
    quota errors before moving on to the next model in the cascade.
    Once a model succeeds it returns immediately.  If every model in the
    cascade fails, the last exception is re-raised.

    Args:
        client:     An authenticated Gemini client.
        contents:   The prompt string to send.
        gen_config: Generation config (mime-type, etc.).

    Returns:
        The model's response text (stripped).

    Raises:
        Exception: The last exception raised if all cascade models fail.
    """
    # Build the ordered candidate list:
    # primary model first, then unique extras from the fallback list.
    primary = settings.llm_model
    cascade: list[str] = [primary] + [
        m for m in settings.gemini_fallback_models if m != primary
    ]

    last_exc: Exception = RuntimeError("No models available.")
    max_retries = 3  # retries *within* a single model before cascading

    for model in cascade:
        model_name = model if model.startswith("models/") else f"models/{model}"
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config,
                )
                if model != primary:
                    logger.info(f"Cascade: succeeded with fallback model '{model}'.")
                return response.text.strip()

            except Exception as exc:
                last_exc = exc
                if _is_quota_error(exc):
                    if attempt < max_retries - 1:
                        wait = (attempt + 1) * 5
                        logger.warning(
                            f"Quota/rate-limit on '{model}' "
                            f"(attempt {attempt + 1}/{max_retries}). "
                            f"Retrying in {wait}s…"
                        )
                        time.sleep(wait)
                    else:
                        logger.warning(
                            f"Quota/rate-limit on '{model}' — cascading to next model."
                        )
                        break  # try next model in cascade
                else:
                    # Non-quota error: don't cascade, surface immediately.
                    logger.error(f"Non-quota error on model '{model}': {exc}")
                    raise

    logger.error("All models in the cascade failed.")
    raise last_exc


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
    """Generates embeddings for a list of texts using the configured provider.

    - **Gemini**: Uses the ``google-genai`` SDK with ``settings.embedding_model``
      (default: ``gemini-embedding-001``).
    - **Ollama**: POSTs to ``/api/embeddings`` on the local Ollama server.

    Args:
        texts:     List of text strings to embed.
        task_type: Gemini task-type hint — ``retrieval_document`` or
                   ``retrieval_query``.

    Returns:
        A list of embedding vectors (one per input text).
    """
    if settings.llm_provider == "gemini":
        client = _get_gemini_client()

        model_name = settings.embedding_model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        task_type_map: dict[str, str] = {
            "retrieval_document": "RETRIEVAL_DOCUMENT",
            "retrieval_query": "RETRIEVAL_QUERY",
        }
        sdk_task = task_type_map.get(task_type, "RETRIEVAL_DOCUMENT")
        embed_config = types.EmbedContentConfig(task_type=sdk_task)

        batch_size = 50
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(6):
                try:
                    response = client.models.embed_content(
                        model=model_name,
                        contents=batch,
                        config=embed_config,
                    )
                    all_embeddings.extend([e.values for e in response.embeddings])
                    time.sleep(1.0)  # stay within free-tier rate limits
                    break
                except Exception as exc:
                    if _is_quota_error(exc) and attempt < 5:
                        wait = (attempt + 1) * 10
                        logger.warning(
                            f"Embedding rate-limit. Retrying batch "
                            f"{i // batch_size + 1} in {wait}s… "
                            f"(attempt {attempt + 1}/6)"
                        )
                        time.sleep(wait)
                    else:
                        logger.error(f"Gemini embedding failed: {exc}")
                        raise

        return all_embeddings

    elif settings.llm_provider == "ollama":
        all_embeddings: list[list[float]] = []
        try:
            with httpx.Client(timeout=60.0) as http:
                for text in texts:
                    resp = http.post(
                        f"{settings.ollama_base_url}/api/embeddings",
                        json={"model": settings.embedding_model, "prompt": text},
                    )
                    resp.raise_for_status()
                    all_embeddings.append(resp.json()["embedding"])
        except httpx.ConnectError:
            raise ConnectionError(
                f"Could not connect to Ollama at {settings.ollama_base_url}. "
                "Is Ollama running?"
            )
        except Exception as exc:
            logger.error(f"Ollama embedding failed: {exc}")
            raise
        return all_embeddings

    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


# ---------------------------------------------------------------------------
# LLM text generation
# ---------------------------------------------------------------------------

def get_llm_response(prompt: str, json_mode: bool = False) -> str:
    """Calls the configured LLM and returns the response text.

    For Gemini, uses a model cascade: tries ``settings.llm_model`` first,
    then falls back through ``settings.gemini_fallback_models`` on quota /
    rate-limit errors.

    Args:
        prompt:    The full prompt string.
        json_mode: When ``True`` the model is instructed to respond with a
                   valid JSON object.

    Returns:
        The model response as a stripped string.
    """
    if settings.llm_provider == "gemini":
        client = _get_gemini_client()
        gen_config = (
            types.GenerateContentConfig(response_mime_type="application/json")
            if json_mode
            else types.GenerateContentConfig()
        )
        return _gemini_generate_with_fallback(client, prompt, gen_config)

    elif settings.llm_provider == "ollama":
        try:
            with httpx.Client(timeout=90.0) as http:
                payload: dict = {
                    "model": settings.llm_model,
                    "prompt": prompt,
                    "stream": False,
                }
                if json_mode:
                    payload["format"] = "json"
                resp = http.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json()["response"].strip()
        except httpx.ConnectError:
            raise ConnectionError(
                f"Could not connect to Ollama at {settings.ollama_base_url}. "
                "Is Ollama running?"
            )
        except Exception as exc:
            logger.error(f"Ollama LLM call failed: {exc}")
            raise

    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


# ---------------------------------------------------------------------------
# Query pre-processing (translation + expansion)
# ---------------------------------------------------------------------------

def process_query_for_search(query: str) -> list[str]:
    """Translates a Czech query to English and optionally splits comparison
    queries into multiple sub-queries for better cross-era retrieval.

    Strategy:
        1. Translate the query to English.
        2. If the query compares or synthesises multiple historical periods
           (Victorian, Edwardian, Tudor, …) split it into 2–3 semicolon-
           separated English sub-queries.
        3. Otherwise return a single translated query.

    Args:
        query: The original user query (Czech or English).

    Returns:
        A non-empty list of English search queries.
    """
    prompt = (
        "You are an assistant that processes user search queries for a RAG system.\n"
        f"The user query is: '{query}'\n\n"
        "Instructions:\n"
        "1. Translate the query to English.\n"
        "2. If the query asks to compare or synthesize information from multiple periods "
        "(e.g., Victorian, Edwardian, Tudor) or different topics, split it into 2 or 3 "
        "distinct English search queries separated by semicolons ';'.\n"
        "3. If it is a simple factual query, just output the English translation.\n"
        "4. Return ONLY the English translation or the semicolon-separated sub-queries. "
        "Do not include any explanation or introduction."
    )

    try:
        response_text = get_llm_response(prompt)
        sub_queries = [q.strip() for q in response_text.split(";") if q.strip()]
        if not sub_queries:
            sub_queries = [query]
        logger.info(f"Processed query '{query}' → search queries: {sub_queries}")
        return sub_queries
    except Exception as exc:
        logger.warning(f"Query expansion failed: {exc}. Falling back to original query.")
        return [query]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(query: str, top_k: int = 3) -> tuple[str, list[dict]]:
    """Translates/expands the query, searches ChromaDB (cosine similarity),
    merges & deduplicates results, and returns context + source metadata.

    Args:
        query:  The original user query.
        top_k:  Number of top chunks to return after merging all sub-query results.

    Returns:
        ``(context_str, sources)`` where *context_str* is the concatenated
        chunk text with timestamps and *sources* is a list of dicts with
        keys ``timestamp``, ``excerpt``, and ``similarity``.
    """
    from app.database import get_chroma_client  # local import — avoids circular deps

    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name="documentary_transcript",
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        raise ValueError("ChromaDB collection is empty. Please run ingestion first.")

    search_queries = process_query_for_search(query)

    try:
        query_embeddings = get_embeddings(search_queries, task_type="retrieval_query")
    except Exception as exc:
        logger.error(f"Failed to generate query embeddings: {exc}")
        raise

    all_results: list[dict] = []
    seen_ids: set[str] = set()

    for q_emb in query_embeddings:
        results = collection.query(query_embeddings=[q_emb], n_results=top_k)

        if results and "documents" in results and results["documents"]:
            for idx, doc_id in enumerate(results["ids"][0]):
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    similarity = 1.0 - results["distances"][0][idx]
                    all_results.append(
                        {
                            "id": doc_id,
                            "document": results["documents"][0][idx],
                            "metadata": results["metadatas"][0][idx],
                            "similarity": similarity,
                        }
                    )

    all_results.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = all_results[:top_k]

    context_parts: list[str] = []
    sources: list[dict] = []

    for res in top_results:
        meta = res["metadata"]
        timestamp = meta.get("timestamp", "00:00:00")
        excerpt = meta.get("excerpt", res["document"])
        context_parts.append(f"[{timestamp}] {res['document']}")
        sources.append({"timestamp": timestamp, "excerpt": excerpt, "similarity": res["similarity"]})

    return "\n\n".join(context_parts), sources


# ---------------------------------------------------------------------------
# Full RAG pipeline
# ---------------------------------------------------------------------------

def answer_question(query: str) -> tuple[str, list[dict]]:
    """Executes the full RAG pipeline.

    Steps:
        1. Retrieve relevant context chunks from ChromaDB.
        2. Build a structured prompt requiring JSON output.
        3. Call the LLM (with model cascade) in JSON mode.
        4. Parse the response and apply the out-of-scope fallback rule.

    Args:
        query: The user's question (Czech or English).

    Returns:
        ``(answer, sources)`` where *answer* is a plain-text string and
        *sources* is a list of dicts with keys ``timestamp`` and ``excerpt``.
    """
    context_str, sources = retrieve_context(query, top_k=3)

    prompt = (
        "Context information is below.\n"
        "---------------------\n"
        f"{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        f"Query: {query}\n\n"
        "Strict Rules:\n"
        "1. Answer the query in natural language based SOLELY on the context provided.\n"
        '2. If the context does not contain the answer, set the \'answer\' field to exactly: '
        '"I am sorry, but the provided material does not contain an answer to this question." '
        "and set the 'sources' field to an empty list []. Do not attempt to fabricate an answer.\n"
        "3. Output MUST be a valid JSON object matching the following structure:\n"
        "{\n"
        '  "answer": "your answer string based solely on context",\n'
        '  "sources": [\n'
        "     {\n"
        '       "timestamp": "the timestamp of the chunk used (e.g. HH:MM:SS or MM:SS)",\n'
        '       "excerpt": "exact short excerpt text from that chunk"\n'
        "     }\n"
        "  ]\n"
        "}\n\n"
        "Do not include any other text, markdown block wraps (like ```json), "
        "or explanations outside of the JSON object."
    )

    logger.info("Sending prompt to LLM (JSON mode) via model cascade…")
    try:
        response_text = get_llm_response(prompt, json_mode=True)

        # Strip markdown fences if the model wraps the output anyway.
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        answer: str = data.get("answer", "")
        sources_list: list = data.get("sources", [])

        formatted_sources: list[dict] = [
            {"timestamp": src["timestamp"], "excerpt": src["excerpt"]}
            for src in sources_list
            if isinstance(src, dict) and "timestamp" in src and "excerpt" in src
        ]

        fallback_msg = (
            "I am sorry, but the provided material does not contain an answer to this question."
        )
        if fallback_msg.lower() in answer.lower():
            answer = fallback_msg
            formatted_sources = []

        return answer, formatted_sources

    except Exception as exc:
        logger.warning(
            f"JSON-mode response failed ({exc}). Falling back to plain-text generation."
        )
        prompt_plain = (
            "Context information is below.\n"
            "---------------------\n"
            f"{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the query.\n"
            f"Query: {query}\n\n"
            "Strict Rules:\n"
            "1. Answer the query in natural language based SOLELY on the context provided.\n"
            '2. If the context does not contain the answer, reply exactly with: '
            '"I am sorry, but the provided material does not contain an answer to this question." '
            "Do not attempt to fabricate an answer."
        )
        answer = get_llm_response(prompt_plain, json_mode=False)
        fallback_msg = (
            "I am sorry, but the provided material does not contain an answer to this question."
        )
        if fallback_msg.lower() in answer.lower():
            answer = fallback_msg
            sources = []
        return answer, sources
