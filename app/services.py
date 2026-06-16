import logging
import time
import google.generativeai as genai
import httpx
from app.config import settings
from app.database import get_chroma_client

logger = logging.getLogger(__name__)

def get_embeddings(texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
    """
    Generates embeddings for a list of texts using the configured provider.
    - Gemini: uses gemini-embedding-001.
    - Ollama: calls /api/embeddings.
    """
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please add it to your .env file or change LLM_PROVIDER to 'ollama'."
            )
        genai.configure(api_key=settings.gemini_api_key)
        
        # Strip "models/" prefix if present in the config, and ensure correct model name
        model_name = settings.embedding_model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        # Batch requests to Gemini if list is large
        batch_size = 50
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for attempt in range(6):
                try:
                    response = genai.embed_content(
                        model=model_name,
                        content=batch,
                        task_type=task_type
                    )
                    all_embeddings.extend(response["embedding"])
                    # Small sleep to protect against free-tier rate limits
                    time.sleep(1.0)
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    if ("429" in err_str or "quota" in err_str or "exhausted" in err_str or "rate" in err_str) and attempt < 5:
                        sleep_time = (attempt + 1) * 10
                        logger.warning(f"Rate limit / Quota exceeded. Retrying batch {i//batch_size + 1} in {sleep_time}s... (Attempt {attempt+1}/6)")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Gemini embedding generation failed: {e}")
                        raise e
        return all_embeddings


    elif settings.llm_provider == "ollama":
        all_embeddings = []
        try:
            with httpx.Client(timeout=60.0) as client:
                for text in texts:
                    response = client.post(
                        f"{settings.ollama_base_url}/api/embeddings",
                        json={
                            "model": settings.embedding_model,
                            "prompt": text
                        }
                    )
                    response.raise_for_status()
                    all_embeddings.append(response.json()["embedding"])
        except httpx.ConnectError:
            raise ConnectionError(
                f"Could not connect to Ollama at {settings.ollama_base_url}. "
                "Is Ollama running?"
            )
        except Exception as e:
            logger.error(f"Ollama embedding generation failed: {e}")
            raise e
        return all_embeddings

    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

def get_llm_response(prompt: str) -> str:
    """
    Calls the LLM with the given prompt using the configured provider.
    """
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please add it to your .env file or change LLM_PROVIDER to 'ollama'."
            )
        
        for attempt in range(6):
            try:
                genai.configure(api_key=settings.gemini_api_key)
                model = genai.GenerativeModel(settings.llm_model)
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                err_str = str(e).lower()
                if ("429" in err_str or "quota" in err_str or "exhausted" in err_str or "rate" in err_str) and attempt < 5:
                    sleep_time = (attempt + 1) * 10
                    logger.warning(f"Rate limit / Quota exceeded on LLM call. Retrying in {sleep_time}s... (Attempt {attempt+1}/6)")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Gemini LLM call failed: {e}")
                    raise e


    elif settings.llm_provider == "ollama":
        try:
            with httpx.Client(timeout=90.0) as client:
                response = client.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json={
                        "model": settings.llm_model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                return response.json()["response"].strip()
        except httpx.ConnectError:
            raise ConnectionError(
                f"Could not connect to Ollama at {settings.ollama_base_url}. "
                "Is Ollama running?"
            )
        except Exception as e:
            logger.error(f"Ollama LLM call failed: {e}")
            raise e
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

def process_query_for_search(query: str) -> list[str]:
    """
    Translates Czech query to English.
    If the query requires comparison or synthesis across different eras (e.g. Victorian vs Edwardian),
    splits it into sub-queries.
    """
    prompt = (
        "You are an assistant that processes user search queries for a RAG system.\n"
        f"The user query is: '{query}'\n\n"
        "Instructions:\n"
        "1. Translate the query to English.\n"
        "2. If the query asks to compare or synthesize information from multiple periods "
        "(e.g., Victorian, Edwardian, Tudor) or different topics, split it into 2 or 3 distinct English search queries "
        "separated by semicolons ';'.\n"
        "3. If it is a simple factual query, just output the English translation.\n"
        "4. Return ONLY the English translation or the semicolon-separated sub-queries. Do not include any explanation or introduction."
    )
    
    try:
        response_text = get_llm_response(prompt)
        sub_queries = [q.strip() for q in response_text.split(";") if q.strip()]
        if not sub_queries:
            sub_queries = [query]
        logger.info(f"Processed query '{query}' into search queries: {sub_queries}")
        return sub_queries
    except Exception as e:
        logger.warning(f"Failed to expand query via LLM: {e}. Falling back to original query.")
        return [query]

def retrieve_context(query: str, top_k: int = 3) -> tuple[str, list[dict]]:
    """
    Performs query translation/expansion, searches ChromaDB, merges results,
    and returns the context string and the source metadata.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name="documentary_transcript",
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        raise ValueError("ChromaDB collection is empty. Please run ingestion first.")

    # Get search queries (English translation + sub-queries if comparison)
    search_queries = process_query_for_search(query)

    # Fetch embeddings for all search queries
    try:
        query_embeddings = get_embeddings(search_queries, task_type="retrieval_query")
    except Exception as e:
        logger.error(f"Failed to generate query embeddings: {e}")
        raise e

    # Execute searches and collect results
    all_results = []
    seen_ids = set()

    for q_emb in query_embeddings:
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k
        )
        
        # ChromaDB query returns lists of lists
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for idx in range(len(docs)):
                doc_id = ids[idx]
                # Cosine similarity = 1 - cosine_distance
                similarity = 1.0 - distances[idx]
                
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_results.append({
                        "id": doc_id,
                        "document": docs[idx],
                        "metadata": metas[idx],
                        "similarity": similarity
                    })

    # Sort all merged results by similarity descending
    all_results.sort(key=lambda x: x["similarity"], reverse=True)

    # Keep only the top_k overall chunks
    top_results = all_results[:top_k]

    # Build context string and extract sources
    context_parts = []
    sources = []

    for res in top_results:
        meta = res["metadata"]
        timestamp = meta.get("timestamp", "00:00:00")
        excerpt = meta.get("excerpt", res["document"])
        
        # Add to context
        context_parts.append(f"[{timestamp}] {res['document']}")
        
        # Add to source references
        sources.append({
            "timestamp": timestamp,
            "excerpt": excerpt,
            "similarity": res["similarity"]
        })

    context_str = "\n\n".join(context_parts)
    return context_str, sources

def answer_question(query: str) -> tuple[str, list[dict]]:
    """
    Executes the full RAG pipeline:
    1. Retrieves relevant context chunks.
    2. Constructs the prompt using the required structure.
    3. Calls the LLM to get the final answer.
    4. Handles the out-of-scope fallback (clears sources if answer is not in material).
    """
    context_str, sources = retrieve_context(query, top_k=3)

    # Build prompt as specified in Step 3
    prompt = (
        "Context information is below.\n"
        "---------------------\n"
        f"{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        f"Query: {query}\n\n"
        "Strict Rules:\n"
        "1. Answer the query in natural language based SOLELY on the context provided.\n"
        "2. If the context does not contain the answer, reply exactly with: "
        "\"I am sorry, but the provided material does not contain an answer to this question.\" "
        "Do not attempt to fabricate an answer."
    )

    logger.info(f"Sending prompt to LLM...")
    answer = get_llm_response(prompt)

    # Check if LLM gave the exact out-of-scope response
    fallback_msg = "I am sorry, but the provided material does not contain an answer to this question."
    if fallback_msg.lower() in answer.lower():
        # Clean answer to match exactly
        answer = fallback_msg
        sources = []  # Clear sources for out-of-scope queries

    return answer, sources
