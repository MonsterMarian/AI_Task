# Architecture & Design Document (RAG Backend)

This document describes the design decisions, retrieval strategy, and implementation details for the Documentary Q&A Backend.

---

## 1. Chunking Strategy

### Decision
We chunk the transcript using the **timestamp lines** (e.g. `00:10:33`) as boundary delimiters. Each chunk consists of a timestamp and all the text lines that follow it until the next timestamp line is reached.

### Rationale
- **Natural Boundaries**: The transcript is pre-divided into segments corresponding to approximately 1 minute of speech (~100–150 words). This is an ideal chunk size for embedding models, preserving semantic coherence without being too large or too small.
- **Exact Metadata Mapping**: Since each chunk is mapped directly to a single timestamp, we can retrieve the exact timecode of where an event occurred in the video. This guarantees 100% accurate source references without having to guess or post-process character offsets.
- **Sentence Preservation**: Timestamp boundaries in the source transcript are positioned between sentences, which prevents cutting a critical sentence in half.

---

## 2. Retrieval Strategy

### Library & Embeddings
- We use **ChromaDB** as our vector database with `cosine` similarity (configured via `metadata={"hnsw:space": "cosine"}`).
- Embeddings are generated dynamically using the configured provider:
  - **Gemini**: Uses the `gemini-embedding-001` model via the `google-genai` SDK.
  - **Ollama**: Uses `nomic-embed-text` (or any other locally configured model).

### Handling Czech-to-English Retrieval (Trap #1)
Because the transcript is in English, but the test queries are in Czech (e.g., *"Jak dopadl pokus s boraxem v mléku?"*), a direct cross-lingual vector search can fail to find precise keyword matches.
- **Solution**: Before querying the vector database, we send the Czech query to the LLM to get an accurate **English translation/expansion**. We then execute the embedding search in ChromaDB using this English translation.

### Handling Cross-Era Synthesis (Trap #2)
Synthesis queries (e.g. comparing Victorian and Edwardian household dangers) require retrieving context from different parts of the document. If we do a single query, vector search will cluster chunks around one era or one keyword, missing the comparison context.
- **Solution**: We implement **Query Deconstruction**. The LLM detects comparison queries and splits them into semicolon-separated sub-queries (e.g. *"Victorian household dangers"* and *"Edwardian household dangers"*). We perform vector retrieval for each sub-query, merge the results, deduplicate by chunk ID, and select the overall top 3 chunks to construct the final LLM context.

---

## 3. Prompt Engineering

We construct the prompt sent to the LLM using the following structure:

```text
Context information is below.
---------------------
[00:10:33] Of a product called borax, An alkali...
[00:11:30] but it gives a pH closer to neutral...
---------------------
Given the context information and not prior knowledge, answer the query.
Query: Jak dopadl pokus s boraxem v mléku?

Strict Rules:
1. Answer the query in natural language based SOLELY on the context provided.
2. If the context does not contain the answer, reply exactly with: "I am sorry, but the provided material does not contain an answer to this question." Do not attempt to fabricate an answer.
```

- **Original Query Preservation**: We pass the *original Czech query* to the LLM alongside the *English context*. Multilingual LLMs (like Gemini 1.5 or Llama 3) can read English context and naturally respond in fluent Czech, while strictly following the English system rules.
- **Out-of-Scope Control**: If the LLM returns the exact fallback message, our service detects this, cleans the output, and clears the `sources` array to return an empty list (`[]`), avoiding false references.

---

## 4. Model Cascade & Latency

To handle API quota limits and transient server-side unavailability (HTTP 429 / 503), the service implements an automatic **model cascade**: it tries the primary model (`gemini-2.5-flash`) first, then falls back through an ordered list of alternative Gemini models until one succeeds. This makes the service resilient to rate-limits at the cost of occasional latency spikes (typically an extra 5–15 s per fallback step). Under normal load, end-to-end response time is 6–16 seconds.

---

## 5. Future Improvements

Given more time, we would implement:
1. **Hybrid Search**: Combine dense vector embeddings with sparse keyword search (BM25) to improve retrieval of exact numbers, chemicals, and rare names (e.g. *Lucy Dean*, *Thomas Crapper*).
2. **Sliding Window Merging**: When a retrieved chunk is selected, also pull its adjacent chunk (+/- 1 minute) to provide the LLM with surrounding context, ensuring smooth transitions.
3. **Metadata Filtering**: Add filters based on parsed eras (Victorian, Edwardian, Tudor) to quickly restrict or bias search space when the query mentions an era.
4. **Streaming responses**: Stream tokens back to the client for a more responsive UX, while still assembling the final JSON for validation.
