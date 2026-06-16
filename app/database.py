import os
import re
import logging
import chromadb
from app.config import settings

logger = logging.getLogger(__name__)

def parse_transcript(file_path: str) -> list[dict]:
    """
    Parses the transcript file.
    Expects timestamps on their own lines (e.g. HH:MM:SS or MM:SS),
    followed by the transcript text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcript file not found at {file_path}")

    chunks = []
    current_timestamp = "00:00:00"
    current_text = []

    # Regex to match timestamps like 00:00:05 or 12:34
    timestamp_pattern = re.compile(r'^\s*(?:\d{1,2}:)?\d{2}:\d{2}\s*$')

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue

            if timestamp_pattern.match(line_str):
                # Save previous chunk if it has text
                if current_text:
                    chunks.append({
                        "timestamp": current_timestamp,
                        "text": " ".join(current_text)
                    })
                current_timestamp = line_str
                current_text = []
            else:
                current_text.append(line_str)

        # Save the last chunk
        if current_text:
            chunks.append({
                "timestamp": current_timestamp,
                "text": " ".join(current_text)
            })

    return chunks

def get_chroma_client():
    """Returns a persistent ChromaDB client."""
    os.makedirs(settings.chroma_db_dir, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_db_dir)

def ingest_data(client, force: bool = False):
    """
    Checks if ChromaDB contains the correct amount of transcript chunks.
    If not, parses the transcript, generates embeddings, and stores them in ChromaDB.
    Clears collection on failures to prevent partial/corrupt states.
    """
    from app.services import get_embeddings  # import here to avoid circular imports

    transcript_path = os.path.join("data", "transcript.txt")
    try:
        chunks = parse_transcript(transcript_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise e

    if not chunks:
        logger.warning("No chunks parsed from transcript. Ingestion aborted.")
        return

    collection_name = "documentary_transcript"

    # Check existing collection
    try:
        collection = client.get_collection(name=collection_name)
        doc_count = collection.count()
    except Exception:
        # Collection does not exist
        collection = None
        doc_count = 0

    if doc_count == len(chunks) and not force:
        logger.info(f"Database already correctly populated with {doc_count} documents. Skipping ingestion.")
        return

    logger.info("Database empty, incomplete, or force-reingest triggered. Preparing ingestion...")

    # Recreate the collection
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # If it didn't exist

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    logger.info(f"Parsed {len(chunks)} chunks from transcript. Generating embeddings...")

    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"timestamp": c["timestamp"], "excerpt": c["text"]} for c in chunks]

    try:
        embeddings = get_embeddings(texts, task_type="retrieval_document")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        # Delete incomplete collection so we don't leave it corrupted
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
        raise e

    try:
        # Add in batches of 100 to avoid ChromaDB batch limits
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]

            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
    except Exception as e:
        logger.error(f"Failed to populate ChromaDB collection: {e}")
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
        raise e

    logger.info(f"Successfully ingested {len(chunks)} chunks into ChromaDB.")
