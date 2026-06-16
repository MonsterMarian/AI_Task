import os
import tempfile
import pytest
from app.database import parse_transcript
from app.config import settings

def test_parse_transcript():
    """Tests that the transcript parser correctly parses timestamps and groups text lines."""
    content = (
        "00:01:10\n"
        "This is segment one.\n"
        "With multiple lines.\n"
        "00:02:20\n"
        "This is segment two.\n"
    )
    with tempfile.NamedTemporaryFile('w', delete=False, suffix=".txt", encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name

    try:
        chunks = parse_transcript(temp_path)
        assert len(chunks) == 2
        assert chunks[0]["timestamp"] == "00:01:10"
        assert chunks[0]["text"] == "This is segment one. With multiple lines."
        assert chunks[1]["timestamp"] == "00:02:20"
        assert chunks[1]["text"] == "This is segment two."
    finally:
        os.remove(temp_path)

def test_config_defaults():
    """Tests that default configs are loaded correctly."""
    assert settings.llm_provider in ["gemini", "ollama"]
    assert settings.chroma_db_dir is not None
