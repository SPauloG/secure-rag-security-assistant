"""
Loads the project configuration from .env.
No key/secret should be hardcoded in any other file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "secure-rag-owasp-nist")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# Embedding model: decision to be documented in the README once finalized.
EMBEDDING_MODEL = "voyage-3"  # embeddings via Voyage AI (Anthropic's recommended partner for Claude users)

def validate() -> None:
    missing = [
        name for name, val in {
            "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
            "VOYAGE_API_KEY": VOYAGE_API_KEY,
            "PINECONE_API_KEY": PINECONE_API_KEY,
        }.items() if not val
    ]
    if missing:
        raise RuntimeError(
            f"Missing variables in .env: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill them in."
        )
