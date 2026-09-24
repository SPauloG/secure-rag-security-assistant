"""Project configuration. Secrets come only from .env."""
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "secure-rag-owasp-nist")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL = "voyage-4-large"
# Must match the Pinecone index size.
EMBEDDING_DIMENSION = 1024

# Voyage embeddings are normalized.
PINECONE_METRIC = "cosine"
# Only region on the Pinecone free tier.
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Cheapest Claude model while in development.
GENERATION_MODEL = "claude-haiku-4-5"
# A ceiling, not a target: 1024 cut long answers off mid-sentence.
GENERATION_MAX_TOKENS = 4096

# Below this score no LLM call is made.
MIN_RETRIEVAL_SCORE = 0.35

# LLM judge for the evaluation (Gemini free tier).
JUDGE_MODEL = "gemini-3.5-flash-lite"

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
