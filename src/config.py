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

# Embeddings via Voyage AI (Anthropic's recommended partner for Claude users).
# See "Why this embedding model" in the README.
EMBEDDING_MODEL = "voyage-4-large"
# Passed explicitly rather than relying on the model's default, because the Pinecone
# index is created with this exact size and a mismatch is rejected at upsert time.
EMBEDDING_DIMENSION = 1024

# Cosine: Voyage returns normalized embeddings, so cosine measures the angle between
# them, which is what semantic similarity means here.
PINECONE_METRIC = "cosine"
# The Pinecone free tier only serves serverless indexes from AWS us-east-1.
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

GENERATION_MODEL = "claude-opus-5"
# Answering from retrieved passages is not a hard reasoning task, and thinking tokens
# bill as output. Low effort keeps cost per answer near $0.02; raise it if the
# evaluation shows answer quality is the bottleneck.
GENERATION_EFFORT = "low"
# A ceiling, not a target: billing is on tokens actually produced, so this only needs
# to be high enough that a complete answer is never cut off. 1024 was too low — a
# thorough answer about prompt injection hit it and stopped mid-sentence.
GENERATION_MAX_TOKENS = 4096

# Below this cosine score the corpus is treated as having nothing on the subject, and
# no LLM call is made. PROVISIONAL: measured from two questions (relevant ones scored
# 0.563-0.596, an off-topic one 0.152-0.196). Recalibrate on the evaluation set.
MIN_RETRIEVAL_SCORE = 0.35

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
