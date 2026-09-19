"""
Retrieval + generation: takes a question, fetches the most relevant chunks
from Pinecone, builds the context and asks Claude for an answer that cites its sources.

Run with: python -m src.retrieval.query "your question here"

DESIGN DECISIONS (document in the README when made):
  - How many chunks to retrieve (top_k) and how to build the context prompt.
  - How to format the source citation in the answer (required by the
    project's definition of done).
  - What to do when retrieval finds nothing relevant (not inventing an
    answer is what separates a well-built RAG from hallucination).
"""
import sys

from src import config


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """
    TODO: embed the question (Voyage AI) and fetch the top_k most similar
    chunks from the Pinecone index.
    """
    raise NotImplementedError


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    TODO: build the prompt with the retrieved chunks as context and call
    the Claude API, asking for an answer that cites its sources (source/title
    of each chunk used).
    """
    raise NotImplementedError


def ask(question: str) -> str:
    config.validate()
    chunks = retrieve(question)
    return generate_answer(question, chunks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.retrieval.query \"your question\"")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    print(ask(question))
