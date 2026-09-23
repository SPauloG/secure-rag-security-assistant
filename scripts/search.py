"""
Retrieval only — no LLM call, no Anthropic credits spent.

Shows which passages the index returns for a question and how strongly each matched,
which is the fastest way to tell a retrieval problem from a generation problem.

Run with:
    python -m scripts.search "how do I prevent prompt injection?"
    python -m scripts.search            # interactive, Ctrl+C to quit
"""
import sys

from src import config
from src.retrieval.query import format_citation, retrieve

TOP_K = 5
PREVIEW_CHARS = 260


def show(question: str) -> None:
    results = retrieve(question, top_k=TOP_K)
    if not results:
        print("  no matches")
        return
    for position, chunk in enumerate(results, start=1):
        preview = " ".join(chunk["text"].split())[:PREVIEW_CHARS]
        print(f"\n  {position}. [{chunk['score']:.3f}] {format_citation(chunk)}")
        print(f"     {preview}...")


def main() -> None:
    config.validate()
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\nQ: {question}")
        show(question)
        return

    print("Ask a question about the indexed corpus. Ctrl+C to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if question:
            show(question)


if __name__ == "__main__":
    main()
