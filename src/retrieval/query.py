"""
Retrieval + generation: fetches the most relevant chunks and asks Claude for an
answer that cites them.

Run with: python -m src.retrieval.query "your question here"
"""
import sys

from src import config


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """
    Returns the top_k most similar chunks, best first, each with its cosine `score`.

    input_type="query" (not "document"): mixing them up silently degrades matches.
    """
    import voyageai

    from src.ingestion.ingest import _pinecone_index

    voyage = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    vector = voyage.embed(
        [question],
        model=config.EMBEDDING_MODEL,
        input_type="query",
        output_dimension=config.EMBEDDING_DIMENSION,
    ).embeddings[0]

    response = _pinecone_index().query(vector=vector, top_k=top_k, include_metadata=True)
    return [
        {
            "id": match["id"],
            "score": match["score"],
            "text": match["metadata"]["text"],
            "source": match["metadata"]["source"],
            "title": match["metadata"]["title"],
            # absent on the format that does not have them (see _vector_metadata)
            "page": int(match["metadata"]["page"]) if "page" in match["metadata"] else None,
            "section": match["metadata"].get("section"),
        }
        for match in response["matches"]
    ]


def format_citation(chunk: dict) -> str:
    """HTML chunks cite the section, PDF chunks the page."""
    if chunk.get("section"):
        return f"{chunk['title']} › {chunk['section']}"
    if chunk.get("page"):
        return f"{chunk['title']}, p. {chunk['page']}"
    return chunk["title"]


NO_ANSWER = (
    "I could not find anything about that in the indexed documents "
    "(OWASP Top 10, OWASP Top 10 for LLM Applications, NIST CSF 2.0)."
)

SYSTEM_PROMPT = """You are a security assistant answering from a corpus of security \
standards: the OWASP Top 10, the OWASP Top 10 for LLM Applications, and the NIST \
Cybersecurity Framework 2.0.

Answer only from the attached documents. If they do not contain the answer, say so \
plainly rather than filling the gap from general knowledge — an unsupported answer is \
worse than no answer here.

Be concise and concrete. Prefer the specific control, requirement or practice the \
documents state over a general summary of the topic."""


def _documents_for(chunks: list[dict]) -> list[dict]:
    """
    One document block per chunk, with API citations enabled, so Claude cannot
    fabricate a reference. A block's position is the `document_index` cited back.
    """
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": chunk["text"]},
            "title": format_citation(chunk),
            "citations": {"enabled": True},
        }
        for chunk in chunks
    ]


TRUNCATED_NOTICE = (
    "[This answer was cut off at the token limit and is incomplete. "
    "Raise GENERATION_MAX_TOKENS in src/config.py.]"
)


def _render(blocks, chunks: list[dict], truncated: bool = False) -> str:
    """Text with [n] markers built from the API's citation data, plus a sources list."""
    answer: list[str] = []
    order: list[int] = []  # document_index of each source, in order of first citation

    for block in blocks:
        if block.type != "text":
            continue
        answer.append(block.text)
        markers = []
        for citation in getattr(block, "citations", None) or []:
            index = citation.document_index
            if index not in order:
                order.append(index)
            marker = order.index(index) + 1
            if marker not in markers:
                markers.append(marker)
        if markers:
            answer.append("".join(f"[{m}]" for m in markers))

    text = "".join(answer).strip()
    if truncated:
        # a cut-off answer must not look complete
        text = f"{text}\n\n{TRUNCATED_NOTICE}"
    if not order:
        return text

    sources = "\n".join(
        f"[{position}] {format_citation(chunks[index])}"
        for position, index in enumerate(order, start=1)
    )
    return f"{text}\n\nSources:\n{sources}"


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Answers from the retrieved passages, citing them."""
    import anthropic

    if not chunks:
        return NO_ANSWER

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=config.GENERATION_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [*_documents_for(chunks), {"type": "text", "text": question}],
        }],
    )
    return _render(response.content, chunks, truncated=response.stop_reason == "max_tokens")


def relevant_chunks(question: str, top_k: int = 5) -> list[dict]:
    """Retrieved passages the corpus is confident enough about to show the model."""
    return [c for c in retrieve(question, top_k=top_k) if c["score"] >= config.MIN_RETRIEVAL_SCORE]


def ask(question: str) -> str:
    config.validate()
    return generate_answer(question, relevant_chunks(question))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.retrieval.query \"your question\"")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    print(ask(question))
