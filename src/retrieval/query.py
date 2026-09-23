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
    Embeds the question and returns the top_k most similar chunks, best first.

    input_type="query" is required here and is not the same as the "document" used
    during ingestion: Voyage embeds a question and a passage differently, and mixing
    them up degrades results silently — nothing errors, the matches are just worse.

    Each result carries its metadata plus the similarity `score` (cosine, 0-1).
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
    """
    How a retrieved passage is identified to the reader.

    HTML chunks cite the section, PDF chunks the page — the finest locator each format
    reliably provides. Both are verifiable: the reader can open the document and check.
    """
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
    Each retrieved passage becomes a document block with citations enabled.

    This is what makes a fabricated citation structurally impossible: Claude does not
    write the references, the API returns which span of which document supported each
    sentence. `title` is already the human-readable citation, and the block's position
    in this list is the `document_index` the response cites back.
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
    """
    Turns the response into text with [n] markers and a sources list.

    The numbering is derived from the API's citation data, not from anything the model
    wrote, so a marker can only point at a passage that was actually retrieved.
    """
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
        # Never let a half-finished answer look complete: it stops mid-sentence, and the
        # part that was cut may be the caveat that changed the meaning.
        text = f"{text}\n\n{TRUNCATED_NOTICE}"
    if not order:
        return text

    sources = "\n".join(
        f"[{position}] {format_citation(chunks[index])}"
        for position, index in enumerate(order, start=1)
    )
    return f"{text}\n\nSources:\n{sources}"


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Answers the question from the retrieved passages, citing them.

    Two independent guards against answering from thin air, because they catch
    different failures: the caller drops low-scoring matches ("the corpus has nothing
    on this"), and the system prompt tells Claude to refuse when the passages it did
    get do not answer the question ("close, but not an answer").
    """
    import anthropic

    if not chunks:
        return NO_ANSWER

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=config.GENERATION_MAX_TOKENS,
        output_config={"effort": config.GENERATION_EFFORT},
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
