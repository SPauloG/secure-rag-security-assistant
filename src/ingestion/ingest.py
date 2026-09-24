"""
Ingestion pipeline: reads data/corpus/, splits it into chunks, embeds them and
upserts them into Pinecone.

Run with: python -m src.ingestion.ingest
"""
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from src import config

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "corpus"

# ~600 tokens per chunk.
MAX_CHUNK_CHARS = 2500
OVERLAP_CHARS = 200

# Left on headings by the HTML loader so chunking can split on them.
HEADING_PREFIX = "## "

# Well inside Voyage's per-request limits (1,000 texts, 120K tokens).
EMBED_BATCH_MAX_TEXTS = 128
EMBED_BATCH_MAX_CHARS = 200_000

# Running headers/footers: lines checked at each page edge, and the share of pages
# a line must appear on to count as boilerplate.
PDF_EDGE_LINES = 3
PDF_BOILERPLATE_MIN_SHARE = 0.5

# Citation titles; the PDFs' own metadata is inconsistent. HTML uses its <h1>.
PDF_TITLES = {
    "owasp-top10-llm-2025.pdf": "OWASP Top 10 for LLM Applications 2025",
    "nist-csf-2.0.pdf": "NIST Cybersecurity Framework 2.0",
}


def _boilerplate_key(line: str) -> str:
    """Header/footer key without page numbers; a bare page number becomes "#"."""
    return re.sub(r"\d+", "", line.strip()) or "#"


def _repeated_edge_lines(pages: list[list[str]]) -> set[str]:
    """Lines that recur at the top or bottom of most pages (edges only, never body)."""
    seen = Counter()
    for lines in pages:
        # blank lines must not push the page number out of the edge window
        content = [line for line in lines if line.strip()]
        edges = content[:PDF_EDGE_LINES] + content[-PDF_EDGE_LINES:]
        seen.update({_boilerplate_key(line) for line in edges})
    threshold = max(2, len(pages) * PDF_BOILERPLATE_MIN_SHARE)
    return {key for key, count in seen.items() if count >= threshold}


def _strip_edges(lines: list[str], boilerplate: set[str]) -> list[str]:
    """Drops up to PDF_EDGE_LINES boilerplate lines from each end of the page."""
    def drop_leading(seq: list[str]) -> list[str]:
        dropped = index = 0
        while index < len(seq) and dropped < PDF_EDGE_LINES:
            if not seq[index].strip():  # blanks are free to skip
                index += 1
            elif _boilerplate_key(seq[index]) in boilerplate:
                index += 1
                dropped += 1
            else:
                break
        return seq[index:]

    return drop_leading(drop_leading(lines)[::-1])[::-1]


def _load_pdf(path: Path) -> list[dict]:
    """One entry per page: keeping `page` allows citing 'title, p. N'."""
    title = PDF_TITLES.get(path.name, path.stem)
    pages = [(page.extract_text() or "").split("\n") for page in PdfReader(path).pages]
    boilerplate = _repeated_edge_lines(pages)

    docs = []
    for number, lines in enumerate(pages, start=1):
        text = "\n".join(_strip_edges(lines, boilerplate)).strip()
        if text:  # empty pages (image-only cover etc.) are useless for search
            docs.append({"text": text, "source": path.name, "title": title, "page": number})
    return docs


def _load_html(path: Path) -> list[dict]:
    """One entry per file. Only the <article>: the rest of the OWASP page is navigation menu."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    article = soup.find("article")
    if article is None:
        return []
    h1 = article.find("h1")
    title = h1.get_text(strip=True) if h1 else path.stem
    if h1:
        h1.decompose()  # the title lives in the metadata; otherwise it becomes a chunk alone
    # mark headings here, split in chunk_documents()
    for heading in article.find_all(["h2", "h3"]):
        text = heading.get_text(strip=True)
        heading.clear()
        heading.append(f"{HEADING_PREFIX}{text}")
    return [{"text": article.get_text("\n", strip=True), "source": path.name, "title": title, "page": None}]


def load_documents() -> list[dict]:
    """Returns [{"text", "source", "title", "page" (None for HTML)}]; splitting is chunk_documents()'s job."""
    loaders = {".pdf": _load_pdf, ".html": _load_html}
    documents = []
    for path in sorted(CORPUS_DIR.iterdir()):
        loader = loaders.get(path.suffix.lower())
        if loader:
            documents.extend(loader(path))
    return documents


def _slug(value: str) -> str:
    """Lowercase, ASCII-safe fragment for chunk ids."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "section"


def _split_by_heading(text: str) -> list[tuple[str | None, str]]:
    """
    Splits HTML text on the marked headings into (section, text). The heading stays in
    the chunk text, since it is part of what the chunk is about.
    """
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in text.split("\n"):
        if line.startswith(HEADING_PREFIX):
            heading = line[len(HEADING_PREFIX):].strip()
            sections.append((heading, [heading]))
        else:
            sections[-1][1].append(line)
    return [(name, "\n".join(lines).strip()) for name, lines in sections if "".join(lines).strip()]


def _tail_lines(lines: list[str], overlap: int) -> list[str]:
    """Last whole lines fitting in `overlap` chars, so a chunk never starts mid-word."""
    tail: list[str] = []
    size = 0
    for line in reversed(lines):
        if size + len(line) > overlap:
            break
        tail.insert(0, line)
        size += len(line) + 1
    return tail


def _split_oversized(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Splits on line boundaries, with overlap so a cut sentence stays whole in one chunk."""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        while len(line) > max_chars:  # single very long line (e.g. a PDF table row)
            if current:
                parts.append("\n".join(current))
                current, size = [], 0
            parts.append(line[:max_chars])
            line = line[max_chars:]
        if current and size + len(line) + 1 > max_chars:
            parts.append("\n".join(current))
            current = _tail_lines(current, overlap)
            size = sum(len(l) + 1 for l in current)
        current.append(line)
        size += len(line) + 1
    if "".join(current).strip():
        parts.append("\n".join(current))
    return parts


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    HTML splits on <h2>/<h3>, PDF keeps one chunk per page; anything above
    MAX_CHUNK_CHARS is split further.
    """
    chunks: list[dict] = []
    counters: dict[tuple, int] = {}
    for doc in documents:
        pieces = _split_by_heading(doc["text"]) if doc["page"] is None else [(None, doc["text"])]
        for section, body in pieces:
            for part in _split_oversized(body):
                key = (doc["source"], doc["page"], section)
                index = counters[key] = counters.get(key, -1) + 1
                locator = f"p{doc['page']}" if doc["page"] else _slug(section or "body")
                chunks.append({
                    "id": f"{doc['source']}#{locator}#{index}",
                    "text": part,
                    "source": doc["source"],
                    "title": doc["title"],
                    "page": doc["page"],
                    "section": section,
                })
    return chunks


def _vector_metadata(chunk: dict) -> dict:
    """Stored with the vector, text included. Pinecone rejects nulls, so they are dropped."""
    fields = {
        "text": chunk["text"],
        "source": chunk["source"],
        "title": chunk["title"],
        "page": chunk["page"],
        "section": chunk["section"],
    }
    return {key: value for key, value in fields.items() if value is not None}


def _embedding_batches(chunks: list[dict]) -> list[list[dict]]:
    """Groups chunks into requests within Voyage's limits (budgeted in characters)."""
    batches: list[list[dict]] = []
    current: list[dict] = []
    chars = 0
    for chunk in chunks:
        too_many = len(current) >= EMBED_BATCH_MAX_TEXTS
        too_large = chars + len(chunk["text"]) > EMBED_BATCH_MAX_CHARS
        if current and (too_many or too_large):
            batches.append(current)
            current, chars = [], 0
        current.append(chunk)
        chars += len(chunk["text"])
    if current:
        batches.append(current)
    return batches


def _pinecone_index():
    """Returns the index handle, creating the index on first run."""
    from pinecone import Pinecone, ServerlessSpec

    pinecone = Pinecone(api_key=config.PINECONE_API_KEY)
    if not pinecone.has_index(config.PINECONE_INDEX_NAME):
        print(f"Creating Pinecone index {config.PINECONE_INDEX_NAME!r} "
              f"(dimension {config.EMBEDDING_DIMENSION}, metric {config.PINECONE_METRIC})...")
        pinecone.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIMENSION,
            metric=config.PINECONE_METRIC,
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
    return pinecone.Index(config.PINECONE_INDEX_NAME)


def embed_and_upsert(chunks: list[dict]) -> None:
    """
    Embeds with input_type="document" (retrieval uses "query") and upserts. Ids are
    deterministic, so re-running overwrites instead of duplicating.
    """
    import voyageai

    voyage = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    index = _pinecone_index()

    batches = _embedding_batches(chunks)
    upserted = 0
    for number, batch in enumerate(batches, start=1):
        embeddings = voyage.embed(
            [chunk["text"] for chunk in batch],
            model=config.EMBEDDING_MODEL,
            input_type="document",
            output_dimension=config.EMBEDDING_DIMENSION,
        ).embeddings
        index.upsert(vectors=[
            {"id": chunk["id"], "values": vector, "metadata": _vector_metadata(chunk)}
            for chunk, vector in zip(batch, embeddings)
        ])
        upserted += len(batch)
        print(f"  batch {number}/{len(batches)}: {upserted}/{len(chunks)} chunks upserted")


def main() -> None:
    config.validate()
    documents = load_documents()
    print(f"Loaded {len(documents)} documents from {CORPUS_DIR}")
    chunks = chunk_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    embed_and_upsert(chunks)
    print(f"Ingestion complete: {len(chunks)} chunks indexed in {config.PINECONE_INDEX_NAME!r}.")


if __name__ == "__main__":
    main()
