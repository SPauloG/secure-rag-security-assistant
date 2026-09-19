"""
Ingestion pipeline: takes the raw documents in data/corpus/,
splits them into chunks, generates embeddings and loads them into Pinecone.

Run with: python -m src.ingestion.ingest

DESIGN DECISIONS (document in the README when made):
  - Chunking strategy: fixed size? by section/heading? how much overlap?
    OWASP and NIST have a heading/section structure — worth using it instead
    of cutting blindly by character count.
  - What goes in each chunk's metadata (source, section title, URL) —
    it is what allows "answering with a cited source" (a project requirement).
"""
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from src import config

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "corpus"

# Human-readable PDF titles (the metadata embedded in the PDFs is inconsistent).
# This is what shows up in the answer's citation. HTML files take the title from their own <h1>.
PDF_TITLES = {
    "owasp-top10-llm-2025.pdf": "OWASP Top 10 for LLM Applications 2025",
    "nist-csf-2.0.pdf": "NIST Cybersecurity Framework 2.0",
}


def _load_pdf(path: Path) -> list[dict]:
    """One entry per page: keeping `page` allows citing 'title, p. N'."""
    title = PDF_TITLES.get(path.name, path.stem)
    docs = []
    for number, page in enumerate(PdfReader(path).pages, start=1):
        text = (page.extract_text() or "").strip()
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
    # the "\n" separator keeps section headings on their own lines, which chunking will use
    return [{"text": article.get_text("\n", strip=True), "source": path.name, "title": title, "page": None}]


def load_documents() -> list[dict]:
    """
    Reads the files in CORPUS_DIR and returns a list of dicts:
    {"text", "source" (file name), "title", "page" (PDF page number; None for HTML)}.

    It only reads and preserves the location; splitting by section is
    chunk_documents()'s responsibility.
    """
    loaders = {".pdf": _load_pdf, ".html": _load_html}
    documents = []
    for path in sorted(CORPUS_DIR.iterdir()):
        loader = loaders.get(path.suffix.lower())
        if loader:
            documents.extend(loader(path))
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    TODO: split each document into smaller chunks, preserving the metadata
    (source, title) in every chunk — it becomes the "source citation" in the answer.
    """
    raise NotImplementedError


def embed_and_upsert(chunks: list[dict]) -> None:
    """
    TODO: generate embeddings for the chunks (Voyage AI) and upsert them into
    the Pinecone index (config.PINECONE_INDEX_NAME), including the metadata.
    """
    raise NotImplementedError


def main() -> None:
    config.validate()
    documents = load_documents()
    chunks = chunk_documents(documents)
    embed_and_upsert(chunks)
    print(f"Ingestion complete: {len(chunks)} chunks indexed.")


if __name__ == "__main__":
    main()
