"""
Measures the corpus and the resulting chunks.

Every number quoted in the README's "Why this chunking strategy" section comes from
this script, so the reasoning behind the strategy can be re-checked instead of trusted.

Run with: python -m scripts.analyze_corpus
"""
import re
import statistics as st
from collections import Counter

from src.ingestion.ingest import MAX_CHUNK_CHARS, chunk_documents, load_documents


def describe(label: str, sizes: list[int]) -> None:
    print(
        f"  {label:34} n={len(sizes):4}  min={min(sizes):5}  "
        f"median={int(st.median(sizes)):5}  max={max(sizes):5}"
    )


def document_sizes(documents: list[dict]) -> None:
    """Sizes of the units load_documents() produces: one PDF page, one HTML file."""
    print("Documents as loaded (chars)")
    for source in sorted({d["source"] for d in documents}):
        sizes = [len(d["text"]) for d in documents if d["source"] == source]
        if source.endswith(".html"):
            continue
        describe(f"{source} (per page)", sizes)
    html = [len(d["text"]) for d in documents if d["source"].endswith(".html")]
    describe("OWASP 2021 HTML (per file)", html)


def heading_detectability(documents: list[dict]) -> None:
    """
    Why parsing sections out of the PDFs was rejected: the headings are not
    reliably detectable in extracted PDF text.
    """
    print("\nSection headings in extracted PDF text")
    llm = "\n".join(d["text"] for d in documents if d["source"].startswith("owasp-top10-llm"))
    nist = "\n".join(d["text"] for d in documents if d["source"].startswith("nist"))

    risk_headings = re.findall(r"LLM\d{2}:\d{4}", llm)
    print(f"  'LLM0X:2025' matches: {len(risk_headings)} for {len(set(risk_headings))} actual risks")
    print("  -> the table of contents repeats every heading, so a regex cannot tell")
    print("     a real section start from its TOC entry without extra rules")

    nist_ids = re.findall(r"\b[A-Z]{2}\.[A-Z]{2}-\d{2}\b", nist)
    print(f"  NIST subcategory ids (e.g. PR.AA-01): {len(nist_ids)} occurrences")
    print("  -> these are rows of an appendix table, not section headings")


def chunk_sizes(chunks: list[dict]) -> None:
    sizes = [len(c["text"]) for c in chunks]
    print(f"\nChunks: {len(chunks)} (limit {MAX_CHUNK_CHARS} chars)")
    print(
        f"  chars   min={min(sizes)}  median={int(st.median(sizes))}  "
        f"mean={int(st.mean(sizes))}  max={max(sizes)}"
    )
    print(f"  ~tokens median={int(st.median(sizes)) // 4}  max={max(sizes) // 4}  (4 chars/token)")

    by_format = Counter("PDF (by page)" if c["page"] else "HTML (by section)" for c in chunks)
    for fmt, count in by_format.most_common():
        print(f"  {fmt:20} {count:4} chunks")

    print("\nHow a chunk gets cited")
    for chunk in (next(c for c in chunks if c["page"]), next(c for c in chunks if c["section"])):
        where = f"p. {chunk['page']}" if chunk["page"] else chunk["section"]
        print(f"  {chunk['title']} — {where}")
        print(f"    id={chunk['id']}")


def main() -> None:
    documents = load_documents()
    if not documents:
        raise SystemExit("No documents in data/corpus/ — see the README there.")
    document_sizes(documents)
    heading_detectability(documents)
    chunk_sizes(chunk_documents(documents))


if __name__ == "__main__":
    main()
