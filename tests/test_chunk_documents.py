import pytest

from src.ingestion.ingest import (
    MAX_CHUNK_CHARS,
    CORPUS_DIR,
    chunk_documents,
    load_documents,
)

pytestmark = pytest.mark.skipif(
    not any(CORPUS_DIR.glob("*.pdf")), reason="corpus not downloaded to data/corpus/"
)


@pytest.fixture(scope="module")
def chunks():
    return chunk_documents(load_documents())


def test_no_chunk_exceeds_the_size_limit(chunks):
    assert chunks
    assert max(len(c["text"]) for c in chunks) <= MAX_CHUNK_CHARS


def test_ids_are_unique_so_reingesting_overwrites_instead_of_duplicating(chunks):
    ids = [c["id"] for c in chunks]
    assert len(set(ids)) == len(ids)


def test_every_chunk_can_be_cited(chunks):
    for c in chunks:
        assert c["text"].strip()
        assert c["source"] and c["title"]
        assert c["page"] is not None or c["section"] is not None


def test_html_is_split_by_section_and_keeps_the_heading_in_the_text(chunks):
    a01 = [c for c in chunks if c["source"] == "owasp-top10-2021-a01.html"]
    sections = {c["section"] for c in a01}
    assert {"Description", "How to Prevent", "Example Attack Scenarios"} <= sections

    prevent = next(c for c in a01 if c["section"] == "How to Prevent")
    assert prevent["text"].startswith("How to Prevent")


def test_pdf_chunks_keep_their_page_number(chunks):
    pdf = [c for c in chunks if c["source"].endswith(".pdf")]
    assert pdf
    assert all(isinstance(c["page"], int) and c["page"] >= 1 for c in pdf)


def test_oversized_text_is_split_with_overlapping_lines():
    from src.ingestion.ingest import _split_oversized

    text = "\n".join(f"line {i} " + "x" * 60 for i in range(100))
    parts = _split_oversized(text, max_chars=500, overlap=120)

    assert len(parts) > 1
    assert all(len(p) <= 500 for p in parts)
    # the tail of one chunk reappears at the head of the next
    assert parts[0].split("\n")[-1] in parts[1]
