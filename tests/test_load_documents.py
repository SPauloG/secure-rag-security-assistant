import pytest

from src.ingestion.ingest import CORPUS_DIR, load_documents

pytestmark = pytest.mark.skipif(
    not any(CORPUS_DIR.glob("*.pdf")), reason="corpus not downloaded to data/corpus/"
)


def test_every_document_has_citation_metadata():
    docs = load_documents()
    assert docs
    for d in docs:
        assert d["text"].strip()
        assert d["source"] and d["title"]


def test_pdf_pages_are_numbered_and_html_has_no_page():
    docs = load_documents()
    pdf = [d for d in docs if d["source"].endswith(".pdf")]
    html = [d for d in docs if d["source"].endswith(".html")]
    assert pdf and all(isinstance(d["page"], int) and d["page"] >= 1 for d in pdf)
    assert html and all(d["page"] is None for d in html)


def test_html_excludes_site_navigation():
    docs = load_documents()
    a01 = next(d for d in docs if d["source"] == "owasp-top10-2021-a01.html")
    assert a01["title"].startswith("A01:2021")
    assert "Welcome Page" not in a01["text"]  # side-menu item
