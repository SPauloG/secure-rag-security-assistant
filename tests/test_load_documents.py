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


def test_pdf_pages_drop_the_running_header():
    """The header repeats on every page, so it adds no signal to the embedding and
    would show up in any passage quoted back to the user."""
    docs = load_documents()
    llm = [d for d in docs if d["source"] == "owasp-top10-llm-2025.pdf"]
    nist = [d for d in docs if d["source"] == "nist-csf-2.0.pdf"]

    assert not any(d["text"].startswith("OWASP Top 10 for LLM Applications v") for d in llm)
    assert not any("genai.owasp.org" in d["text"].split("\n")[0] for d in llm)
    assert not any(d["text"].startswith("NIST CSWP 29") for d in nist)


def test_boilerplate_detection_ignores_the_page_number_in_a_header():
    from src.ingestion.ingest import _boilerplate_key, _repeated_edge_lines

    # the page number moves around: glued to the header, then on its own line
    pages = [
        [f"{n}handbook.example", "", f"{n}", f"body of page {n}", "unique tail"]
        for n in range(1, 11)
    ]
    pages.append(["handbook.example", "", "11", "body of page 11", "unique tail"])
    boilerplate = _repeated_edge_lines(pages)

    assert _boilerplate_key("7handbook.example") in boilerplate  # header, numbered or not
    assert "#" in boilerplate  # the bare page-number line


def test_repeated_body_text_is_never_treated_as_boilerplate():
    """Only page edges are inspected, so a phrase repeated mid-page survives."""
    from src.ingestion.ingest import _repeated_edge_lines

    pages = [
        ["Header", "a", "b", "SEE THE APPENDIX", "c", "d", f"tail {n}"]
        for n in range(10)
    ]
    boilerplate = _repeated_edge_lines(pages)

    assert "Header" in boilerplate
    assert "SEE THE APPENDIX" not in boilerplate


def test_html_excludes_site_navigation():
    docs = load_documents()
    a01 = next(d for d in docs if d["source"] == "owasp-top10-2021-a01.html")
    assert a01["title"].startswith("A01:2021")
    assert "Welcome Page" not in a01["text"]  # side-menu item
