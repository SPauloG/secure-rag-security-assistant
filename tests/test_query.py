"""Covers the pure parts of retrieval and answer rendering; the API calls themselves
are exercised by scripts/search.py and src.retrieval.query."""
from types import SimpleNamespace

import pytest

from src import config
from src.retrieval import query as q
from src.retrieval.query import NO_ANSWER, format_citation


def block(text: str, *document_indexes: int):
    """A response text block, optionally carrying citations to document N."""
    citations = [SimpleNamespace(document_index=i) for i in document_indexes]
    return SimpleNamespace(type="text", text=text, citations=citations or None)


def chunk(title: str, **overrides) -> dict:
    return {"title": title, "section": None, "page": 1, "score": 0.9} | overrides


def test_html_chunks_are_cited_by_section():
    chunk = {"title": "A01:2021 – Broken Access Control", "section": "How to Prevent", "page": None}
    assert format_citation(chunk) == "A01:2021 – Broken Access Control › How to Prevent"


def test_pdf_chunks_are_cited_by_page():
    chunk = {"title": "NIST Cybersecurity Framework 2.0", "section": None, "page": 25}
    assert format_citation(chunk) == "NIST Cybersecurity Framework 2.0, p. 25"


def test_a_chunk_with_no_locator_falls_back_to_the_title():
    assert format_citation({"title": "Some Document", "section": None, "page": None}) == "Some Document"


def test_markers_and_sources_come_from_the_api_citations():
    chunks = [chunk("Doc A", page=9), chunk("Doc B", page=25)]
    rendered = q._render([block("Use least privilege", 0), block(" and log access", 1)], chunks)

    assert "Use least privilege[1] and log access[2]" in rendered
    assert rendered.endswith("Sources:\n[1] Doc A, p. 9\n[2] Doc B, p. 25")


def test_the_same_source_cited_twice_keeps_one_number():
    chunks = [chunk("Doc A", page=9), chunk("Doc B", page=25)]
    rendered = q._render([block("first", 1), block(" second", 1)], chunks)

    assert "first[1] second[1]" in rendered
    assert rendered.count("[1] Doc B, p. 25") == 1
    assert "[2]" not in rendered


def test_source_numbering_follows_first_use_not_retrieval_rank():
    """Sources are numbered as they appear in the answer, so [1] is the first one read."""
    chunks = [chunk("Doc A", page=1), chunk("Doc B", page=2), chunk("Doc C", page=3)]
    rendered = q._render([block("x", 2), block("y", 0)], chunks)

    assert rendered.endswith("Sources:\n[1] Doc C, p. 3\n[2] Doc A, p. 1")


def test_a_truncated_answer_says_so_instead_of_looking_complete():
    rendered = q._render([block("the answer stops mid-sen", 0)], [chunk("Doc A")], truncated=True)

    assert q.TRUNCATED_NOTICE in rendered
    assert rendered.index(q.TRUNCATED_NOTICE) < rendered.index("Sources:")


def test_a_complete_answer_carries_no_truncation_notice():
    assert q.TRUNCATED_NOTICE not in q._render([block("done", 0)], [chunk("Doc A")])


def test_an_uncited_answer_has_no_sources_section():
    rendered = q._render([block("I could not find that.")], [chunk("Doc A")])

    assert rendered == "I could not find that."
    assert "Sources:" not in rendered


def test_no_chunks_means_no_api_call(monkeypatch):
    monkeypatch.setattr(
        "anthropic.Anthropic", lambda **_: pytest.fail("the API must not be called")
    )
    assert q.generate_answer("anything", []) == NO_ANSWER


def test_low_scoring_matches_are_dropped_before_generation(monkeypatch):
    """An off-topic question still returns nearest neighbours; the threshold is what
    stops them from being handed to the model as if they were answers."""
    monkeypatch.setattr(config, "validate", lambda: None)
    monkeypatch.setattr(q, "retrieve", lambda question, **kw: [
        chunk("Relevant", score=config.MIN_RETRIEVAL_SCORE + 0.1),
        chunk("Noise", score=config.MIN_RETRIEVAL_SCORE - 0.1),
    ])
    passed = {}
    monkeypatch.setattr(q, "generate_answer", lambda question, chunks: passed.setdefault("chunks", chunks))

    q.ask("a question")

    assert [c["title"] for c in passed["chunks"]] == ["Relevant"]


def test_an_entirely_off_topic_question_never_reaches_the_model(monkeypatch):
    monkeypatch.setattr(config, "validate", lambda: None)
    monkeypatch.setattr(q, "retrieve", lambda question, **kw: [chunk("Pizza", score=0.19)])
    monkeypatch.setattr(
        "anthropic.Anthropic", lambda **_: pytest.fail("the API must not be called")
    )

    assert q.ask("how do I make pizza?") == NO_ANSWER
