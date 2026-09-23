"""The evaluation harness grades attribution, so its own grading needs covering."""
import json

from scripts.evaluate import QUESTIONS_FILE, check
from src.retrieval.query import NO_ANSWER

ANSWERABLE = {"question": "q", "expects_answer": True, "expected_sources": ["a.pdf"], "grounds": ""}
UNANSWERABLE = {"question": "q", "expects_answer": False, "expected_sources": [], "grounds": ""}

CITED = "Some answer[1]\n\nSources:\n[1] A Document, p. 1"


def chunks(*sources: str) -> list[dict]:
    return [{"source": s, "score": 0.6, "title": s, "page": 1, "section": None} for s in sources]


def test_an_answer_citing_the_right_document_passes():
    assert check(ANSWERABLE, chunks("a.pdf"), CITED)[0] == "PASS"


def test_an_answer_with_no_sources_section_fails():
    assert check(ANSWERABLE, chunks("a.pdf"), "Some answer with no citations")[0] == "FAIL"


def test_refusing_an_answerable_question_fails():
    assert check(ANSWERABLE, chunks("a.pdf"), NO_ANSWER)[0] == "FAIL"


def test_citing_only_the_wrong_document_fails():
    status, reason = check(ANSWERABLE, chunks("other.pdf"), CITED)
    assert status == "FAIL"
    assert "expected material from" in reason


def test_a_partial_multi_document_answer_passes_but_says_what_was_missed():
    case = ANSWERABLE | {"expected_sources": ["a.pdf", "b.pdf"]}
    status, reason = check(case, chunks("a.pdf"), CITED)
    assert status == "PASS"
    assert "missing ['b.pdf']" in reason


def test_a_threshold_refusal_of_an_unanswerable_question_passes():
    assert check(UNANSWERABLE, [], NO_ANSWER)[0] == "PASS"


def test_a_model_written_reply_to_an_unanswerable_question_needs_a_human():
    """No string match separates an honest 'the documents don't say' from an invention."""
    reply = "The documents don't state a maximum GDPR fine.[1]\n\nSources:\n[1] A09"
    assert check(UNANSWERABLE, chunks("a.html"), reply)[0] == "REVIEW"


def test_every_question_in_the_set_is_well_formed():
    cases = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]
    assert len(cases) >= 10
    for case in cases:
        assert case["question"].strip() and case["grounds"].strip()
        assert isinstance(case["expects_answer"], bool)
        # an answerable question must say which document backs it
        assert bool(case["expected_sources"]) == case["expects_answer"]
