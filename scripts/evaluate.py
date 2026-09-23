"""
Runs the fixed evaluation set and checks two things per question:

  1. answered vs refused matches what the question expects
  2. when answered, the answer cites sources, and they come from the document that
     actually contains the material

This is a harness, not a judge: it verifies that answers are grounded and attributed,
which is the Cycle 1 definition of done. It does not score whether the prose is good —
that needs a human or an LLM judge, and comes with the full 30-question set.

Costs real API credits (~$0.03 per answered question).

Run with:
    python -m scripts.evaluate
    python -m scripts.evaluate --show     # also print each answer
"""
import json
import sys
from pathlib import Path

from src import config
from src.retrieval.query import NO_ANSWER, format_citation, generate_answer, relevant_chunks

QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "evaluation" / "questions.json"


def check(case: dict, chunks: list[dict], answer: str) -> tuple[str, str]:
    """
    Returns (status, reason) where status is PASS, FAIL or REVIEW.

    REVIEW exists because of a real limit: for a question the corpus cannot answer, the
    thing to verify is that the reply did not fabricate one. The model does not emit the
    canned NO_ANSWER for these — it writes its own, better refusal ("the documents don't
    state a maximum GDPR fine..."), and no string match reliably tells an honest refusal
    from a confident invention. That judgement needs a human or an LLM judge, so the
    harness surfaces these instead of pretending to grade them.
    """
    refused = not answer.strip() or answer.strip().startswith(NO_ANSWER[:40])

    if not case["expects_answer"]:
        if refused:
            return "PASS", "declined before reaching the model"
        return "REVIEW", "model replied — read it: does it decline without inventing an answer?"

    if refused:
        return "FAIL", "refused a question the corpus can answer"
    if "Sources:" not in answer:
        return "FAIL", "answered without citing anything"

    cited = {chunk["source"] for chunk in chunks}
    expected = set(case["expected_sources"])
    if not cited & expected:
        return "FAIL", f"cited {sorted(cited)}, expected material from {sorted(expected)}"
    missing = expected - cited
    return "PASS", f"cited, missing {sorted(missing)}" if missing else "cited correctly"


def main() -> None:
    config.validate()
    show = "--show" in sys.argv
    cases = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]

    results = []
    for number, case in enumerate(cases, start=1):
        chunks = relevant_chunks(case["question"])
        answer = generate_answer(case["question"], chunks)
        status, reason = check(case, chunks, answer)
        results.append((status, case, reason))

        top = f"{chunks[0]['score']:.3f}" if chunks else "  -  "
        print(f"{status:6} {number:2}. [{top}] {case['question']}")
        print(f"            {reason}")
        if chunks:
            print(f"            retrieved: {', '.join(format_citation(c) for c in chunks[:3])}")
        # a REVIEW case is only useful if its answer is on screen
        if show or status == "REVIEW":
            print("\n" + "\n".join(f"      {line}" for line in answer.splitlines()) + "\n")

    graded = [r for r in results if r[0] != "REVIEW"]
    passed = sum(1 for status, _, _ in graded if status == "PASS")
    review = len(results) - len(graded)
    print(f"\n{passed}/{len(graded)} automated checks passed, {review} flagged for review")
    for status, case, reason in results:
        if status == "FAIL":
            print(f"  FAILED: {case['question']}\n          {reason}\n          {case['grounds']}")


if __name__ == "__main__":
    main()
