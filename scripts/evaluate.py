"""
Runs the fixed evaluation set. Per question:

  1. answered vs refused matches what the question expects
  2. when answered, the answer cites the document that actually contains the material
  3. when answered, an LLM judge scores faithfulness and relevancy (see scripts/judge.py)

Costs real API credits.

Run with:
    python -m scripts.evaluate
    python -m scripts.evaluate --show     # also print each answer
"""
import json
import sys
from pathlib import Path

from scripts.judge import Judge
from src import config
from src.retrieval.query import NO_ANSWER, format_citation, generate_answer, relevant_chunks

QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "evaluation" / "questions.json"


def check(case: dict, chunks: list[dict], answer: str) -> tuple[str, str]:
    """
    Returns (status, reason): PASS, FAIL or REVIEW. REVIEW is a model-written reply to a
    question the corpus cannot answer; no string match tells an honest refusal from an
    invented answer, so a human reads it.
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

    judge = Judge()
    results, scores = [], []
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
        if case["expects_answer"] and status == "PASS":
            score = judge.score(case["question"], answer, chunks)
            scores.append(score)
            print(f"            faithfulness {score['faithfulness']:.2f}  relevancy {score['answer_relevancy']:.2f}")
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
