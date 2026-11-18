
from typing import Literal

from pydantic import BaseModel

from evaluator import RetrievalEvaluation


class RoutingDecision(BaseModel):
    """Action selected by the self-corrective RAG router."""

    action: Literal[
        "use_local",
        "refine_retrieval",
        "web_search",
        "abstain",
    ]

    reason: str
    query: str


def route_query(
    question: str,
    evaluation: RetrievalEvaluation,
) -> RoutingDecision:
    """
    Select the next action based on retrieval evaluation.

    This version decides the action but does not execute web search
    or query rewriting yet.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if evaluation.decision == "correct" and evaluation.evidence_sufficient:
        return RoutingDecision(
            action="use_local",
            reason="Retrieved evidence was judged sufficient.",
            query=question,
        )

    if evaluation.decision == "ambiguous":
        return RoutingDecision(
            action="refine_retrieval",
            reason=(
                "Some useful evidence may exist, but retrieval needs "
                "further refinement."
            ),
            query=question,
        )

    if evaluation.decision == "incorrect":
        return RoutingDecision(
            action="web_search",
            reason=(
                "Local evidence was judged inadequate. "
                "A fallback source is needed."
            ),
            query=question,
        )

    return RoutingDecision(
        action="abstain",
        reason="The evaluation was inconsistent or could not be trusted.",
        query=question,
    )


def main():
    print("Self-Corrective RAG Router Tester")
    print("-" * 50)

    test_cases = [
        ("correct", True),
        ("ambiguous", False),
        ("incorrect", False),
    ]

    for decision, sufficient in test_cases:
        # Build a minimal evaluation for testing routing logic.
        evaluation = RetrievalEvaluation(
            decision=decision,
            relevance_score=0.9 if sufficient else 0.3,
            evidence_sufficient=sufficient,
            reasoning="Test evaluation.",
            useful_chunk_ids=[],
            missing_information="Test data.",
        )

        result = route_query(
            "What problem does CRAG solve?",
            evaluation,
        )

        print(f"\nEvaluation: {decision}")
        print(f"Evidence sufficient: {sufficient}")
        print(f"Selected action: {result.action}")
        print(f"Reason: {result.reason}")


if __name__ == "__main__":
    main()
