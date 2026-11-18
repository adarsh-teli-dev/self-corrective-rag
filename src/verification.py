
import re
from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


MODEL_NAME = "qwen2.5:3b"


class ClaimCheck(BaseModel):
    """Verification result for one factual claim."""

    claim: str = Field(description="A factual claim from the answer.")
    supported: bool = Field(
        description="True only if the provided evidence supports the claim."
    )
    supporting_evidence_ids: list[int] = Field(
        description="1-based indexes of evidence items supporting the claim."
    )
    reasoning: str = Field(
        description="Brief explanation of why the claim is or is not supported."
    )


class VerificationResult(BaseModel):
    """Overall answer verification."""

    checks: list[ClaimCheck]
    overall_status: Literal[
        "verified", "partially_supported", "unsupported"
    ]
    summary: str


def create_verifier():
    """Create a local, structured-output verification model."""

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0,
        num_ctx=4096,
        num_predict=500,
        keep_alive="10m",
    )

    return llm.with_structured_output(
        VerificationResult,
        method="json_schema",
    )


def verify_answer(answer: str, evidence: list[dict]) -> dict:
    """
    Check answer citations and whether its factual claims
    are supported by the supplied evidence.

    Evidence IDs in the prompt are 1-based list positions.
    """

    if not answer or not answer.strip():
        return {
            "citation_check": {
                "valid": False,
                "used_ids": [],
                "invalid_ids": [],
            },
            "claim_verification": {
                "checks": [],
                "overall_status": "unsupported",
                "summary": "The answer is empty.",
            },
            "verified": False,
        }

    if not evidence:
        return {
            "citation_check": {
                "valid": False,
                "used_ids": [],
                "invalid_ids": [],
            },
            "claim_verification": {
                "checks": [],
                "overall_status": "unsupported",
                "summary": "No evidence was supplied to verify the answer.",
            },
            "verified": False,
        }

    # The generator uses citations such as [1] and [2].
    # Collect citation numbers from the answer text.
    used_ids = sorted(
        {
            int(number)
            for number in re.findall(r"\[(\d+)\]", answer)
        }
    )

    valid_ids = set(range(1, len(evidence) + 1))
    invalid_ids = [
        number for number in used_ids if number not in valid_ids
    ]

    formatted_evidence = []

    for index, item in enumerate(evidence, start=1):
        formatted_evidence.append(
            {
                "evidence_id": index,
                "title": item.get("title", "Unknown source"),
                "source": item.get("source", "Unknown source"),
                "page": item.get("page"),
                "url": item.get("url"),
                "content": str(item.get("content", ""))[:2500],
            }
        )

    prompt = f"""
You are a strict factual claim verifier for a RAG system.

ANSWER:
{answer}

EVIDENCE:
{formatted_evidence}

Instructions:
1. Extract the important factual claims made in the answer.
2. Check each claim against the supplied evidence only.
3. Mark supported=true only when the evidence actually supports the claim.
4. Do not treat a source title or URL alone as proof of a claim.
5. A citation being present does not automatically make a claim supported.
6. Use only the integer evidence_id values supplied above.
7. Do not invent evidence or facts.
8. If a claim is only partly supported, mark it unsupported.
9. Ignore greetings, opinions, and non-factual transition sentences.
10. If all factual claims are supported, overall_status must be "verified".
11. If some factual claims are supported and others are not,
    overall_status must be "partially_supported".
12. If no meaningful factual claims are supported,
    overall_status must be "unsupported".
13. Keep reasoning and summary concise.

Return the required structured output.
"""

    try:
        result = create_verifier().invoke(prompt)

        # Discard any evidence indexes the model invented.
        for check in result.checks:
            check.supporting_evidence_ids = sorted(
                {
                    evidence_id
                    for evidence_id in check.supporting_evidence_ids
                    if evidence_id in valid_ids
                }
            )

            # A claim cannot be considered supported if the model
            # supplies no valid supporting evidence IDs.
            if not check.supporting_evidence_ids:
                check.supported = False

        has_supported = any(c.supported for c in result.checks)
        has_unsupported = any(not c.supported for c in result.checks)

        if not result.checks or not has_supported:
            result.overall_status = "unsupported"
        elif has_unsupported:
            result.overall_status = "partially_supported"
        else:
            result.overall_status = "verified"

        citation_valid = len(invalid_ids) == 0

        # A citation referencing a nonexistent evidence item means
        # the answer cannot receive a fully verified status.
        fully_verified = (
            result.overall_status == "verified"
            and citation_valid
        )

        return {
            "citation_check": {
                "valid": citation_valid,
                "used_ids": used_ids,
                "invalid_ids": invalid_ids,
            },
            "claim_verification": result.model_dump(),
            "verified": fully_verified,
        }

    except Exception as exc:
        # Verification failures must never be reported as success.
        return {
            "citation_check": {
                "valid": len(invalid_ids) == 0,
                "used_ids": used_ids,
                "invalid_ids": invalid_ids,
            },
            "claim_verification": {
                "checks": [],
                "overall_status": "unsupported",
                "summary": f"Verification failed: {exc}",
            },
            "verified": False,
        }


def main():
    """Run a small verification test without using the full pipeline."""

    sample_evidence = [
        {
            "title": "Example research paper",
            "source": "example.pdf",
            "page": 1,
            "content": (
                "RAGAS evaluates retrieval-augmented generation systems "
                "using metrics that assess retrieval and generated answers."
            ),
        }
    ]

    sample_answer = (
        "RAGAS evaluates retrieval-augmented generation systems "
        "using metrics for retrieval and generated answers [1]."
    )

    print("Testing claim and citation verification...")
    result = verify_answer(sample_answer, sample_evidence)

    print("\nVerification result:")
    print(result)


if __name__ == "__main__":
    main()
