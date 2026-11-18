
import json
import time
from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from retrieval import retrieve_documents


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "qwen2.5:3b"
TOP_K = 3
MAX_CHARS_PER_CHUNK = 1500


# --------------------------------------------------
# Structured evaluation schema
# --------------------------------------------------

class RetrievalEvaluation(BaseModel):
    """Assessment of relevance and sufficiency of retrieved evidence."""

    decision: Literal["correct", "ambiguous", "incorrect"] = Field(
        description=(
            "correct: evidence sufficiently answers the question; "
            "ambiguous: useful evidence exists but is incomplete; "
            "incorrect: no retrieved passage provides useful evidence."
        )
    )

    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated relevance of the retrieved evidence.",
    )

    evidence_sufficient: bool = Field(
        description="Whether the evidence adequately supports an answer."
    )

    reasoning: str = Field(
        description="A concise explanation of the decision."
    )

    useful_chunk_ids: list[str] = Field(
        description=(
            "Exact IDs of retrieved passages that support answering "
            "the user's question."
        )
    )

    missing_information: str = Field(
        description=(
            "Important information missing from the evidence, or None "
            "when the evidence is sufficient."
        )
    )


# --------------------------------------------------
# Local evaluator
# --------------------------------------------------

def create_evaluator():
    """Create the local Ollama evaluator."""

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0,
        num_ctx=4096,
        num_predict=400,
        keep_alive="10m",
        format=RetrievalEvaluation.model_json_schema(),
    )

    return llm.with_structured_output(
        RetrievalEvaluation,
        method="json_schema",
    )


# --------------------------------------------------
# Evaluate retrieved documents
# --------------------------------------------------

def evaluate_documents(
    question: str,
    documents: list,
) -> RetrievalEvaluation:
    """Evaluate retrieved passages using the local LLM."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not documents:
        return RetrievalEvaluation(
            decision="incorrect",
            relevance_score=0.0,
            evidence_sufficient=False,
            reasoning="No documents were retrieved.",
            useful_chunk_ids=[],
            missing_information=(
                "All information needed to answer the question."
            ),
        )

    # Prepare compact evidence for the model.
    formatted_documents = []

    for document, distance in documents[:TOP_K]:
        formatted_documents.append(
            {
                "chunk_id": str(
                    document.metadata.get("chunk_id", "unknown")
                ),
                "source": str(
                    document.metadata.get("source", "unknown")
                ),
                "page": document.metadata.get("page", "unknown"),
                "distance": round(float(distance), 4),
                "content": document.page_content[:MAX_CHARS_PER_CHUNK],
            }
        )

    available_chunk_ids = [
        item["chunk_id"] for item in formatted_documents
    ]

    prompt = f"""
You are evaluating evidence for a research-paper question-answering system.

QUESTION:
{question}

RETRIEVED PASSAGES:
{json.dumps(formatted_documents, ensure_ascii=False, indent=2)}

AVAILABLE CHUNK IDS:
{json.dumps(available_chunk_ids)}

Evaluate each passage independently.

RELEVANCE:
- A passage is useful if it directly helps answer the question.
- One strong passage can be sufficient.
- Irrelevant passages must not invalidate a relevant passage.
- Judge the actual passage content, not just its title or distance.

SUFFICIENCY:
- Decide whether the evidence supports a useful and accurate answer.
- Do not require every technical detail if the question asks for a
  high-level explanation.
- For questions about RAGAS, evidence describing its evaluation
  framework, metrics, or evaluation process can support an answer.
- Do not assume that a passage contains information not shown in its text.

DECISION:
- correct: the evidence sufficiently answers the question.
- ambiguous: useful evidence exists, but important information is missing.
- incorrect: none of the passages provides useful evidence.

CHUNK IDS:
- Return the exact IDs of passages that support the answer.
- Use only IDs in AVAILABLE CHUNK IDS.
- Do not invent IDs.
- Return an empty list only when no passage is useful.

CONSISTENCY:
- If evidence_sufficient is true, decision must be correct.
- If evidence_sufficient is false but some passages are useful,
  decision must be ambiguous.
- If no passages are useful, decision must be incorrect.
- If evidence is sufficient, missing_information must be "None".
- Keep the reasoning concise and do not contradict your decision.

SCORING:
- relevance_score must be between 0.0 and 1.0.
- This score is an estimate, not a calibrated probability.

Return every required field in the structured schema.
"""

    evaluator = create_evaluator()
    result = evaluator.invoke(prompt)

    # Only accept IDs belonging to the retrieved documents.
    valid_chunk_ids = set(available_chunk_ids)

    result.useful_chunk_ids = [
        chunk_id
        for chunk_id in result.useful_chunk_ids
        if chunk_id in valid_chunk_ids
    ]

    # If the model omitted IDs, attempt a limited recovery from reasoning.
    if not result.useful_chunk_ids:
        reasoning_lower = result.reasoning.lower()

        for chunk_id in available_chunk_ids:
            if chunk_id.lower() in reasoning_lower:
                result.useful_chunk_ids.append(chunk_id)

    
# Normalize decisions using the evaluator's evidence judgment.
    if result.evidence_sufficient:
        result.decision = "correct"
        result.missing_information = "None"

    elif result.useful_chunk_ids:
        result.decision = "ambiguous"

    else:
        result.decision = "incorrect"

        if not result.missing_information or (
            result.missing_information.strip().lower() == "none"
        ):
            result.missing_information = (
                "The retrieved passages do not contain evidence "
                "needed to answer the question."
            )


    return result


# --------------------------------------------------
# Interactive test
# --------------------------------------------------

def main():
    print("Local Retrieval Evaluator")
    print("-" * 50)
    print("Model:", MODEL_NAME)

    while True:
        question = input(
            "\nEnter a question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            start = time.perf_counter()

            print("\n[1/3] Retrieving documents...", flush=True)
            documents = retrieve_documents(question, k=TOP_K)

            print(
                f"[2/3] Retrieved {len(documents)} documents "
                f"in {time.perf_counter() - start:.2f}s",
                flush=True,
            )

            evaluation_start = time.perf_counter()

            print(
                "[3/3] Evaluating evidence with local LLM...",
                flush=True,
            )

            evaluation = evaluate_documents(question, documents)

            print(
                f"Evaluation completed in "
                f"{time.perf_counter() - evaluation_start:.2f}s"
            )

            print("\nEvaluation")
            print("=" * 50)
            print("Decision:", evaluation.decision)
            print("Relevance score:", evaluation.relevance_score)
            print("Evidence sufficient:", evaluation.evidence_sufficient)
            print("Reasoning:", evaluation.reasoning)
            print("Useful chunks:", evaluation.useful_chunk_ids)
            print("Missing information:", evaluation.missing_information)

        except Exception as exc:
            print(f"Evaluation failed: {exc}")


if __name__ == "__main__":
    main()
