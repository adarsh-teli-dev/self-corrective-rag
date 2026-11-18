
import time

from retrieval import retrieve_documents
from evaluator import evaluate_documents
from router import route_query
from verification import verify_answer
from web_search import search_web
from generator import generate_answer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MAX_CORRECTION_ATTEMPTS = 1

ABSTENTION_MESSAGE = (
    "I could not establish a sufficiently supported answer from the "
    "available evidence. Please consult reliable sources or provide "
    "additional documents."
)


# --------------------------------------------------
# Local evidence formatting
# --------------------------------------------------

def format_local_evidence(documents: list) -> list[dict]:
    """Convert retrieved LangChain documents into generator evidence."""

    evidence = []

    for document, distance in documents:
        evidence.append(
            {
                "content": document.page_content,
                "source": document.metadata.get("source", "unknown"),
                "title": document.metadata.get("source", "Local document"),
                "page": document.metadata.get("page"),
                "chunk_id": document.metadata.get("chunk_id"),
                "distance": float(distance),
            }
        )

    return evidence


# --------------------------------------------------
# Answer correction
# --------------------------------------------------

def correct_answer(
    question: str,
    previous_answer: str,
    evidence: list[dict],
    verification: dict,
) -> str:
    """Regenerate an answer using verification feedback."""

    failed_claims = []

    claim_result = verification.get("claim_verification", {})

    for check in claim_result.get("checks", []):
        if not check.get("supported", False):
            failed_claims.append(
                {
                    "claim": check.get("claim", ""),
                    "reasoning": check.get("reasoning", ""),
                }
            )

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
You are correcting an answer in a retrieval-augmented generation system.

QUESTION:
{question}

PREVIOUS ANSWER:
{previous_answer}

VERIFICATION FEEDBACK:
{failed_claims}

VERIFIER SUMMARY:
{claim_result.get("summary", "The answer was not fully verified.")}

CITATION CHECK:
{verification.get("citation_check", {})}

AVAILABLE EVIDENCE:
{formatted_evidence}

Instructions:
1. Answer the question using only the available evidence.
2. Remove or rewrite claims that are not supported.
3. Do not invent facts, sources, URLs, page numbers, or citations.
4. Cite evidence using its 1-based evidence_id, such as [1] or [2].
5. Make sure each citation refers to the evidence supporting the claim.
6. If the evidence is insufficient, clearly state that limitation.
7. Do not claim that your answer is verified.
8. Return only the corrected answer, not your internal reasoning.
"""

    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
        num_ctx=4096,
        num_predict=500,
        keep_alive="10m",
    )

    response = llm.invoke(prompt)
    return response.content.strip()


# --------------------------------------------------
# Verification and correction loop
# --------------------------------------------------

def verify_and_correct(
    question: str,
    answer: str,
    evidence: list[dict],
    timings: dict | None = None,
) -> dict:
    """
    Verify an answer, attempt one correction if needed,
    and abstain if the final answer remains unverified.
    """

    verification_history = []

    current_answer = answer
    correction_attempts = 0

    if timings is None:
        timings = {}

    if not evidence:
        start = time.perf_counter()
        verification = verify_answer(ABSTENTION_MESSAGE, [])
        timings["verification_initial"] = round(
            time.perf_counter() - start, 2
        )

        return {
            "answer": ABSTENTION_MESSAGE,
            "verification": verification,
            "history": [],
            "correction_attempts": 0,
            "final_action": "abstain_no_evidence",
        }

    # First verification
    start = time.perf_counter()

    current_verification = verify_answer(
        current_answer,
        evidence,
    )

    timings["verification_initial"] = round(
        time.perf_counter() - start, 2
    )

    verification_history.append(
        {
            "attempt": 0,
            "verification": current_verification,
        }
    )

    if current_verification.get("verified", False):
        return {
            "answer": current_answer,
            "verification": current_verification,
            "history": verification_history,
            "correction_attempts": 0,
            "final_action": "accepted",
        }

    # Correction loop
    for attempt in range(1, MAX_CORRECTION_ATTEMPTS + 1):

        try:
            print(
                f"\n[SAFETY] Verification failed. "
                f"Attempting correction {attempt}...",
                flush=True,
            )

            correction_start = time.perf_counter()

            corrected_answer = correct_answer(
                question=question,
                previous_answer=current_answer,
                evidence=evidence,
                verification=current_verification,
            )

            timings[f"correction_generation_{attempt}"] = round(
                time.perf_counter() - correction_start, 2
            )

            correction_attempts += 1

            if not corrected_answer:
                print(
                    "[SAFETY] Correction returned an empty answer.",
                    flush=True,
                )
                break

            verification_start = time.perf_counter()

            corrected_verification = verify_answer(
                corrected_answer,
                evidence,
            )

            timings[f"verification_after_correction_{attempt}"] = round(
                time.perf_counter() - verification_start, 2
            )

            verification_history.append(
                {
                    "attempt": attempt,
                    "verification": corrected_verification,
                }
            )

            current_answer = corrected_answer
            current_verification = corrected_verification

            if current_verification.get("verified", False):
                return {
                    "answer": current_answer,
                    "verification": current_verification,
                    "history": verification_history,
                    "correction_attempts": correction_attempts,
                    "final_action": "corrected_and_accepted",
                }

            print(
                "[SAFETY] Corrected answer is still not fully verified.",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[SAFETY] Correction failed: {exc}",
                flush=True,
            )
            break

    return {
        "answer": ABSTENTION_MESSAGE,
        "verification": current_verification,
        "history": verification_history,
        "correction_attempts": correction_attempts,
        "final_action": "abstain_unverified",
    }


# --------------------------------------------------
# Main RAG pipeline
# --------------------------------------------------

def run_pipeline(question: str) -> dict:
    """Run the complete self-corrective RAG workflow."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    pipeline_start = time.perf_counter()

    timings = {}

    # --------------------------------------------------
    # 1. Local retrieval
    # --------------------------------------------------

    stage_start = time.perf_counter()

    local_documents = retrieve_documents(question, k=3)

    timings["local_retrieval"] = round(
        time.perf_counter() - stage_start, 2
    )

    print("\n[DEBUG] Retrieved local passages:", flush=True)

    for index, (document, distance) in enumerate(
        local_documents, start=1
    ):
        print(f"\nResult {index}", flush=True)
        print("Distance:", round(float(distance), 4), flush=True)
        print(
            "Source:",
            document.metadata.get("source"),
            flush=True,
        )
        print(
            "Page:",
            document.metadata.get("page"),
            flush=True,
        )
        print(
            "Chunk ID:",
            document.metadata.get("chunk_id"),
            flush=True,
        )
        print(
            "Content:",
            document.page_content[:700],
            flush=True,
        )

    # --------------------------------------------------
    # 2. Evaluate local evidence
    # --------------------------------------------------

    stage_start = time.perf_counter()

    evaluation = evaluate_documents(
        question,
        local_documents,
    )

    timings["evaluation"] = round(
        time.perf_counter() - stage_start, 2
    )

    print("\n[DEBUG] Initial evaluation:", flush=True)
    print("Decision:", evaluation.decision, flush=True)
    print(
        "Evidence sufficient:",
        evaluation.evidence_sufficient,
        flush=True,
    )
    print("Reasoning:", evaluation.reasoning, flush=True)
    print("Useful chunks:", evaluation.useful_chunk_ids, flush=True)

    # --------------------------------------------------
    # 3. Select routing strategy
    # --------------------------------------------------

    stage_start = time.perf_counter()

    routing = route_query(question, evaluation)

    timings["routing"] = round(
        time.perf_counter() - stage_start, 2
    )

    print("\n[DEBUG] Initial routing:", flush=True)
    print("Action:", routing.action, flush=True)
    print("Reason:", routing.reason, flush=True)

    evidence = []
    route_taken = routing.action

    # Initialize additional stage timings
    timings["refined_retrieval"] = 0
    timings["refined_evaluation"] = 0
    timings["web_search"] = 0

    # --------------------------------------------------
    # 4. Use sufficient local evidence
    # --------------------------------------------------

    if routing.action == "use_local":

        evidence = format_local_evidence(local_documents)

    # --------------------------------------------------
    # 5. Refine local retrieval if necessary
    # --------------------------------------------------

    elif routing.action == "refine_retrieval":

        stage_start = time.perf_counter()

        refined_documents = retrieve_documents(
            question,
            k=5,
        )

        timings["refined_retrieval"] = round(
            time.perf_counter() - stage_start, 2
        )

        stage_start = time.perf_counter()

        refined_evaluation = evaluate_documents(
            question,
            refined_documents,
        )

        timings["refined_evaluation"] = round(
            time.perf_counter() - stage_start, 2
        )

        print("\n[DEBUG] Refined evaluation:", flush=True)
        print(
            "Decision:",
            refined_evaluation.decision,
            flush=True,
        )
        print(
            "Evidence sufficient:",
            refined_evaluation.evidence_sufficient,
            flush=True,
        )

        if (
            refined_evaluation.decision == "correct"
            and refined_evaluation.evidence_sufficient
        ):

            evidence = format_local_evidence(refined_documents)
            route_taken = "refined_local"

        else:

            try:
                stage_start = time.perf_counter()

                web_results = search_web(
                    question,
                    max_results=3,
                )

                timings["web_search"] = round(
                    time.perf_counter() - stage_start, 2
                )

                if web_results:
                    evidence = web_results
                    route_taken = "web_search_after_refinement"

            except Exception as exc:
                print(
                    f"[WARNING] Fallback web search failed: {exc}",
                    flush=True,
                )

    # --------------------------------------------------
    # 6. Web fallback for incorrect local evidence
    # --------------------------------------------------

    elif routing.action == "web_search":

        try:
            stage_start = time.perf_counter()

            web_results = search_web(
                question,
                max_results=3,
            )

            timings["web_search"] = round(
                time.perf_counter() - stage_start, 2
            )

            if web_results:
                evidence = web_results
                route_taken = "web_search"

        except Exception as exc:
            print(
                f"[WARNING] Fallback web search failed: {exc}",
                flush=True,
            )

    # --------------------------------------------------
    # 7. Generate initial answer
    # --------------------------------------------------

    stage_start = time.perf_counter()

    answer = generate_answer(
        question,
        evidence,
    )

    timings["generation"] = round(
        time.perf_counter() - stage_start, 2
    )

    # --------------------------------------------------
    # 8. Verify and correct answer
    # --------------------------------------------------

    stage_start = time.perf_counter()

    safety_result = verify_and_correct(
        question=question,
        answer=answer,
        evidence=evidence,
        timings=timings,
    )

    timings["verification_and_correction"] = round(
        time.perf_counter() - stage_start, 2
    )

    final_answer = safety_result["answer"]
    final_verification = safety_result["verification"]

    # --------------------------------------------------
    # 9. Total execution time
    # --------------------------------------------------

    timings["total"] = round(
        time.perf_counter() - pipeline_start, 2
    )

    # --------------------------------------------------
    # 10. Print performance report
    # --------------------------------------------------

    print("\n" + "=" * 55, flush=True)
    print("[PERFORMANCE] PIPELINE TIMINGS", flush=True)
    print("=" * 55, flush=True)

    for stage, duration in timings.items():
        print(
            f"{stage:<32}: {duration:>8.2f}s",
            flush=True,
        )

    print("=" * 55, flush=True)

    # --------------------------------------------------
    # 11. Return complete pipeline result
    # --------------------------------------------------

    return {
        "question": question,
        "answer": final_answer,

        "initial_evaluation": evaluation.model_dump(),
        "initial_route": routing.model_dump(),

        "route_taken": route_taken,
        "evidence_count": len(evidence),

        "sources": [
            {
                "title": item.get("title", "Untitled source"),
                "url": item.get("url"),
                "source": item.get("source", "unknown"),
                "page": item.get("page"),
                "chunk_id": item.get("chunk_id"),
            }
            for item in evidence
        ],

        "verification": final_verification,
        "verification_history": safety_result["history"],

        "correction_attempts": safety_result["correction_attempts"],
        "final_action": safety_result["final_action"],

        "timings": timings,
    }


# --------------------------------------------------
# Interactive application
# --------------------------------------------------

def main():

    print("Self-Corrective RAG")
    print("=" * 50)

    while True:

        question = input(
            "\nEnter your question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            result = run_pipeline(question)

            print("\nANSWER")
            print("-" * 50)
            print(result["answer"])

            print("\nPIPELINE DETAILS")
            print("-" * 50)

            print(
                "Initial decision:",
                result["initial_evaluation"]["decision"],
            )

            print(
                "Initial route:",
                result["initial_route"]["action"],
            )

            print("Route taken:", result["route_taken"])
            print("Evidence count:", result["evidence_count"])
            print(
                "Correction attempts:",
                result["correction_attempts"],
            )
            print("Final action:", result["final_action"])

            print("\nSOURCES")
            print("-" * 50)

            for source in result["sources"]:
                print(source)

            print("\nFINAL VERIFICATION")
            print("-" * 50)

            verification = result["verification"]

            print("Fully verified:", verification["verified"])

            print(
                "Citation check:",
                verification["citation_check"]["valid"],
            )

            claim_result = verification["claim_verification"]

            print(
                "Overall status:",
                claim_result["overall_status"],
            )

            print("Summary:", claim_result["summary"])

            for check in claim_result["checks"]:

                print("\nClaim:", check["claim"])
                print("Supported:", check["supported"])

                print(
                    "Supporting evidence:",
                    check["supporting_evidence_ids"],
                )

                print("Reasoning:", check["reasoning"])

            print("\nVERIFICATION HISTORY")
            print("-" * 50)

            for entry in result["verification_history"]:

                print(
                    f"Attempt {entry['attempt']}:",
                    entry["verification"]["claim_verification"][
                        "overall_status"
                    ],
                )

        except Exception as exc:
            print(f"Pipeline failed: {exc}")


if __name__ == "__main__":
    main()
