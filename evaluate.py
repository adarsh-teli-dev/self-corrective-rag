
import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"

sys.path.insert(0, str(SRC_DIR))

from retrieval import retrieve_documents
from generator import generate_answer
from verification import verify_answer
from pipeline import run_pipeline, format_local_evidence


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_dataset():
    """Load and validate the evaluation dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list) or not questions:
        raise ValueError("The evaluation dataset must be a non-empty list.")

    required_fields = {
        "id",
        "question",
        "category",
        "reference_answer",
        "expected_behavior",
    }

    for item in questions:
        missing = required_fields - item.keys()

        if missing:
            raise ValueError(
                f"Question {item.get('id', 'unknown')} "
                f"is missing fields: {sorted(missing)}"
            )

    return questions


def safe_verification_fields(verification):
    """Extract consistent verification fields."""

    if not isinstance(verification, dict):
        return {
            "verified": False,
            "status": "unavailable",
            "citation_valid": False,
        }

    claim_result = verification.get("claim_verification", {})

    return {
        "verified": verification.get("verified", False),
        "status": claim_result.get(
            "overall_status", "unavailable"
        ),
        "citation_valid": verification.get(
            "citation_check", {}
        ).get("valid", False),
    }


def run_baseline(question):
    """
    Conventional local RAG:
    retrieve local documents and generate an answer.

    Verification is performed afterward for measurement only.
    It does not correct or alter the baseline answer.
    """

    start = time.perf_counter()

    documents = retrieve_documents(question, k=3)
    evidence = format_local_evidence(documents)

    retrieval_complete = time.perf_counter()

    answer = generate_answer(question, evidence)

    generation_complete = time.perf_counter()

    return {
        "answer": answer,
        "evidence": evidence,
        "evidence_count": len(evidence),
        "retrieval_seconds": retrieval_complete - start,
        "generation_seconds": generation_complete - retrieval_complete,
        "system_seconds": generation_complete - start,
    }


def evaluate_one(item, system):
    """Run one dataset item through the selected system."""

    question = item["question"]

    base = {
        "question_id": item["id"],
        "category": item["category"],
        "question": question,
        "reference_answer": item["reference_answer"],
        "expected_behavior": item["expected_behavior"],
        "system": system,
        "answer": "",
        "route_taken": "",
        "evidence_count": 0,
        "retrieval_seconds": "",
        "generation_seconds": "",
        "system_seconds": "",
        "verification_seconds": "",
        "verified": False,
        "verification_status": "unavailable",
        "citation_valid": False,
        "correction_attempts": 0,
        "final_action": "",
        "web_search_used": False,
        "error": "",
    }

    try:
        if system == "baseline":
            result = run_baseline(question)

            base.update({
                "answer": result["answer"],
                "route_taken": "local_only",
                "evidence_count": result["evidence_count"],
                "retrieval_seconds": round(
                    result["retrieval_seconds"], 3
                ),
                "generation_seconds": round(
                    result["generation_seconds"], 3
                ),
                "system_seconds": round(
                    result["system_seconds"], 3
                ),
                "final_action": "generated",
            })

            # Post-hoc measurement only. This verification does not
            # affect the conventional RAG answer.
            verification_start = time.perf_counter()

            verification = verify_answer(
                result["answer"],
                result["evidence"],
            )

            base["verification_seconds"] = round(
                time.perf_counter() - verification_start, 3
            )

            fields = safe_verification_fields(verification)

            base.update({
                "verified": fields["verified"],
                "verification_status": fields["status"],
                "citation_valid": fields["citation_valid"],
            })

        elif system == "crag":
            start = time.perf_counter()

            result = run_pipeline(question)

            elapsed = time.perf_counter() - start

            verification = result.get("verification", {})
            fields = safe_verification_fields(verification)

            base.update({
                "answer": result.get("answer", ""),
                "route_taken": result.get("route_taken", ""),
                "evidence_count": result.get("evidence_count", 0),
                "system_seconds": round(elapsed, 3),
                "verified": fields["verified"],
                "verification_status": fields["status"],
                "citation_valid": fields["citation_valid"],
                "correction_attempts": result.get(
                    "correction_attempts", 0
                ),
                "final_action": result.get("final_action", ""),
                "web_search_used": (
                    "web_search" in result.get("route_taken", "")
                ),
            })

        else:
            raise ValueError(f"Unknown system: {system}")

    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        base["final_action"] = "error"

        print(
            f"\n[ERROR] {system} failed for {item['id']}: {exc}"
        )

    return base


# --------------------------------------------------
# Benchmark runner
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare conventional RAG with self-corrective RAG."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum questions to run. Use 0 for the entire dataset.",
    )

    parser.add_argument(
        "--systems",
        default="baseline,crag",
        help="Comma-separated systems: baseline,crag",
    )

    args = parser.parse_args()

    systems = [
        system.strip().lower()
        for system in args.systems.split(",")
        if system.strip()
    ]

    allowed_systems = {"baseline", "crag"}

    if not systems or not set(systems).issubset(allowed_systems):
        raise ValueError(
            "Systems must be baseline, crag, or baseline,crag."
        )

    questions = load_dataset()

    if args.limit > 0:
        questions = questions[:args.limit]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = RESULTS_DIR / f"benchmark_{timestamp}.csv"
    json_path = RESULTS_DIR / f"benchmark_{timestamp}.json"

    rows = []
    total_runs = len(questions) * len(systems)
    run_number = 0

    print("=" * 65)
    print("SELF-CORRECTIVE RAG BENCHMARK")
    print("=" * 65)
    print("Dataset questions:", len(questions))
    print("Systems:", ", ".join(systems))
    print("Total runs:", total_runs)
    print("Results directory:", RESULTS_DIR)
    print(
        "\nNote: baseline verification is post-hoc and excluded "
        "from baseline system latency."
    )

    for item in questions:
        for system in systems:
            run_number += 1

            print(
                f"\n[{run_number}/{total_runs}] "
                f"{system.upper()} | {item['id']}: "
                f"{item['question']}"
            )

            row = evaluate_one(item, system)
            rows.append(row)

            print("Route:", row["route_taken"])
            print("Final action:", row["final_action"])
            print("Verified:", row["verified"])
            print("Latency:", row["system_seconds"], "seconds")

            if row["error"]:
                print("Error:", row["error"])

            # Save after every run so progress survives interruption.
            with open(
                csv_path, "w", newline="", encoding="utf-8-sig"
            ) as file:
                writer = csv.DictWriter(
                    file, fieldnames=list(rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(rows)

            with open(json_path, "w", encoding="utf-8") as file:
                json.dump(rows, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)

    for system in systems:
        system_rows = [
            row for row in rows
            if row["system"] == system
        ]

        successful = [
            row for row in system_rows if not row["error"]
        ]

        verified_count = sum(
            row["verified"] is True for row in successful
        )

        web_count = sum(
            row["web_search_used"] is True for row in successful
        )

        latencies = [
            float(row["system_seconds"])
            for row in successful
            if row["system_seconds"] != ""
        ]

        average_latency = (
            sum(latencies) / len(latencies)
            if latencies else 0.0
        )

        print(f"\nSystem: {system}")
        print("Runs:", len(system_rows))
        print("Successful runs:", len(successful))
        print("Errors:", len(system_rows) - len(successful))
        print("Verified answers:", verified_count)
        print("Web-search runs:", web_count)
        print(f"Mean system latency: {average_latency:.2f} seconds")

    print("\nResults saved:")
    print(csv_path)
    print(json_path)

    print(
        "\nReminder: verification rate is not answer correctness. "
        "Review answers against reference answers and evidence "
        "before drawing conclusions."
    )


if __name__ == "__main__":
    main()
