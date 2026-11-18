
import sys
import time
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from fastapi.encoders import jsonable_encoder

# Project root: parent of backend/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# Allow the existing pipeline's imports to resolve.
sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("self-corrective-rag")

app = FastAPI(
    title="Self-Corrective RAG API",
    description="API for retrieval, fallback routing, generation, and verification.",
    version="1.0.0",
)

# Development configuration. Restrict origins before deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    question: str
    answer: str
    route_taken: str | None = None
    evidence_count: int = 0
    final_action: str | None = None
    verification: dict[str, Any] | None = None
    sources: list[Any] = Field(default_factory=list)
    initial_evaluation: Any = None
    initial_route: Any = None
    verification_history: list[Any] = Field(default_factory=list)
    correction_attempts: int = 0
    elapsed_seconds: float


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "self-corrective-rag",
        "pipeline_available": True,
    }


@app.post("/api/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Please enter a question.",
        )

    try:
        # Import only when needed, so startup errors are easier to diagnose.
        from pipeline import run_pipeline

        started = time.perf_counter()

        # The local LLM call is blocking. Keep it off the async event loop.
        result = await run_in_threadpool(run_pipeline, question)

        elapsed = round(time.perf_counter() - started, 2)

        if not isinstance(result, dict):
            raise RuntimeError(
                "run_pipeline() did not return a dictionary."
            )

        result = jsonable_encoder(result)

        # Keep the response aligned with the actual pipeline result.
        payload = {
            "question": result.get("question", question),
            "answer": result.get("answer", ""),
            "route_taken": result.get("route_taken"),
            "evidence_count": result.get("evidence_count", 0),
            "final_action": result.get("final_action"),
            "verification": result.get("verification"),
            "sources": result.get("sources") or [],
            "initial_evaluation": result.get("initial_evaluation"),
            "initial_route": result.get("initial_route"),
            "verification_history": (
                result.get("verification_history") or []
            ),
            "correction_attempts": result.get("correction_attempts", 0),
            "elapsed_seconds": elapsed,
        }

        return AskResponse(**payload)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pipeline execution failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "The RAG pipeline failed. Check the backend terminal "
                "for the underlying error."
            ),
        ) from exc
