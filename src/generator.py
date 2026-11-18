
from langchain_ollama import ChatOllama


MODEL_NAME = "qwen2.5:3b"


def generate_answer(question: str, evidence: list[dict]) -> str:
    """
    Generate an answer grounded in local or web evidence.

    Each evidence item should contain:
    - content
    - source
    - optionally title, url, page, and chunk_id
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not evidence:
        return (
            "I could not find sufficient evidence to answer this question "
            "reliably. Please try a more specific question or provide "
            "additional sources."
        )

    formatted_evidence = []

    for index, item in enumerate(evidence, start=1):
        content = item.get("content", "")
        if not content:
            continue

        formatted_evidence.append(
            {
                "evidence_id": index,
                "source": item.get("source", "unknown"),
                "title": item.get("title", "Untitled source"),
                "url": item.get("url"),
                "page": item.get("page"),
                "chunk_id": item.get("chunk_id"),
                "content": content[:2000],
            }
        )

    if not formatted_evidence:
        return "No usable evidence was available to generate an answer."

    prompt = f"""
You are a careful research assistant.

Answer the user's question using only the supplied evidence.

Rules:
1. Do not invent facts or use unsupported outside knowledge.
2. If the evidence does not answer the question, explicitly say so.
3. Distinguish facts supported by evidence from uncertainty.
4. Cite claims using the evidence IDs, for example [1] or [2].
5. At the end, list the sources used, including their titles and URLs
   when available. For local sources, include the document and page.
6. Do not claim to have verified a source beyond the supplied content.

Question:
{question}

Evidence:
{formatted_evidence}
"""

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0,
        num_ctx=4096,
        num_predict=500,
        keep_alive="10m",
    )

    response = llm.invoke(prompt)
    return response.content.strip()


def main():
    print("Grounded Answer Generator Tester")
    print("-" * 50)

    question = "What is the capital of Japan?"

    evidence = [
        {
            "source": "web",
            "title": "Japan overview",
            "url": "https://example.com/japan",
            "content": (
                "Tokyo is the capital of Japan. "
                "This is demonstration evidence for testing."
            ),
        }
    ]

    print("Question:", question)
    print("\nGenerating answer...\n")
    print(generate_answer(question, evidence))


if __name__ == "__main__":
    main()
