
from pathlib import Path
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "research_papers"

TOP_K = 5


# --------------------------------------------------
# Load and cache vector database
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_vectorstore():
    """
    Load the embedding model and ChromaDB collection once.

    Reuse the same instances for subsequent retrieval requests.
    """

    print("\n[RETRIEVAL] Loading embedding model...", flush=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("[RETRIEVAL] Connecting to ChromaDB...", flush=True)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )

    print(
        "[RETRIEVAL] Vector database loaded successfully.",
        flush=True,
    )

    return vectorstore


# --------------------------------------------------
# Retrieve relevant documents
# --------------------------------------------------

def retrieve_documents(question: str, k: int = TOP_K):
    """
    Retrieve the top-k chunks with their relevance distances.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search_with_score(
        question,
        k=k,
    )

    return results


# --------------------------------------------------
# Interactive retrieval tester
# --------------------------------------------------

def main():

    print("Local RAG Retrieval Tester")
    print("-" * 50)

    vectorstore = load_vectorstore()

    print("Loaded collection:", COLLECTION_NAME)
    print("Indexed chunks:", vectorstore._collection.count())

    while True:

        question = input(
            "\nEnter a question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            break

        try:

            results = retrieve_documents(
                question,
                k=TOP_K,
            )

            if not results:
                print("No documents retrieved.")
                continue

            for rank, (document, distance) in enumerate(
                results,
                start=1,
            ):

                print(f"\n{'=' * 60}")
                print(f"Result {rank}")
                print(f"Distance: {distance:.4f}")

                print(
                    "Source:",
                    document.metadata.get("source", "Unknown"),
                )

                print(
                    "Page:",
                    document.metadata.get("page", "Unknown"),
                )

                print(
                    "Chunk ID:",
                    document.metadata.get("chunk_id", "Unknown"),
                )

                print("-" * 60)

                print(document.page_content[:1200])

        except Exception as exc:
            print(f"Retrieval failed: {exc}")


if __name__ == "__main__":
    main()
