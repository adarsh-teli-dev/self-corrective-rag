
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

# Initial chunking configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


def load_pdfs():
    """Load PDFs and retain source filename and page metadata."""
    pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {RAW_DATA_DIR}. "
            "Add research papers to data/raw/."
        )

    documents = []

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf_path.name
            # PyPDFLoader normally uses zero-based page numbers.
            page.metadata["page"] = page.metadata.get("page", 0) + 1

        documents.extend(pages)
        print(f"  Loaded {len(pages)} pages")

    return documents


def split_documents(documents):
    """Split page documents using token-based chunk-size measurement."""
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

    def token_length(text):
        return len(tokenizer.encode(text, add_special_tokens=False))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_length,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    # Give each chunk a stable, human-readable identifier.
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{index:06d}"

    return chunks


def create_vectorstore(chunks):
    """Embed chunks and persist them in ChromaDB."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name="research_papers",
    )

    return vectorstore


def main():
    print("Starting document ingestion...")

    documents = load_pdfs()
    print(f"\nTotal pages loaded: {len(documents)}")

    chunks = split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        print("\nSample chunk metadata:")
        print(chunks[0].metadata)

        print("\nSample chunk text:")
        print(chunks[0].page_content[:500])

    vectorstore = create_vectorstore(chunks)

    print(f"\nVector database saved to: {VECTORSTORE_DIR}")
    print(f"Collection: {vectorstore._collection.name}")
    print(f"Indexed chunks: {vectorstore._collection.count()}")
    print("\nIngestion completed successfully!")


if __name__ == "__main__":
    main()
