<img width="1024" height="1536" alt="WhatsApp Image 2026-11-19 at 12 44 53 AM" src="https://github.com/user-attachments/assets/173db309-e5a8-4b76-b8af-adfe5370d1e5" /># Self-Corrective Retrieval-Augmented Generation (C-RAG)

## 1. Problem Statement

Traditional Retrieval-Augmented Generation (RAG) systems retrieve relevant documents from a knowledge base and use them to generate answers. However, they often rely on retrieved results without adequately evaluating whether the information is relevant, accurate, or sufficient to answer the user's query.

Retrieving irrelevant or insufficient information can lead to inaccurate responses, unsupported claims, and hallucinations.

**The objective of this project is to develop a Self-Corrective RAG (C-RAG) system that evaluates retrieved evidence, dynamically selects corrective actions, and verifies generated answers to improve response reliability and factual grounding.**


## 2. Project Overview

C-RAG extends the traditional RAG workflow by introducing retrieval evaluation, adaptive routing, evidence refinement, web search fallback, and answer verification.

Instead of blindly trusting retrieved documents, the system evaluates their quality and determines whether to use local evidence, refine retrieval, or retrieve additional information from the web.

## 3. Key Features

- **Semantic Retrieval:** Retrieves relevant document chunks using embeddings and ChromaDB.
- **Retrieval Evaluation:** Uses Qwen 2.5 3B to assess evidence relevance and sufficiency.
- **Adaptive Routing:** Selects an appropriate action based on retrieval quality.
- **Retrieval Refinement:** Performs additional retrieval when the initial evidence is ambiguous.
- **Web Search Fallback:** Uses Tavily to acquire external evidence when local retrieval is inadequate.
- **Grounded Answer Generation:** Generates responses using retrieved evidence and citations.
- **Claim Verification:** Checks generated claims against the supplied evidence.
- **Self-Correction:** Allows one correction attempt when the generated answer is not adequately supported.

## 4. System Architecture

The project consists of four major components:

1. **Knowledge Base & Retrieval:** PDF processing, text chunking, embeddings, ChromaDB storage, and semantic search.
2. **Evaluation & Routing:** Retrieval quality evaluation and adaptive decision-making.
3. **Answer Generation & Verification:** Evidence-grounded generation, citation validation, claim verification, and correction.
4. **Frontend Application:** Interactive React dashboard for submitting queries and viewing answers, sources, routing decisions, and verification results.

## 5. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM | Qwen 2.5 3B |
| Local Model Runtime | Ollama |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Web Search | Tavily |
| Backend | FastAPI |
| Frontend | React + TypeScript |
| Styling | Tailwind CSS |

## 6. Workflow

1. User submits a question through the React dashboard.
2. Relevant document chunks are retrieved from ChromaDB.
3. The retrieval evaluator assesses evidence quality.
4. The adaptive router selects an appropriate action:
   - **Correct:** Use local evidence.
   - **Ambiguous:** Refine retrieval.
   - **Incorrect:** Perform web search.
5. Selected evidence is passed to the answer generator.
6. The generated answer undergoes claim and citation verification.
7. If verification fails, the system attempts one correction.
8. The final response, sources, and verification details are displayed.

## 7. Project Objective

To build a more reliable and transparent RAG system that goes beyond basic document retrieval by evaluating evidence quality, correcting retrieval failures, and verifying generated responses before presenting them to users.

## 8. Future Enhancements

- Hybrid search combining semantic and keyword retrieval.
- Advanced query rewriting and reranking.
- Independent LLM-based verification.
- Improved hallucination detection.
- Streaming responses and real-time pipeline monitoring.
- Expanded evaluation using RAGAS metrics.

---

**Project:** Self-Corrective Retrieval-Augmented Generation (C-RAG)  
**Category:** Artificial Intelligence | Machine Learning | Generative AI  
**Focus:** Reliable, evidence-grounded question answering.
