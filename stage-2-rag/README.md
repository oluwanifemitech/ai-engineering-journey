# Stage 2 — RAG & Vector Databases

Building a full Retrieval Augmented Generation (RAG) pipeline from scratch — embeddings, vector storage, retrieval, and grounded generation with Claude.

## What's in here

### `embeddings_intro.py`
Generating sentence embeddings and measuring semantic similarity using cosine similarity. Demonstrates that meaning — not keywords — drives retrieval.

### `vector_store.py`
Storing and querying embeddings in ChromaDB with cosine similarity space. Includes the fix for ChromaDB's default L2 distance metric.

### `rag_pipeline.py`
Full RAG pipeline: question → embed → retrieve → augment prompt → Claude generates grounded answer with source citations.

### `rag_real_document.py`
RAG over a real document with chunking strategy — chunk size, overlap, and why both matter for retrieval quality.

---

## Key learnings

- Embeddings capture **meaning not keywords** — "training accuracy is high but validation is low" retrieves overfitting docs with no shared words
- Always specify `hnsw:space: cosine` in ChromaDB — the default L2 metric produces misleading negative similarity scores
- **Similarity score predicts answer quality** — low similarity means Claude will correctly flag gaps rather than hallucinate
- Chunk size and overlap are hyperparameters — tune them like you would any ML parameter
- Sweet spot for most documents: **150–300 words per chunk, 10–15% overlap**
- A RAG system that admits knowledge gaps is more trustworthy than one that fills them with hallucinations

---

## Results

| Query | Top Similarity | Outcome |
|---|---|---|
| L1 vs L2 regularization | 0.639 | Complete, cited answer |
| Random Forest vs Gradient Boosting | 0.731 | Complete, cited answer |
| Imbalanced classification metrics | 0.432 | Partial — gaps correctly flagged |

---

## Setup

```bash
pip install anthropic sentence-transformers chromadb
export ANTHROPIC_API_KEY="your-key-here"
python rag_pipeline.py
```

## Companion article
[RAG from Scratch — A Data Scientist builds a document Q&A system](https://medium.com/@nifemiafolayanofficial)

**GitHub repo:** [github.com/oluwanifemitech/ai-engineering-journey](https://github.com/oluwanifemitech/ai-engineering-journey)
