"""
RAG Pipeline — Stage 2
AI Engineering Journey

Full Retrieval Augmented Generation pipeline:
1. Embed documents using sentence-transformers
2. Store in ChromaDB with cosine similarity
3. Retrieve relevant chunks for a query
4. Pass context to Claude for grounded generation

Author: Oluwanifemi Afolayan
Medium: https://medium.com/@nifemiafolayanofficial
GitHub: https://github.com/oluwanifemitech
"""

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer


# ── Models & Clients ──────────────────────────────────────────
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
anthropic_client = anthropic.Anthropic()
db_client = chromadb.PersistentClient(path="./stage2_db")


# ── Knowledge Base ────────────────────────────────────────────
ML_DOCUMENT = """
# Machine Learning Reference Guide

## Overfitting and Underfitting
Overfitting occurs when a model learns the training data too well, capturing noise 
and random fluctuations rather than the underlying pattern. An overfit model performs 
excellently on training data but poorly on unseen data. Common signs include a large 
gap between training and validation accuracy.

Underfitting occurs when a model is too simple to capture the underlying pattern in 
the data. An underfit model performs poorly on both training and validation data. 
This is also called high bias.

## Regularization Techniques
L1 regularization (Lasso) adds the absolute value of weights as a penalty term to 
the loss function. This drives some weights to exactly zero, effectively performing 
feature selection. L1 is useful when you suspect many features are irrelevant.

L2 regularization (Ridge) adds the squared value of weights as a penalty term. 
This shrinks all weights toward zero but rarely makes them exactly zero. L2 is 
useful when most features contribute somewhat to the prediction.

Dropout is a regularization technique specific to neural networks. During training, 
random neurons are temporarily disabled with a given probability. This prevents 
neurons from co-adapting and forces the network to learn more robust features.

## Model Evaluation
The confusion matrix is a table that describes the performance of a classification 
model. It shows true positives, true negatives, false positives, and false negatives. 
From these, metrics like precision, recall, and F1-score can be derived.

Precision measures the proportion of positive predictions that were actually correct. 
It is calculated as true positives divided by true positives plus false positives. 
High precision means few false alarms.

Recall measures the proportion of actual positives that were correctly identified. 
It is calculated as true positives divided by true positives plus false negatives. 
High recall means few missed positives.

F1-score is the harmonic mean of precision and recall. It provides a single metric 
that balances both concerns. It is particularly useful when class distribution is uneven.

## Cross Validation
K-fold cross validation splits the dataset into k equal subsets. The model is trained 
k times, each time using a different subset as the validation set and the remaining 
subsets as training data. The final performance metric is the average across all k folds.

Stratified k-fold ensures each fold has the same proportion of class labels as the 
full dataset. This is especially important for imbalanced classification problems.

## Feature Engineering
Feature scaling normalizes the range of input features. StandardScaler transforms 
features to have zero mean and unit variance. MinMaxScaler scales features to a 
fixed range, usually 0 to 1. Many algorithms like SVM and KNN are sensitive to scale.

## Ensemble Methods
Random Forest builds multiple decision trees using bootstrapped samples of the data 
and random subsets of features. Predictions are made by averaging (regression) or 
majority vote (classification). Random Forest is robust to overfitting and handles 
high dimensional data well.

Gradient Boosting builds trees sequentially where each tree corrects the errors of 
the previous one. XGBoost and LightGBM are popular implementations. Gradient boosting 
often achieves higher accuracy than random forests but requires more careful tuning.
"""


# ── Chunking ──────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 150, overlap: int = 20) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Overlap ensures context isn't lost at chunk boundaries.
    Sweet spot for most documents: 150-300 words, 10-15% overlap.
    
    Args:
        text: Raw document text
        chunk_size: Words per chunk
        overlap: Words shared between consecutive chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# ── Vector Store ──────────────────────────────────────────────
def build_collection(documents: list[str], collection_name: str) -> chromadb.Collection:
    """
    Embed documents and store in ChromaDB with cosine similarity.
    
    Note: Always specify hnsw:space=cosine explicitly.
    Default L2 metric produces misleading negative similarity scores.
    """
    collection = db_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    embeddings = embedding_model.encode(documents).tolist()
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(documents))]
    )
    print(f"✅ Stored {collection.count()} chunks in '{collection_name}'")
    return collection


# ── Retrieval ─────────────────────────────────────────────────
def retrieve(query: str, collection: chromadb.Collection, n_results: int = 2):
    """
    Retrieve most relevant chunks for a query.
    
    Returns:
        docs: List of retrieved document chunks
        similarities: Cosine similarity scores (0-1)
    """
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    docs = results['documents'][0]
    similarities = [1 - d for d in results['distances'][0]]
    return docs, similarities


# ── RAG Pipeline ──────────────────────────────────────────────
def rag_answer(question: str, collection: chromadb.Collection) -> dict:
    """
    Full RAG pipeline: question → retrieve → augment → generate.
    
    Args:
        question: User's question
        collection: ChromaDB collection to retrieve from
    
    Returns:
        dict with answer, sources, similarities, token usage
    """
    docs, similarities = retrieve(question, collection)

    context = "\n\n".join([
        f"[Source {i+1} | Relevance: {sim:.2f}]\n{doc}"
        for i, (doc, sim) in enumerate(zip(docs, similarities))
    ])

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        temperature=0,
        system="""You are a precise Data Science tutor.
Answer using ONLY the provided context.
Be specific and cite sources like [Source 1].
If context is insufficient, say exactly what's missing.""",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )

    # Always check stop_reason in production
    if response.stop_reason != "end_turn":
        print(f"⚠️  Warning: stop_reason={response.stop_reason}")

    return {
        "answer": response.content[0].text,
        "sources": docs,
        "similarities": similarities,
        "tokens": response.usage.input_tokens + response.usage.output_tokens
    }


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building knowledge base from ML reference document...\n")

    chunks = chunk_text(ML_DOCUMENT, chunk_size=100, overlap=20)
    collection = build_collection(chunks, "ml_reference_guide")

    test_questions = [
        "What is the difference between L1 and L2 regularization?",
        "How does Random Forest differ from Gradient Boosting?",
        "What metrics should I use for an imbalanced classification problem?"
    ]

    print("\nRunning RAG pipeline...\n")
    for question in test_questions:
        result = rag_answer(question, collection)
        print(f"\n{'='*60}")
        print(f"❓ {question}")
        print(f"{'='*60}")
        print(result["answer"])
        print(f"\n📊 Tokens: {result['tokens']} | "
              f"Top similarity: {result['similarities'][0]:.3f}")
