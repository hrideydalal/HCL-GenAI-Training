# ============================================================
# Week 2 Day 4 - Hybrid Retrieval for RAG
# ============================================================

# ============================================================
# Project Overview
# ============================================================
# In this task, I improved a basic RAG pipeline by adding
# hybrid retrieval.
#
# Instead of using only vector search, I combined:
# 1. Vector similarity search using embeddings
# 2. Keyword-based search using word overlap
#
# Then, I compared the hybrid retrieval results with plain
# vector search and generated final answers for both methods.
#
# This helps reduce irrelevant retrieval results and improves
# answer quality.
# ============================================================


# =========================
# 1. Install Libraries
# =========================
!pip install -q sentence-transformers faiss-cpu openai


# =========================
# 2. Import Libraries
# =========================
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from openai import OpenAI
import getpass


# =========================
# 3. OpenAI API Setup
# =========================
api_key = getpass.getpass("Enter OpenAI API Key: ")
client = OpenAI(api_key=api_key)


# =========================
# 4. Sample Documents
# =========================
documents = [
    "Artificial Intelligence is transforming industries worldwide.",
    "Machine learning is a subset of AI that learns from data.",
    "Deep learning uses neural networks with many layers.",
    "Natural Language Processing helps computers understand text.",
    "RAG improves answer quality by retrieving relevant context first.",
    "FAISS is used for fast vector similarity search.",
    "Embeddings convert text into numerical vectors.",
    "BM25 is a keyword-based retrieval method.",
    "Hybrid retrieval combines semantic search and keyword search.",
    "Reranking improves retrieval by reordering the most relevant results."
]


# =========================
# 5. Load Embedding Model
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# 6. Generate Embeddings
# =========================
doc_embeddings = model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype("float32")


# =========================
# 7. Store in FAISS
# =========================
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)


# =========================
# 8. Vector Search Function
# =========================
def vector_search(query, top_k=3):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        score = float(distances[0][rank])
        results.append((documents[idx], score))

    return results


# =========================
# 9. Keyword Search Function
# =========================
def keyword_score(query, document):
    """
    Simple keyword score based on overlapping words.
    """
    query_words = set(query.lower().split())
    doc_words = set(document.lower().split())

    overlap = query_words.intersection(doc_words)
    return len(overlap)


# =========================
# 10. Hybrid Search Function
# =========================
def hybrid_search(query, top_k=3, alpha=0.5):
    """
    alpha controls balance:
    alpha = weight for vector search
    (1 - alpha) = weight for keyword search
    """

    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, len(documents))

    vector_scores = {}

    # Convert distance into similarity-like score
    for rank, idx in enumerate(indices[0]):
        distance = distances[0][rank]
        vector_scores[idx] = 1 / (1 + distance)

    keyword_scores = {}
    for i, doc in enumerate(documents):
        keyword_scores[i] = keyword_score(query, doc)

    # Normalize keyword scores
    max_keyword = max(keyword_scores.values()) if max(keyword_scores.values()) > 0 else 1

    hybrid_results = []
    for i, doc in enumerate(documents):
        normalized_keyword = keyword_scores[i] / max_keyword
        hybrid_score = alpha * vector_scores.get(i, 0) + (1 - alpha) * normalized_keyword
        hybrid_results.append((doc, hybrid_score, vector_scores.get(i, 0), normalized_keyword))

    # Sort by hybrid score in descending order
    hybrid_results.sort(key=lambda x: x[1], reverse=True)

    return hybrid_results[:top_k]


# =========================
# 11. Answer Generation Function
# =========================
def generate_answer(query, retrieved_docs):
    context = "\n\n".join([doc for doc in retrieved_docs])

    prompt = f"""
Use only the context below to answer the question.
If the answer is not found, say "I don't know".

Context:
{context}

Question:
{query}
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text


# =========================
# 12. Compare Plain Vector vs Hybrid Retrieval
# =========================
query = "How does hybrid retrieval improve RAG?"

print("=" * 60)
print("Query:", query)
print("=" * 60)

# Plain vector search
vector_results = vector_search(query, top_k=3)

print("\nPlain Vector Search Results:\n")
vector_docs = []

for i, (doc, score) in enumerate(vector_results, 1):
    vector_docs.append(doc)
    print(f"{i}. Score: {score:.4f}")
    print(doc)
    print()

vector_answer = generate_answer(query, vector_docs)

print("Final Answer using Plain Vector Search:\n")
print(vector_answer)


# Hybrid search
hybrid_results = hybrid_search(query, top_k=3, alpha=0.5)

print("\n" + "=" * 60)
print("Hybrid Search Results:\n")

hybrid_docs = []

for i, (doc, hybrid_score, vector_score, keyword_score_value) in enumerate(hybrid_results, 1):
    hybrid_docs.append(doc)
    print(f"{i}. Hybrid Score: {hybrid_score:.4f} | Vector Score: {vector_score:.4f} | Keyword Score: {keyword_score_value:.4f}")
    print(doc)
    print()

hybrid_answer = generate_answer(query, hybrid_docs)

print("Final Answer using Hybrid Search:\n")
print(hybrid_answer)
