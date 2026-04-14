# ============================================================
# Week 2 Day 1 - RAG Basics (Embeddings + Similarity Search)
# ============================================================

# ============================================================
# Project Overview
# ============================================================
# In this task, I explored the basic concepts of Retrieval-
# Augmented Generation (RAG).
#
# I used Sentence Transformers to convert text passages into
# embeddings (numerical vectors).
#
# Then, I stored these embeddings and performed similarity
# search using FAISS to find the most relevant passages for a
# given query.
#
# This demonstrates how embeddings and similarity search help
# improve LLM accuracy by retrieving relevant context.
# ============================================================


# =========================
# 1. Install Libraries
# =========================
!pip install -q sentence-transformers faiss-cpu


# =========================
# 2. Import Libraries
# =========================
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


# =========================
# 3. Sample Passages
# =========================
# You can replace these with Wikipedia or any text

passages = [
    "Artificial Intelligence is transforming industries worldwide.",
    "Machine learning is a subset of AI that learns from data.",
    "Deep learning uses neural networks with many layers.",
    "Natural Language Processing helps computers understand text.",
    "RAG combines retrieval and generation for better answers.",
    "FAISS is a library for fast similarity search.",
    "Embeddings convert text into numerical vectors.",
    "Transformers are widely used in modern AI models."
]


# =========================
# 4. Load Embedding Model
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# 5. Generate Embeddings
# =========================
print("Generating embeddings...")

embeddings = model.encode(passages)

# Convert to numpy array
embeddings = np.array(embeddings).astype("float32")

print("Embeddings shape:", embeddings.shape)


# =========================
# 6. Store in FAISS
# =========================
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index created with", index.ntotal, "passages")


# =========================
# 7. Similarity Search
# =========================
def search(query, top_k=3):
    query_embedding = model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = [passages[i] for i in indices[0]]
    return results


# =========================
# 8. Test Query
# =========================
query = "What is machine learning?"

results = search(query)

print("\nQuery:", query)
print("\nTop Results:\n")

for i, res in enumerate(results, 1):
    print(f"{i}. {res}")
