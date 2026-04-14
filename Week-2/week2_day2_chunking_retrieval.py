# ============================================================
# Week 2 Day 2 - Chunking and Retrieval using FAISS
# ============================================================

# ============================================================
# Project Overview
# ============================================================
# In this task, I implemented basic chunking and retrieval for a
# simple RAG-style pipeline.
#
# First, I created a small text dataset and split it into smaller
# chunks using a fixed-size chunking approach.
#
# Then, I converted the chunks into embeddings using the
# SentenceTransformer model.
#
# After that, I stored the embeddings in a FAISS vector database
# for similarity search.
#
# Finally, I wrote a retrieve(query, top_k=3) function that
# returns the most relevant chunks for a user query.
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
# 3. Sample Dataset
# =========================
documents = [
    """Artificial Intelligence is transforming many industries.
    Machine learning is a subset of AI that allows systems to learn
    from data. Deep learning is a further subset that uses neural
    networks with many layers.""",

    """Natural Language Processing helps computers understand and
    generate human language. It is widely used in chatbots,
    translation systems, and sentiment analysis.""",

    """Retrieval-Augmented Generation combines information retrieval
    with language generation. It improves answer quality by fetching
    relevant context before generating a response.""",

    """FAISS is a library used for fast similarity search. It is
    commonly used in retrieval systems to compare vector embeddings
    and find the closest matches.""",

    """Chroma is a vector database designed for storing embeddings
    and performing semantic search. It is often used in RAG
    applications and AI assistants."""
]


# =========================
# 4. Chunking Function
# =========================
def chunk_text(text, chunk_size=250, overlap=50):
    """
    Split text into smaller chunks using a sliding window approach.

    chunk_size = number of characters in each chunk
    overlap = shared characters between consecutive chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


# =========================
# 5. Create All Chunks
# =========================
all_chunks = []

for doc in documents:
    chunks = chunk_text(doc, chunk_size=250, overlap=50)
    all_chunks.extend(chunks)

print("Total chunks created:", len(all_chunks))

for i, chunk in enumerate(all_chunks, 1):
    print(f"\nChunk {i}:")
    print(chunk)


# =========================
# 6. Load Embedding Model
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# 7. Generate Embeddings
# =========================
print("\nGenerating embeddings...")
embeddings = model.encode(all_chunks)

# Convert to NumPy float32 for FAISS
embeddings = np.array(embeddings).astype("float32")

print("Embeddings shape:", embeddings.shape)


# =========================
# 8. Store in FAISS
# =========================
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index created with", index.ntotal, "chunks")


# =========================
# 9. Retrieval Function
# =========================
def retrieve(query, top_k=3):
    """
    Convert query into embedding and return top matching chunks.
    """
    query_embedding = model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = [all_chunks[i] for i in indices[0]]
    return results


# =========================
# 10. Test Retrieval
# =========================
query = "How does RAG improve answer quality?"

results = retrieve(query, top_k=3)

print("\nQuery:", query)
print("\nTop Retrieved Passages:\n")

for i, passage in enumerate(results, 1):
    print(f"{i}. {passage}\n")
