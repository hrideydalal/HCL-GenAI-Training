# ============================================================
# Week 2 Day 3 - RAG Pipeline (Retrieval + Generation)
# ============================================================

# ============================================================
# Project Overview
# ============================================================
# In this task, I built a simple RAG (Retrieval-Augmented
# Generation) pipeline.
#
# First, I created a dataset and converted it into chunks.
#
# Then, I generated embeddings using Sentence Transformers
# and stored them in a FAISS vector database.
#
# When a user asks a question, the system retrieves the most
# relevant chunks and inserts them into a prompt.
#
# Finally, the prompt is sent to an OpenAI model to generate
# a final answer based only on the retrieved context.
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
# 4. Sample Dataset
# =========================
documents = [
    "Artificial Intelligence is transforming industries worldwide.",
    "Machine learning is a subset of AI that learns from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural Language Processing helps machines understand text.",
    "RAG improves accuracy by retrieving relevant information before answering.",
    "FAISS is used for fast similarity search.",
    "Embeddings convert text into numerical vectors.",
    "Transformers are widely used in modern AI models."
]


# =========================
# 5. Load Model
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# 6. Generate Embeddings
# =========================
embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype("float32")


# =========================
# 7. Store in FAISS
# =========================
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


# =========================
# 8. Retrieval Function
# =========================
def retrieve(query, top_k=3):
    query_embedding = np.array(model.encode([query])).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    return [documents[i] for i in indices[0]]


# =========================
# 9. Prompt + Answer Function
# =========================
def generate_answer(query):
    context = "\n\n".join(retrieve(query))

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
# 10. Test with Questions
# =========================
questions = [
    "What is machine learning?",
    "How does RAG improve accuracy?",
    "What is FAISS used for?",
    "What are embeddings?",
    "What does NLP do?"
]

for q in questions:
    print("=" * 50)
    print("Question:", q)
    print("\nAnswer:")
    print(generate_answer(q))


# User Question
#     ↓
# Retrieve relevant chunks
#     ↓
# Put chunks into prompt
#     ↓
# Send to LLM
#     ↓
# Generate final answer
