# ============================================================
# RAG Q&A Chatbot Application
# ============================================================

# ============================================================
# Project Overview
# ============================================================
# In this task, I built a simple Q&A chatbot application using
# a Retrieval-Augmented Generation (RAG) pipeline.
#
# The system allows users to upload PDF, TXT, and Markdown files.
# It extracts text from the uploaded documents and splits the text
# into smaller chunks for easier retrieval.
#
# Each chunk is converted into embeddings using the OpenAI
# Embeddings API and stored in a FAISS vector database.
#
# When the user asks a question, the system retrieves the most
# relevant chunks and sends them to the OpenAI model as context.
#
# Finally, the model generates a natural language answer based
# on the retrieved document content.
# ============================================================


# =========================
# 1. Install Libraries (Colab only)
# =========================
!pip install -q openai faiss-cpu pdfplumber numpy

# =========================
# 2. Import Libraries
# =========================
import os
import getpass
import textwrap
import numpy as np
import pdfplumber
import faiss
from openai import OpenAI
from google.colab import files

# =========================
# 3. Enter API Key
# =========================
api_key = getpass.getpass("Enter your OpenAI API key: ")
client = OpenAI(api_key=api_key)

# =========================
# 4. Upload Files
# =========================
print("Upload your documents:")
uploaded = files.upload()

# =========================
# 5. Extract Text
# =========================
def extract_text(filename, data):
    if filename.lower().endswith(".pdf"):
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "wb") as f:
            f.write(data)

        text = ""
        with pdfplumber.open(temp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    else:
        return data.decode("utf-8", errors="ignore")

# =========================
# 6. Chunking
# =========================
def split_chunks(text, size=500):
    chunks = []
    for i in range(0, len(text), size):
        chunk = text[i:i+size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# =========================
# 7. Process Files
# =========================
all_chunks = []

for filename, data in uploaded.items():
    text = extract_text(filename, data)
    chunks = split_chunks(text)
    all_chunks.extend(chunks)

print("Total chunks:", len(all_chunks))

# =========================
# 8. Create Embeddings
# =========================
print("Creating embeddings...")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=all_chunks
)

vectors = np.array([item.embedding for item in response.data], dtype="float32")

# =========================
# 9. Store in FAISS
# =========================
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

print("FAISS index ready")

# =========================
# 10. Retrieval Function
# =========================
def retrieve(question, k=4):
    q_embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    q_vector = np.array([q_embed.data[0].embedding], dtype="float32")
    distances, indices = index.search(q_vector, k)

    return [all_chunks[i] for i in indices[0]]

# =========================
# 11. Answer Generation
# =========================
def answer(question):
    context = "\n\n".join(retrieve(question))

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Answer using only the provided context. If the answer is not present, say 'I don't know based on the uploaded documents.'"
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.2
    )

    return response.output_text

# =========================
# 12. Chat Loop
# =========================
print("\nReady! Ask questions about your uploaded documents.")
print("Type 'quit' to exit.\n")

while True:
    q = input("You: ").strip()

    if q.lower() == "quit":
        print("Done.")
        break

    result = answer(q)

    print("\nAssistant:")
    print(textwrap.fill(result, 100))
    print()

# =========================
# 13. Performance Notes
# =========================
# This chatbot was tested by asking multiple factual questions
# from the uploaded documents.
#
# Observations:
# - It works well when the relevant chunk is retrieved correctly.
# - Answer quality depends on chunk size and document quality.
# - PDF extraction quality can affect final results.
# - Broad or unclear questions may return incomplete answers.
