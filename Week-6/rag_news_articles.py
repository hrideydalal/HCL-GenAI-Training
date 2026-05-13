# ============================================================
# Week 6 - RAG News Articles Q&A
# ============================================================

# ============================================================
# Project Overview
# ============================================================
# In this task, I built a RAG-based chatbot for news articles.
#
# The system allows users to upload news article files such as
# PDF or TXT documents.
#
# It extracts text from the uploaded files, splits the text into
# smaller overlapping chunks, stores those chunks in ChromaDB,
# and uses OpenAI embeddings for semantic search.
#
# When the user asks a question, the system retrieves the most
# relevant chunks from ChromaDB and sends them to an OpenAI model
# to generate a concise answer based only on the retrieved context.
# ============================================================


# =========================
# 1. Install Libraries
# =========================
!pip install -q openai chromadb pypdf


# =========================
# 2. Import Libraries
# =========================
import os
import getpass
import chromadb
from openai import OpenAI
from pypdf import PdfReader
from google.colab import files
from chromadb.utils import embedding_functions


# =========================
# 3. OpenAI API Setup
# =========================
api_key = getpass.getpass("Enter your OpenAI API key: ")

client = OpenAI(api_key=api_key)


# =========================
# 4. Upload News Article Files
# =========================
print("Upload news article files (PDF or TXT):")
uploaded = files.upload()


# =========================
# 5. Extract Text from Files
# =========================
def extract_text_from_file(filename, data):
    """
    Extract text from PDF or TXT files.
    """
    if filename.lower().endswith(".pdf"):
        temp_path = f"/tmp/{filename}"

        with open(temp_path, "wb") as file:
            file.write(data)

        reader = PdfReader(temp_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text

    elif filename.lower().endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    else:
        return ""


documents = []

for filename, data in uploaded.items():
    text = extract_text_from_file(filename, data)

    if text.strip():
        documents.append({
            "source": filename,
            "text": text
        })

print("Total documents loaded:", len(documents))

if len(documents) == 0:
    raise ValueError("No valid text found. Please upload PDF or TXT files with readable text.")


# =========================
# 6. Chunk Text
# =========================
def split_text(text, chunk_size=1000, chunk_overlap=100):
    """
    Split long text into smaller overlapping chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - chunk_overlap

    return chunks


chunked_documents = []

for doc in documents:
    chunks = split_text(doc["text"], chunk_size=1000, chunk_overlap=100)

    for i, chunk in enumerate(chunks, 1):
        chunked_documents.append({
            "id": f"{doc['source']}_chunk_{i}",
            "text": chunk,
            "source": doc["source"]
        })

print("Total chunks created:", len(chunked_documents))

if len(chunked_documents) == 0:
    raise ValueError("No chunks were created from the uploaded documents.")


# =========================
# 7. Setup ChromaDB with OpenAI Embeddings
# =========================
openai_embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small"
)

chroma_client = chromadb.Client()

collection_name = "news_article_qa_collection"

# Delete old collection if it exists, so the file can run again cleanly
try:
    chroma_client.delete_collection(name=collection_name)
except Exception:
    pass

collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=openai_embedding_function
)


# =========================
# 8. Store Chunks in ChromaDB
# =========================
ids = [doc["id"] for doc in chunked_documents]
texts = [doc["text"] for doc in chunked_documents]
metadatas = [{"source": doc["source"]} for doc in chunked_documents]

collection.add(
    ids=ids,
    documents=texts,
    metadatas=metadatas
)

print("Chunks stored in ChromaDB:", collection.count())


# =========================
# 9. Retrieve Relevant Chunks
# =========================
def fetch_relevant_chunks(question, n=3):
    """
    Retrieve most relevant chunks for the question.
    """
    results = collection.query(
        query_texts=[question],
        n_results=n,
        include=["documents", "metadatas"]
    )

    retrieved_chunks = []

    for i, doc_text in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"]

        retrieved_chunks.append({
            "source": source,
            "text": doc_text
        })

    return retrieved_chunks


# =========================
# 10. Generate Answer
# =========================
def answer_question(question, relevant_chunks):
    """
    Generate answer using retrieved context.
    """
    context = ""

    for i, chunk in enumerate(relevant_chunks, 1):
        context += f"\nSource {i}: {chunk['source']}\n"
        context += f"{chunk['text']}\n"

    prompt = f"""
You are a helpful assistant for answering questions from news articles.

Use only the retrieved context below to answer the question.
If the answer is not available in the context, say:
"I don't know based on the uploaded articles."

Keep the answer concise and clear.

Retrieved Context:
{context}

Question:
{question}
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.2
    )

    return response.output_text


# =========================
# 11. Chatbot Loop
# =========================
def chatbot():
    print("\nWelcome to the RAG News Articles Chatbot!")
    print("Ask questions about the uploaded news articles.")
    print("Type 'quit' or 'stop' to exit.\n")

    while True:
        user_question = input("User: ").strip()

        if user_question.lower() in ["quit", "stop", "exit"]:
            print("\nChatbot: Goodbye!")
            break

        relevant_chunks = fetch_relevant_chunks(user_question, n=3)
        answer = answer_question(user_question, relevant_chunks)

        print("\nChatbot:")
        print(answer)
        print()


# =========================
# 12. Run Chatbot
# =========================
chatbot()
