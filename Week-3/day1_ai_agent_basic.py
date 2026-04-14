# ============================================================
# Week 3 Day 1 - Basic AI Agent (Tool + Memory)
# ============================================================

# ============================================================
# Project Overview
# ============================================================
# In this task, I built a simple AI agent that can use a tool
# to fetch information and then generate a final answer.
#
# The agent follows a basic loop:
# 1. Read the user query
# 2. Use a tool to fetch information
# 3. Store the interaction in memory
# 4. Generate the final answer
#
# This demonstrates basic agent concepts such as tool usage,
# memory, and simple reasoning.

# Input → Reason → Tool → Output → Repeat
# ============================================================

# =========================
# 1. Install Libraries
# =========================
!pip install -q openai requests

# =========================
# 2. Import Libraries
# =========================
import requests
import getpass
from openai import OpenAI

# =========================
# 3. API Setup
# =========================
api_key = getpass.getpass("Enter OpenAI API Key: ")
client = OpenAI(api_key=api_key)

# =========================
# 4. Memory
# =========================
memory = []

# =========================
# 5. Tool: Fetch Wikipedia Summary
# =========================
def fetch_wikipedia_summary(topic):
    """
    Fetch short summary from Wikipedia API.
    """
    clean_topic = topic.strip().replace("?", "").replace("Who is ", "").replace("What is ", "")
    clean_topic = clean_topic.strip().replace(" ", "_")

    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + clean_topic

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return None, f"No information found for '{topic}'."

        data = response.json()
        summary = data.get("extract")

        if not summary:
            return None, f"No summary available for '{topic}'."

        return summary, None

    except requests.RequestException as e:
        return None, f"Error fetching data: {e}"

# =========================
# 6. Agent Function
# =========================
def agent(user_query):
    """
    Simple agent loop:
    user query -> tool call -> memory -> final answer
    """

    print("\n[Agent is thinking...]")

    tool_output, error = fetch_wikipedia_summary(user_query)

    # Store conversation in memory
    memory.append({
        "query": user_query,
        "tool_output": tool_output if tool_output else error
    })

    if error:
        return error

    prompt = f"""
You are a helpful AI agent.

Use the information below to answer the user's question clearly.

Information:
{tool_output}

Question:
{user_query}
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text

# =========================
# 7. Chat Loop
# =========================
print("Basic AI Agent Ready")
print("Tip: Ask simple topic-style questions such as 'Alan Turing', 'Machine learning', or 'Artificial intelligence'")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "quit":
        print("Done.")
        break

    final_answer = agent(user_input)

    print("\nAgent:")
    print(final_answer)
    print()
