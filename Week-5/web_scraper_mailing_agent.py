# ============================================================
# Week 5 Project - Web Scraper and Mailing Agent
# ============================================================

# ============================================================
# Project Overview
# ============================================================
# In this project, I built a web scraper and mailing agent.
#
# The agent can:
# 1. Fetch latest news headlines for a topic
# 2. Summarize the news using an OpenAI model
# 3. Send the summary to an email address
#
# The project uses tool calling, Pydantic validation, web scraping,
# and SMTP email sending.
#
# Note:
# This code is designed for Google Colab.
# API keys and email credentials should not be hardcoded.
# ============================================================


# =========================
# 1. Install Libraries
# =========================
!pip install -q openai requests beautifulsoup4 pydantic


# =========================
# 2. Import Libraries
# =========================
import json
import requests
import smtplib
import ssl

from bs4 import BeautifulSoup
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from google.colab import userdata


# =========================
# 3. API and Email Setup
# =========================
# Store these values in Google Colab Secrets:
# OPENAI_API_KEY
# SENDER_EMAIL
# SENDER_PASSWORD

OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")
SENDER_EMAIL = userdata.get("SENDER_EMAIL")
SENDER_PASSWORD = userdata.get("SENDER_PASSWORD")

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# 4. Tool Input Schemas
# =========================
class NewsInput(BaseModel):
    topic: str = Field(..., description="Topic to search news for")


class EmailInput(BaseModel):
    to: str = Field(..., description="Recipient email address")
    subject: str = Field(..., description="Email subject")
    message_text: str = Field(..., description="Email body content")


# =========================
# 5. Tool 1 - Get News Headlines
# =========================
def get_news(topic: str) -> List[str]:
    """
    Fetch latest news headlines from Google News.
    """
    url = f"https://news.google.com/search?q={topic}&hl=en-IN&gl=IN&ceid=IN:en"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        headlines = []

        for item in soup.select("a"):
            text = item.get_text(strip=True)
            if text and len(text) > 20:
                headlines.append(text)

        # Remove duplicate headlines
        unique_headlines = []
        seen = set()

        for headline in headlines:
            if headline not in seen:
                unique_headlines.append(headline)
                seen.add(headline)

        return unique_headlines[:5]

    except Exception as e:
        return [f"Error fetching news: {e}"]


# =========================
# 6. Tool 2 - Send Email
# =========================
def send_email(to: str, subject: str, message_text: str):
    """
    Send email using Gmail SMTP.
    """
    port = 465
    smtp_server = "smtp.gmail.com"

    if not SENDER_EMAIL or not SENDER_PASSWORD:
        return "Email credentials are missing. Please add them in Colab Secrets."

    context = ssl.create_default_context()

    try:
        message = f"Subject: {subject}\n\n{message_text}"

        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to, message)

        return "Email sent successfully."

    except Exception as e:
        return f"Error sending email: {e}"


# =========================
# 7. Tool Definitions for OpenAI
# =========================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Fetch latest news headlines for a given topic",
            "parameters": NewsInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email with a subject and message",
            "parameters": EmailInput.model_json_schema()
        }
    }
]


# =========================
# 8. Agent Function
# =========================
def run_agent(user_prompt):
    """
    Agent loop:
    1. Read user request
    2. Call news tool
    3. Summarize results
    4. Call email tool
    5. Return final response
    """

    messages = [
        {
            "role": "system",
            "content": """
You are a web scraper and mailing agent.

Follow these steps:
1. Fetch latest news for the requested topic.
2. Summarize the news clearly.
3. Include possible impact on the stock market and economy sectors.
4. Send the summary to the email address provided by the user.
5. Do not skip the required tool steps.
"""
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"\nTool Call: {tool_name}")
                print("Arguments:", tool_args)

                if tool_name == "get_news":
                    validated_args = NewsInput(**tool_args)
                    result = get_news(**validated_args.model_dump())

                elif tool_name == "send_email":
                    validated_args = EmailInput(**tool_args)
                    result = send_email(**validated_args.model_dump())

                else:
                    result = "Unknown tool requested."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

        else:
            print("\nFinal Response:\n")
            print(message.content)
            break


# =========================
# 9. Main Function
# =========================
if __name__ == "__main__":
    user_input = input("Enter news topic and recipient email: ")
    run_agent(user_input)
