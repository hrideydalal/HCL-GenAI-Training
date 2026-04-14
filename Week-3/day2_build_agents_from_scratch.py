# ============================================================
# Week 3 Day 2 - Web Scraping Agent with Memory, State, and Tools
# ============================================================

# ============================================================
# Project Overview
# ============================================================
# In this task, I built an agent from scratch using:
# 1. Memory
# 2. State
# 3. Tool invocation
#
# The agent can:
# - fetch webpage content
# - extract headlines
# - extract email addresses
# - process multiple URLs
# - save daily updates to a text file
#
# This demonstrates how an agent can use tools and maintain
# memory while performing useful tasks.
# ============================================================


# =========================
# 1. Install Libraries
# =========================
!pip install -q requests beautifulsoup4

# =========================
# 2. Import Libraries
# =========================
import requests
import re
from bs4 import BeautifulSoup
from datetime import datetime

# =========================
# 3. Agent Memory and State
# =========================
memory = []

state = {
    "last_action": None,
    "last_result": None
}

# =========================
# 4. Tool 1 - Fetch Webpage
# =========================
def fetch_webpage(url):
    """
    Fetch webpage HTML content.
    """
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error fetching webpage: {e}"

# =========================
# 5. Tool 2 - Extract Headlines
# =========================
def extract_headlines(html):
    """
    Extract headlines from h1, h2, and h3 tags.
    """
    soup = BeautifulSoup(html, "html.parser")

    headlines = []
    seen = set()

    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = tag.get_text(strip=True)
        if text and text not in seen:
            headlines.append(text)
            seen.add(text)

    return headlines[:10]

# =========================
# 6. Tool 3 - Extract Emails
# =========================
def extract_emails(text):
    """
    Extract email addresses using regular expressions.
    """
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails = re.findall(email_pattern, text)

    return list(set(emails))

# =========================
# 7. Save Daily Update to File
# =========================
def save_update_to_file(summary, file_name="daily_update_report.txt"):
    """
    Save summary output to a text file.
    """
    with open(file_name, "a", encoding="utf-8") as file:
        file.write(summary)
        file.write("\n" + "=" * 60 + "\n")

# =========================
# 8. Daily Update Function
# =========================
def daily_update_agent(urls):
    """
    Process multiple URLs and create a daily update summary.
    """
    all_summaries = []

    for url in urls:
        html = fetch_webpage(url)

        if html.startswith("Error"):
            summary = f"""
URL: {url}
Status: Failed
Reason: {html}
"""
        else:
            headlines = extract_headlines(html)
            emails = extract_emails(html)

            summary = f"""
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
URL: {url}
Status: Success

Top Headlines:
"""
            if headlines:
                for i, headline in enumerate(headlines, 1):
                    summary += f"{i}. {headline}\n"
            else:
                summary += "No headlines found.\n"

            summary += "\nEmails Found:\n"
            if emails:
                for i, email in enumerate(emails, 1):
                    summary += f"{i}. {email}\n"
            else:
                summary += "No email addresses found.\n"

        all_summaries.append(summary)

    return all_summaries

# =========================
# 9. Agent Function
# =========================
def agent(user_input, urls=None):
    """
    Main agent function:
    - understands user request
    - uses the required tool
    - updates memory and state
    """
    action = None
    result = None

    user_input_lower = user_input.lower()

    if "headline" in user_input_lower or "news" in user_input_lower:
        action = "extract_headlines"

        if not urls:
            result = "Please provide at least one URL."
        else:
            result = []
            for url in urls:
                html = fetch_webpage(url)
                if html.startswith("Error"):
                    result.append({"url": url, "headlines": html})
                else:
                    result.append({"url": url, "headlines": extract_headlines(html)})

    elif "email" in user_input_lower:
        action = "extract_emails"

        if not urls:
            result = "Please provide at least one URL."
        else:
            result = []
            for url in urls:
                html = fetch_webpage(url)
                if html.startswith("Error"):
                    result.append({"url": url, "emails": html})
                else:
                    result.append({"url": url, "emails": extract_emails(html)})

    elif "daily update" in user_input_lower or "summary" in user_input_lower:
        action = "daily_update"

        if not urls:
            result = "Please provide at least one URL."
        else:
            result = daily_update_agent(urls)

            for summary in result:
                save_update_to_file(summary)

    elif "memory" in user_input_lower:
        action = "show_memory"
        result = memory

    elif "state" in user_input_lower:
        action = "show_state"
        result = state

    else:
        action = "default"
        result = (
            "I can help with:\n"
            "- extracting headlines\n"
            "- extracting email addresses\n"
            "- creating a daily update summary\n"
            "- showing memory and state"
        )

    state["last_action"] = action
    state["last_result"] = result

    memory.append({
        "action": action,
        "urls": urls,
        "result": result
    })

    return result

# =========================
# 10. Chat Loop
# =========================
print("Web Scraping Agent Ready")
print("Available commands:")
print("- extract headlines")
print("- extract emails")
print("- daily update summary")
print("- show memory")
print("- show state")
print("- quit")
print("\nNote: This works best on static HTML pages.")
print("Some websites may block scraping or load content dynamically.")

while True:
    print("\n" + "-" * 60)
    user_input = input("Enter command: ").strip()

    if user_input.lower() == "quit":
        print("Agent stopped.")
        break

    urls = None

    if any(keyword in user_input.lower() for keyword in ["headline", "news", "email", "daily update", "summary"]):
        raw_urls = input("Enter one or more URLs separated by commas: ").strip()
        urls = [url.strip() for url in raw_urls.split(",") if url.strip()]

    output = agent(user_input, urls)

    print("\nOutput:\n")

    if isinstance(output, list):
        for item in output:
            print(item)
            print()
    else:
        print(output)
