# ============================================================
# Week 4 Day 1 - AI Test Case Generator Agent
# ============================================================

# ============================================================
# Project Overview
# ============================================================
# In this task, I built an AI Test Case Generator Agent using
# LangChain and a ReAct-style agent flow.
#
# The agent takes business requirements as input and generates
# software test cases.
#
# It uses tools to:
# 1. Extract important requirement points
# 2. Identify test scenarios
# 3. Generate structured test cases
#
# This demonstrates how an agent can reason, use tools, and
# produce useful QA/testing outputs from business requirements.
# ============================================================


# =========================
# 1. Install Libraries
# =========================
!pip install -q langchain langchain-openai openai


# =========================
# 2. Import Libraries
# =========================
import getpass
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


# =========================
# 3. OpenAI API Setup
# =========================
api_key = getpass.getpass("Enter OpenAI API Key: ")


# =========================
# 4. Define Tools
# =========================
@tool
def extract_requirement_points(requirement: str) -> str:
    """
    Extract important functional points from a business requirement.
    """
    return f"""
Important requirement points:
- Identify the main user action
- Identify input fields
- Identify expected system behavior
- Identify validation rules
- Identify error scenarios
- Identify success scenarios

Requirement:
{requirement}
"""


@tool
def identify_test_scenarios(requirement: str) -> str:
    """
    Identify positive, negative, and edge test scenarios.
    """
    return f"""
Possible test scenarios:
1. Positive scenario where the user enters valid data.
2. Negative scenario where required fields are missing.
3. Negative scenario where invalid data is entered.
4. Edge scenario with boundary values.
5. UI validation scenario.
6. System response/error message scenario.

Requirement:
{requirement}
"""


@tool
def generate_test_cases(requirement: str) -> str:
    """
    Generate structured test cases from a business requirement.
    """
    return f"""
Generate test cases in this format:

Test Case ID:
Title:
Preconditions:
Test Steps:
Test Data:
Expected Result:
Priority:

Business Requirement:
{requirement}
"""


# =========================
# 5. Create LLM
# =========================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=api_key
)


# =========================
# 6. Create Agent
# =========================
tools = [
    extract_requirement_points,
    identify_test_scenarios,
    generate_test_cases
]

system_prompt = """
You are a QA Test Case Generator Agent.

Your job is to read business requirements and generate clear,
structured, beginner-friendly software test cases.

Use the available tools when useful.
Think in a ReAct-style flow:
- understand the requirement
- identify scenarios
- generate test cases

Generate test cases for:
- positive scenarios
- negative scenarios
- edge cases

Keep the final output clean and structured.
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)


# =========================
# 7. Sample Business Requirement
# =========================
business_requirement = """
A user should be able to log in to the application using a valid
email address and password. If the credentials are correct, the user
should be redirected to the dashboard. If the credentials are incorrect,
an error message should be displayed. Email and password fields are
mandatory.
"""


# =========================
# 8. Run Agent
# =========================
user_prompt = f"""
Create test cases for the following business requirement:

{business_requirement}
"""

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": user_prompt
        }
    ]
})


# =========================
# 9. Print Final Output
# =========================
print("\nGenerated Test Cases:\n")
print(response["messages"][-1].content)
