"""Let an AI model choose arithmetic tools for the requested expression."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"), override=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing from .env.")

    agent = create_agent(
        model=ChatOpenAI(
            model="gpt-4.1-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            max_retries=0,
        ),
        tools=[add, multiply],
        system_prompt=(
            "You are a calculator assistant. Always use the provided tools "
            "for arithmetic. Respect parentheses and use earlier tool "
            "results as inputs to later calculations."
        ),
    )
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Calculate (2 + 3) * 5"}]},
        config={"recursion_limit": 10},
    )
    for index, message in enumerate(response["messages"], start=1):
        print(f"\nMessage {index}")
        message.pretty_print()


if __name__ == "__main__":
    main()
