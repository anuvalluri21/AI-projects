"""Calculate an expression directly with an AI through a LangChain chain."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"), override=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing from .env.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Calculate the expression. Show brief calculation steps and the final answer."),
        ("human", "Calculate {expression}"),
    ])

    model = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        max_retries=0,
    )

    # Keep the message objects so the full exchange can be printed in order.
    prompt_value = prompt.invoke({"expression": "(2 + 3) * 5"})
    answer = model.invoke(prompt_value)
    messages = [*prompt_value.to_messages(), answer]
    for index, message in enumerate(messages, start=1):
        print(f"\nMessage {index}")
        message.pretty_print()


if __name__ == "__main__":
    main()
