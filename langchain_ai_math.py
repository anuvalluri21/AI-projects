"""Calculate an expression directly with an AI through a LangChain chain."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
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

    # LangChain passes the prompt to the model, then extracts its text response.
    chain = prompt | model | StrOutputParser()
    answer = chain.invoke({"expression": "(2 + 3) * 5"})
    print(answer)


if __name__ == "__main__":
    main()
