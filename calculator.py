"""Calculate (2 + 3) * 5 using LangChain tools and a LangGraph workflow."""

from typing import TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


class CalculatorState(TypedDict):
    a: int
    b: int
    factor: int
    subtotal: int
    result: int


def add_step(state: CalculatorState) -> dict[str, int]:
    subtotal = add.invoke({"a": state["a"], "b": state["b"]})
    print(f"add({state['a']}, {state['b']}) = {subtotal}")
    return {"subtotal": subtotal}


def multiply_step(state: CalculatorState) -> dict[str, int]:
    result = multiply.invoke({"a": state["subtotal"], "b": state["factor"]})
    print(f"multiply({state['subtotal']}, {state['factor']}) = {result}")
    return {"result": result}


def build_workflow():
    graph = StateGraph(CalculatorState)
    graph.add_node("add_step", add_step)
    graph.add_node("multiply_step", multiply_step)
    graph.add_edge(START, "add_step")
    graph.add_edge("add_step", "multiply_step")
    graph.add_edge("multiply_step", END)
    return graph.compile()


if __name__ == "__main__":
    workflow = build_workflow()
    final_state = workflow.invoke(
        {"a": 2, "b": 3, "factor": 5, "subtotal": 0, "result": 0}
    )
    print(f"(2 + 3) * 5 = {final_state['result']}")
