# Calculator message output

Both scripts print numbered messages in order after a successful AI response.
The examples below illustrate the output; AI wording, tool-call IDs, and the
agent's number of messages may vary. Both scripts require `OPENAI_API_KEY`.

## Differences between the scripts

| Feature | `ai_calculator.py` | `langchain_ai_math.py` |
| --- | --- | --- |
| Approach | AI agent with Python arithmetic tools | Prompt passed directly to an AI model |
| Who calculates? | AI selects tool calls; Python performs addition and multiplication | AI produces the calculation steps and answer directly |
| Execution flow | Can involve several model calls and tool executions | One model invocation |
| Tools | `add` and `multiply` | No tools |
| Printed messages | Human request, AI tool calls, tool results, and final AI answer | System prompt, human request, and AI response |
| System prompt printed? | No; it is not in the returned conversation list | Yes |
| Output timing | All returned messages print after the agent finishes | All messages print after the model responds |
| Inspectability | Shows tool arguments, call IDs, and Python results | Shows the calculation steps written by the AI |
| Model | `gpt-4.1-mini` | `gpt-4.1-mini` |
| API key required? | Yes | Yes |

For `(2 + 3) * 5`, the agent typically requests `add(2, 3)` and then
`multiply(5, 5)`. The direct model script generates the answer as text.
Python tools compute their supplied arguments deterministically, but the AI
still chooses the operations and arguments.

The updated `langchain_ai_math.py` keeps the AI message object for printing.
It no longer uses `StrOutputParser`, which extracts response text rather than
parsing or evaluating an arithmetic expression.

## `langchain_ai_math.py`

This script prints the system prompt, human request, and AI response. It has no
tool calls or tool-result messages.

```text
Message 1 — System
Calculate the expression. Show brief calculation steps and the final answer.

Message 2 — Human
Calculate (2 + 3) * 5

Message 3 — AI
2 + 3 = 5
5 × 5 = 25
The final answer is 25.
```

The system and human text above match the code. The AI response is an example,
not a captured live response. `pretty_print()` displays message-type headers.

## `ai_calculator.py`

This script prints the returned conversation messages: human request, AI tool
calls, tool results, and final AI answer. Its system prompt is not included in
the returned message list. A typical exchange is:

```text
Message 1 — Human
Calculate (2 + 3) * 5

Message 2 — AI
Tool call: add
Arguments: a=2, b=3

Message 3 — Tool
5

Message 4 — AI
Tool call: multiply
Arguments: a=5, b=5

Message 5 — Tool
25

Message 6 — AI
(2 + 3) * 5 = 25
```

This is an illustrative exchange, not a captured live response. The actual
formatted output also includes tool-call IDs and message-type headers.
