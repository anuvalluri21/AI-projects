# Calculator message output

Both scripts print numbered messages in order after a successful AI response.
The examples below illustrate the output; AI wording, tool-call IDs, and the
agent's number of messages may vary. Both scripts require `OPENAI_API_KEY`.

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
