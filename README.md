# AI calculator examples

For asynchronous PDF ingestion into Neo4j and Pinecone, see
[PDF_KNOWLEDGE_GRAPH.md](PDF_KNOWLEDGE_GRAPH.md).

The standalone `rag_single_file.py` combines all RAG ingestion classes and the
CLI in one file. After installing `rag/requirements.txt`, run:

```powershell
& '.\.venv\Scripts\python.exe' rag_single_file.py 'C:\path\document.pdf'
```

Add `--files-only` to skip embeddings and database connections. The standalone
version loads `rag/.env` when available, then fills missing configuration from
the `.env` beside the script. Its default output is `pdf_data/` beside the
script. It needs the same external libraries and services as the modular
version, but does not import any application code from `rag/`.

Three Python examples calculate `(2 + 3) * 5`:

- `calculator.py`: LangChain arithmetic tools run in a fixed LangGraph workflow. No API key needed.
- `ai_calculator.py`: an AI agent chooses the add and multiply tools, using LangChain's LangGraph-backed agent.
- `langchain_ai_math.py`: a prompt, AI model, and text parser form a LangChain chain.

## Setup (PowerShell)

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and replace the placeholder with your OpenAI API key. `.env` is ignored by Git. AI examples require separately billed OpenAI API credits.

## Run

```powershell
& '.\.venv\Scripts\python.exe' calculator.py
& '.\.venv\Scripts\python.exe' ai_calculator.py
& '.\.venv\Scripts\python.exe' langchain_ai_math.py
```

The fixed workflow prints `25`. AI responses may vary in wording.

## Validation

The fixed workflow ran successfully and returned `25`. AI script syntax was checked; the agent API request returned HTTP 429 `credit_balance_exhausted`, so a successful AI response has not yet been verified.
