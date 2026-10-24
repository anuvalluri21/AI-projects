# Electrical issue: vector RAG versus GraphRAG

A Python teaching example using the same three documents and question in both approaches.
Neo4j serves as **both the vector database and the relationship graph**; no separate Chroma service is needed.
Ollama provides local text embeddings and a local chat model. No hosted model API key is required.

## What you will compare

```text
Same question -> Same embedding -> Same Neo4j vector search
                                      |
                  +-------------------+---------------------+
                  |                                         |
           Vector-only RAG                            GraphRAG
           Top-k documents                       Same top-k documents
                  |                              -> linked entities
                  |                              -> upstream dependencies
                  |                              -> supporting documents
                  +-------------------+---------------------+
                                      |
                         Same LLM and system prompt
                                      |
                          Two answers + visible evidence
```

The question is: **Why might the final inspection be delayed?**

| Source | Fact |
| --- | --- |
| D1: Electrical inspection report | The electrical panel failed inspection; the issue is unresolved. |
| D2: Wiring work schedule | Wiring cannot finish until the electrical-panel issue is resolved. |
| D3: Final inspection project plan | Final inspection requires completed wiring. |

```text
Electrical panel issue -[BLOCKS; D2]-> Wiring completion
Wiring completion -[REQUIRED_FOR; D3]-> Final inspection
```

The graph is manually curated from those facts. This demo does not automatically extract an ontology, validate construction schedules, or determine an actual delay duration. GraphRAG here means vector-seeded graph retrieval; it is not Microsoft's community-summary GraphRAG pipeline and does not require the `neo4j-graphrag` library.

## Requirements

- Python 3.10 or later.
- Docker Desktop with Compose, or an existing Neo4j 5.26 database.
- Ollama installed and running on the host at `http://localhost:11434`.
- RAM/disk appropriate for the downloaded local models. The first run may take longer while models load.

## Setup: Windows PowerShell

Open a terminal in this folder after extracting the ZIP.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Replace this demonstration password for your local database.
$env:NEO4J_PASSWORD = 'electrical-demo-password'
docker compose up -d

ollama pull nomic-embed-text
ollama pull llama3.2
```

Wait for Neo4j to be ready. Its browser is available at http://localhost:7474.
Sign in as `neo4j`, using the password above. Existing Docker volumes retain the password originally used to initialize the database; changing the environment variable does not reset it.

If using Neo4j Desktop or Aura instead, skip Docker and set your connection information:

```powershell
$env:NEO4J_URI = 'neo4j://localhost:7687' # Use your actual URI; Aura uses neo4j+s://...
$env:NEO4J_USER = 'neo4j'
$env:NEO4J_PASSWORD = 'your-password'
$env:NEO4J_DATABASE = 'neo4j'
```

The demo creates only its own `ElectricalDemo*` nodes, named constraints, and vector index. It does not delete existing database data. Use a dedicated demo database to avoid unrelated schema conflicts. Model downloads and Docker images require internet access during setup; local inference does not require a hosted API.

## Run both approaches

```powershell
.\.venv\Scripts\python.exe demo.py --top-k 1 --output comparison-k1.json
```

The program embeds the documents and question together, seeds Neo4j, creates/waits for its vector index, and prints:

1. Actual vector-search document IDs and similarity scores.
2. The vector-only evidence context and generated answer.
3. The expanded graph evidence, source-linked paths, and generated answer.

The JSON output contains the same evidence and answers for inspection. Generated answers are not guaranteed to follow the prompt perfectly; verify citations against the printed evidence.

To inspect retrieval without downloading/running the chat model:

```powershell
.\.venv\Scripts\python.exe demo.py --top-k 1 --skip-generation
```

Embeddings and Neo4j are still required for this command.

### Try a fairer, larger retrieval budget

```powershell
.\.venv\Scripts\python.exe demo.py --top-k 3 --output comparison-k3.json
```

With all three documents retrieved, vector-only RAG may produce the same complete explanation. GraphRAG is not guaranteed to outperform vector RAG. At `top-k=1`, the expected lesson is how graph expansion can recover evidence outside the initial vector result, not proof of a general quality advantage. Ranking depends on the embedding model; D3 is not hardcoded as the top result.

The graph-expanded approach can receive more source text than the baseline. This is a retrieval-mechanism illustration, not a token-budget-controlled benchmark. A production evaluation should compare equivalent context budgets, latency, quality, and access controls.

## Core implementation

### Vector-only retrieval

```cypher
CALL db.index.vector.queryNodes($index, $k, $embedding)
YIELD node, score
RETURN node.id, node.text, score
ORDER BY score DESC;
```

The implementation uses Neo4j 5.26-compatible vector procedures. Newer Neo4j versions may recommend different vector-search syntax. See the [vector index documentation](https://neo4j.com/docs/cypher-manual/current/indexes/search-indexes/vector-indexes/).

### Graph expansion

Starting from entities mentioned in the actual vector results:

```cypher
MATCH (d:ElectricalDemoDocument)-[:MENTIONS]->(seed:ElectricalDemoEntity)
WHERE d.id IN $seedIds
MATCH p=(upstream:ElectricalDemoEntity)-[:BLOCKS|REQUIRED_FOR*1..3]->(seed)
RETURN p;
```

This follows **incoming upstream dependencies**, up to three hops. The application retrieves documents mentioning path entities and documents supporting path edges. It passes that evidence plus paths to the same answer generator. If no paths exist, the original seed documents remain available.

For D3 as a seed, the graph can expose the upstream wiring/panel chain. D1 supplies the unresolved issue's status; D2 and D3 support its dependency consequences. The answer should express a potential delay, not invent a confirmed delay or duration.

### Change the local models

```powershell
$env:EMBED_MODEL = 'nomic-embed-text'
$env:CHAT_MODEL = 'llama3.2'
$env:OLLAMA_URL = 'http://localhost:11434'
```

The script refuses a model-name/dimension change against existing demo configuration. Use a fresh dedicated demo database if changing embedding models. Tags can change over time; pin model artifacts and Docker image digests for reproducible production experiments.

## Tests and validation status

Offline tests need only Python's standard library:

```powershell
python -m unittest -v
```

They cover evidence boundaries, source-linked paths, use of actual seed IDs, unchanged generation prompts, invalid embeddings, and full-context baseline coverage. Mocked database responses test orchestration, not actual Cypher execution.

**Authoring validation:** Offline tests and Python compilation were run. Live Neo4j vector queries and Ollama inference could not be run in the authoring environment because Docker, Ollama, and a Neo4j instance were unavailable. No generated comparison output is presented as a measured result.

For a live smoke check, run both `--top-k 1` and `--top-k 3`, inspect the retrieved IDs, and confirm that graph paths cite D2/D3 and expanded evidence includes D1. Stop the local database with `docker compose stop` when finished. This preserves its named volume.

## Troubleshooting

- **Cannot reach Ollama:** Start the Ollama application/service; check `OLLAMA_URL`.
- **Model missing:** Run `ollama pull` for the configured model.
- **Neo4j authentication failure:** Check the password used when the volume was first initialized.
- **Connection refused:** Wait for startup and check the URI/Bolt port.
- **Index procedure or syntax error:** Confirm Neo4j 5.26+ and an account allowed to create indexes/constraints.
- **Unexpected ranking:** Inspect actual scores; try `--top-k 2` or `3`. No ranking is fabricated.
- **Incomplete or overconfident LLM answer:** Inspect retrieved evidence; this demo does not claim automatic factual verification.

## Official references

- [Neo4j Python query execution](https://neo4j.com/docs/python-manual/current/query-simple/)
- [Neo4j vector indexes](https://neo4j.com/docs/cypher-manual/current/indexes/search-indexes/vector-indexes/)
- [Ollama embeddings API](https://docs.ollama.com/api/embed)
- [Ollama chat API](https://docs.ollama.com/api/chat)

The example uses direct Python APIs so the retrieval difference stays visible without a framework abstraction.
