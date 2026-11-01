# Pinecone and Neo4j hybrid document search

See [the end-to-end walkthrough](END_TO_END_FLOW.md) for every project file,
the ingestion and retrieval flow, and the worked four-chunk policy example.

Retrieve a relevant chunk with Pinecone, then traverse entities across all
chunks of the same document in Neo4j to collect supporting evidence.

The fictional policy in `data/security_policy.txt` has four chunks: contractor
classification, customer data classification, device restrictions, and exception
approval. Each graph fact points to its source chunk.

## Setup in PowerShell

From the repository root:

```powershell
cd pinecone_neo4j_hybrid_search
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Use Python 3.10 or newer. If PyTorch installation reports a Windows path-length
error, create a virtual environment at a shorter path, such as
`$env:USERPROFILE\Documents\Codex\.venvs\hybrid1024`, and use its
`Scripts\python.exe` for the same install and run commands. In Pinecone, use a **document index** whose schema has a dense `embedding` field with
dimension **1024** and cosine metric, without integrated embeddings. Fill in its index host
and your API key in `.env`. This project uses the Pinecone 10.x SDK documents API. The existing index
uses a schema with `embedding` as a 1024-dimensional dense field.

Use a Neo4j Aura instance or Neo4j 5.28 or newer. Set its URI, username,
password, and database name in `.env`. For a local instance, the URI can be
`neo4j://localhost:7687`. The Neo4j account needs schema and write permissions.
The repository ignores `.env`; keep credentials out of Git.

The BGE Large model downloads on first use (approximately 1.3 GB).
Its cache defaults to the ignored `.model_cache/` folder. On Windows, set
`EMBEDDING_CACHE_DIR` in `.env` to a shorter path if necessary. Query embeddings
include the BGE retrieval instruction; document embeddings do not. Pinecone and Neo4j usage follows
your service account pricing.

## Run

```powershell
& .\.venv\Scripts\python.exe hybrid_search.py ingest
& .\.venv\Scripts\python.exe hybrid_search.py check
& .\.venv\Scripts\python.exe hybrid_search.py search
& .\.venv\Scripts\python.exe hybrid_search.py search --question "Who approves device restriction exceptions?"
```

`ingest` writes four vectors to Pinecone and six directed facts to Neo4j.
Pinecone is eventually consistent. Ingestion waits up to 60 seconds for fetch
visibility; search visibility may lag further, so retry an empty search after
a few seconds. `check` verifies vector records and a three-hop Neo4j path.

## Retrieval flow

1. Read all chunks and ingest the curated entities and facts into Neo4j.
2. Embed each chunk with `BAAI/bge-large-en-v1.5` and store it in Pinecone.
3. Embed the question and retrieve the top chunk. The namespace and metadata
   filters restrict the search to this document and embedding model.
4. Link entities in both the question and vector hits to graph nodes.
5. Traverse one to three Neo4j edges. Every node and edge in the path must
   match the document ID. Return stored fact direction and source chunk IDs.
6. Combine the vector and graph evidence and fetch source text from Neo4j.

The JSON output includes vector scores, seed entities, graph facts, graph-added
chunk IDs, and final context. Scores are similarities, not confidence estimates.
The code returns evidence; it does not generate an LLM answer.

## Graph schema

```text
(PolicyChunk {document_id, chunk_id, text})-[:MENTIONS]->(PolicyEntity)
(PolicyEntity {document_id, name})
    -[:POLICY_FACT {document_id, predicate, chunk_id}]->(PolicyEntity)
```

The `TRAVERSE` constant contains the Cypher query. Undirected paths help discover
evidence, while `startNode` and `endNode` retain each fact's original direction.

One seed reaches the approver in three hops:

```text
customer data -> restricted data -> device restriction -> Security Director
     chunk 24           chunk 51              chunk 79
```

The default question also seeds the contractor entity, so final context should
contain all four chunks even with `--top-k 1`. A supported answer is: downloads
require a company-managed device, unless the Security Director approves an
exception beforehand. Personal ownership alone does not establish management
status. Cite chunks 8, 24, 51, and 79 when explaining contractor applicability.

## Validation

Run the offline tests without service credentials or extra dependencies:

```powershell
python -m unittest discover -s tests -v
```

Live validation completed with the existing 1024-dimensional document index:
four chunks ingested, the Neo4j three-hop check passed, and search retrieved
chunk 51 from Pinecone plus chunks 8, 24, and 79 through graph expansion.

The working Windows runtime for this checkout is:

```powershell
& "$env:USERPROFILE\Documents\Codex\.venvs\hybrid1024\Scripts\python.exe" hybrid_search.py search
```

These tests cover document parsing, entity linking, document filtering, and
evidence assembly using service test doubles. They do not validate live SDK
calls, embeddings, or Cypher execution. Use `ingest`, `check`, and `search` with
your configured instances for live verification.

## Scope

`FACTS` and `ALIASES` are curated for this sample, not automatically extracted.
For other documents, add reviewed extraction and entity linking with provenance.
Traversal is document scoped. Bound and rank expansion for larger graphs.

Ingestion is idempotent for this fixed sample, but does not remove obsolete
facts after edits. Use a new document/version ID and namespace for revisions.
Writes across the two services are not atomic; rerun ingestion if it fails.
This project does not delete database data or read other projects' credentials.

## References

- [Pinecone Python SDK](https://sdk.pinecone.io/python/reference/sync-index.html)
- [Neo4j Python driver](https://neo4j.com/docs/python-manual/current/query-simple/)
- [Sentence Transformers](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)
