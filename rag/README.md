# Asynchronous PDF knowledge graph

`rag.RagPipeline` handles a 50-page PDF (or any page count). It extracts
each page, splits its text into overlapping character chunks, saves text and
metadata to local folders, generates OpenAI embeddings, and upserts into
Pinecone and Neo4j. This is a document structure graph; entity and semantic
relationship extraction are not included.

```text
(Document)-[:HAS_PAGE]->(Page)-[:HAS_CHUNK]->(Chunk)
(Document)-[:HAS_CHUNK]->(Chunk)
(Chunk)-[:NEXT_CHUNK]->(Chunk)
```

`NEXT_CHUNK` follows page number and chunk number, including across page
boundaries and skipping pages with no text. Asynchronous completion order
does not determine the links. The same chunk ID is used in Neo4j, Pinecone, and
local metadata. Each chunk records its page number, character offsets, and
text file path.

## Setup and run (PowerShell)

Use Python 3.11 or newer and a running Neo4j instance (Desktop, Docker, or Aura).

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r rag/requirements.txt
```

Copy `rag/.env.example` to `rag/.env` and set these values, using your own credentials:

```dotenv
OPENAI_API_KEY=your-api-key-here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password
NEO4J_DATABASE=neo4j
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_HOST=your-index-host
PINECONE_NAMESPACE=pdf_chunks
```

```powershell
& '.\.venv\Scripts\python.exe' -m rag 'C:\path\document.pdf' --page-workers 5 --chunk-workers 8
```

Create a Pinecone dense-vector index with dimension **1536** and cosine metric
for the default embedding model, then copy its host into `.env`. The script
uses an existing index; it does not create one. Other embedding models need
an index with their matching dimension. Do not mix embedding models in the
same namespace, even if their dimensions match.

The script sends extracted chunk text to OpenAI for embeddings and uses API
credits. `--embedding-model` defaults to `text-embedding-3-small`.

To create folders without API calls or database connections:

```powershell
& '.\.venv\Scripts\python.exe' -m rag 'C:\path\document.pdf' --files-only
```

## Folder layout

```text
rag/pdf_data/
  <document-id>/
    source.pdf                     # Snapshot used for extraction
    document.json                  # Status, settings, ordered pages/chunks
    pages/
      page_0001/
        page.txt
        page.json
        chunks/
          chunk_0001.txt
          chunk_0001.json          # IDs, offsets, previous/next references
      page_0002/
        ...
```

Defaults are 1,200 characters per chunk with 200 characters overlap. Override
with `--chunk-size` and `--overlap`. Chunks do not cross page boundaries.
Whitespace-only chunks are skipped. Empty/scanned pages are retained and
listed in `empty_pages`; OCR is not included.

## Concurrency and recovery

Page workers run extraction in threads, each with its own PDF reader.
Within each page, chunk workers save files and request embeddings concurrently.
A global semaphore limits active chunk processing across all pages. Embedding
calls, Pinecone upserts, and Neo4j writes are asynchronous. Page progress can
print out of order.

Document IDs include the PDF checksum, chunk settings, and embedding model.
Re-running the same input/settings uses deterministic IDs and database upserts.
Different settings create a separate document version. Do not run the same
document/settings twice simultaneously. Failed runs can leave partial files
or records: there is no transaction spanning both databases. A document is
marked completed only after all chunks and ordering links succeed. A retry
repairs partial ingestion; it re-embeds chunks and consumes credits again.

## Inspect the graph

```cypher
MATCH (d:Document)-[:HAS_PAGE]->(p:Page)-[:HAS_CHUNK]->(c:Chunk)
WHERE d.status = 'completed'
RETURN d.id, p.page_number, c.chunk_number, c.text_path
ORDER BY p.page_number, c.chunk_number;
```

```cypher
MATCH (a:Chunk)-[:NEXT_CHUNK]->(b:Chunk)
RETURN a.id, b.id LIMIT 25;
```

For semantic search, supply an embedding from the same model to Pinecone's
`index.query`, use the configured namespace and a `document_id` metadata
filter, then use returned chunk IDs to follow Neo4j references. Chunk text is
stored in Pinecone metadata and local text files. Pinecone is eventually
consistent, so freshly upserted vectors may take time to appear in queries.

API references: [Neo4j async driver](https://neo4j.com/docs/api/python-driver/current/async_api.html),
[Pinecone async client](https://sdk.pinecone.io/python/reference/async-pinecone.html),
[pypdf extraction](https://github.com/py-pdf/pypdf/blob/main/docs/user/extract-text.md).

## Offline validation

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s rag/tests -v
```

Tests cover a generated 50-page PDF, overlapping text coverage, bounded chunk
concurrency, stable IDs on rerun, cross-page ordering, empty pages, and failure
status. Database and embedding services are simulated in ordering/failure
tests; live service ingestion requires configured credentials and a PDF.

## Classes and integration

| File | Class | Responsibility |
| --- | --- | --- |
| `settings.py` | `Settings` | Configuration and validation |
| `pdf_extractor.py` | `PDFExtractor` | Page counting and text extraction |
| `text_chunker.py` | `TextChunker` | Overlapping chunks and character offsets |
| `file_store.py` | `FileStore` | PDF snapshots and JSON metadata |
| `async_executor.py` | `AsyncExecutor` | Bounded asynchronous worker pools |
| `embedding_service.py` | `EmbeddingService` | OpenAI embedding requests |
| `neo4j_store.py` | `Neo4jGraphStore` | Nodes and graph relationships |
| `pinecone_store.py` | `PineconeVectorStore` | Vector upserts and dimension checks |
| `stores.py` | `Stores` | Coordinate embeddings, vectors, and graph writes |
| `page_processor.py` | `PageProcessor` | Extract one page and schedule its chunks |
| `chunk_processor.py` | `ChunkProcessor` | Save one chunk and ingest it into stores |
| `pipeline.py` | `RagPipeline` | Integrate extraction, chunk processing, and ordering |

`cli.py` wires the configuration, `Stores`, and `RagPipeline` together.
`__main__.py` provides the `python -m rag` entry point. The root-level
`pdf_knowledge_graph.py` remains a compatibility entry point. Configuration
loads from `rag/.env`, then fills missing values from the repository `.env`;
existing environment variables take precedence. Output defaults to
`rag/pdf_data/`, configurable with `--output`.

The package implements ingestion for RAG. Retrieval and answer generation
are not part of this pipeline.

```python
import asyncio
from pathlib import Path
from rag import Settings, RagPipeline
from rag.stores import Stores

async def run():
    settings = Settings(output=Path("rag/pdf_data"))
    stores = Stores()
    try:
        await stores.open(settings)
        return await RagPipeline(settings, stores).ingest(Path("document.pdf"))
    finally:
        await stores.close()

# Configure environment variables before running this programmatic example.
result = asyncio.run(run())
```

For files only, construct `RagPipeline(settings)` without `Stores`.
Components accept injected replacements so the pipeline can be tested
without external services. Tests live under `rag/tests/`.
