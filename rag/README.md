# Asynchronous PDF knowledge graph

`rag.RagPipeline` handles a 50-page PDF (or any page count). It extracts
each page, splits its text into overlapping character chunks, saves text and
metadata to local folders, generates OpenAI embeddings, and upserts into
Pinecone and Neo4j. It also renders every page as a PNG and uses Trustcall's
LangGraph-based extractor to build a page-scoped entity/relationship graph.

```text
(Document)-[:HAS_PAGE]->(Page)-[:HAS_CHUNK]->(Chunk)
(Document)-[:HAS_CHUNK]->(Chunk)
(Chunk)-[:NEXT_CHUNK]->(Chunk)
(Page)-[:MENTIONS]->(PageEntity)
(Page)-[:HAS_RELATION]->(PageRelation)
(PageRelation)-[:SUBJECT]->(PageEntity)
(PageRelation)-[:OBJECT]->(PageEntity)
```

`NEXT_CHUNK` follows page number and chunk number, including across page
boundaries and skipping pages with no text. Asynchronous completion order
does not determine the links. The same chunk ID is used in Neo4j, Pinecone, and
local metadata. Each chunk records its page number, character offsets, and
text file path, source page PNG path, and page knowledge JSON path.

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
Every page image and its extracted text are also sent to the vision-capable
`--entity-model` (default `gpt-4.1-mini`) through Trustcall, using API credits.
Use `--skip-entities` to retain page PNGs and normal chunk ingestion without
making page-knowledge model calls. `--files-only` skips all model/database calls.

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
        page.png                   # Full page render, including diagrams/images
        knowledge.json             # Validated entities/relations or explicit skipped/failed status
        chunks/
          chunk_0001.txt
          chunk_0001.json          # IDs, offsets, previous/next references
      page_0002/
        ...
```

Defaults are 1,200 characters per chunk with 200 characters overlap. Override
with `--chunk-size` and `--overlap`. Chunks do not cross page boundaries.
Whitespace-only chunks are skipped. Empty/scanned pages are retained and
listed in `empty_pages` if no text is recovered. With Trustcall enabled, pages
without machine-readable text use model-transcribed image text for chunking.
In files-only or skip-entities mode there is no image transcription. Model
transcription and extracted facts need review; schema validity does not prove
factual correctness. The original PNG is retained for inspection.

## Concurrency and recovery

Page workers run extraction in threads, each with its own PDF reader.
PNG rendering uses a separate pool of at most two processes because PyMuPDF
does not support concurrent use in threads. Default resolution is 120 DPI,
configurable with `--render-dpi` (72–300); image longest side is capped at
4096 pixels. Rendering processes are closed even when page processing fails.
At most `--page-workers` pages can perform Trustcall extraction concurrently.
Trustcall repairs schema validation errors using its LangGraph workflow;
the graph has a recursion limit of 12. Invalid final extraction fails ingestion
instead of being silently recorded as an empty successful graph.
Within each page, chunk workers save files and request embeddings concurrently.
A global semaphore limits active chunk processing across all pages. Embedding
calls, Pinecone upserts, and Neo4j writes are asynchronous. Page progress can
print out of order.

Document IDs include the PDF checksum, chunk settings, embedding model,
render resolution, entity-extraction setting and model, and an ID-scheme version.
Re-running the same input/settings uses deterministic IDs and database upserts.
Different settings create a separate document version. Do not run the same
document/settings twice simultaneously. Failed runs can leave partial files
or records: there is no transaction spanning both databases. A document is
marked completed only after all chunks and ordering links succeed. A retry
repairs partial ingestion; it re-embeds chunks and consumes credits again.
On retry, Neo4j replaces each page's extracted entity graph in a single
transaction so stale facts are not accumulated. Entities are deliberately
page-scoped; identical names on two pages remain separate nodes. No cross-page
entity resolution is performed. Relation predicates are stored as properties
of `PageRelation` nodes, not interpolated into Cypher relationship types.

## Reference page images and knowledge later

`page.json` and the Neo4j Page node contain `image_path`, `knowledge_path`,
`knowledge_status`, and extraction counts. `knowledge.json` contains entity
names/types, relationship endpoints, source evidence, model name, and model-
reported confidence. Chunk metadata in files, Pinecone, and Neo4j contains
`page_image_path`, `page_knowledge_path`, and `page_id`.

After vector retrieval, use a returned chunk's `page_id`:

```python
context = await stores.page_context(chunk_metadata["page_id"])
image_path = context["image_path"]
entities = context["entities"]
relationships = context["relationships"]
```

This helper retrieves graph provenance; it does not automatically call a
question-answering LLM. A future vision answer tool can load the PNG at
`image_path` when the relevant diagram or page layout is needed. Absolute
paths refer to the ingestion host, so another host needs shared storage or
a path mapping. Page contents are treated as data, never as tool instructions.

```cypher
MATCH (p:Page)-[:HAS_RELATION]->(r:PageRelation)
MATCH (r)-[:SUBJECT]->(a:PageEntity), (r)-[:OBJECT]->(b:PageEntity)
RETURN p.page_number, p.image_path, a.name, r.predicate, b.name, r.evidence;
```

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
Trustcall reference: [official repository and examples](https://github.com/hinthornw/trustcall).

## Offline validation

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s rag/tests -v
```

Tests cover a generated 50-page PDF, overlapping text coverage, bounded chunk
concurrency, stable IDs on rerun, cross-page ordering, empty pages, and failure
status. Database and embedding services are simulated in ordering/failure
tests; live service ingestion requires configured credentials and a PDF.
Additional tests render 50 genuine page PNGs, run an actual Trustcall graph
with a fake tool-calling model, validate relationship endpoints, retain PNGs
on extraction failure, and verify image/graph references for scanned pages.

## Classes and integration

| File | Class | Responsibility |
| --- | --- | --- |
| `settings.py` | `Settings` | Configuration and validation |
| `pdf_extractor.py` | `PDFExtractor` | Page counting and text extraction |
| `page_renderer.py` | `PageRenderer` | Isolated-process PNG rendering |
| `page_knowledge.py` | `PageKnowledgeExtractor` | Trustcall extraction with Pydantic validation and evidence |
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
if __name__ == "__main__":
    result = asyncio.run(run())
```

For files only, construct `RagPipeline(settings)` without `Stores`.
Components accept injected replacements so the pipeline can be tested
without external services. Tests live under `rag/tests/`.
Keep `asyncio.run()` under the main guard in scripts that render pages, because
the renderer spawns processes on Windows.
