"""Single-file PDF ingestion into Neo4j and Pinecone.

Requires Python 3.11+ and dependencies from rag/requirements.txt.
Run: python rag_single_file.py document.pdf
Offline: python rag_single_file.py document.pdf --files-only
All application classes are defined here; no rag package imports are required.
"""
import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path



# Components from settings.py
@dataclass(frozen=True)
class Settings:
    output: Path
    chunk_size: int = 1200
    overlap: int = 200
    page_workers: int = 5
    chunk_workers: int = 8
    embedding_model: str = "text-embedding-3-small"

    def validate(self):
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("Require 0 <= overlap < chunk_size")
        if self.page_workers < 1 or self.chunk_workers < 1:
            raise ValueError("Worker counts must be positive")


# Components from pdf_extractor.py
class PDFExtractor:
    @staticmethod
    def page_count(pdf):
        from pypdf import PdfReader
        return len(PdfReader(pdf).pages)

    @staticmethod
    def extract_page(pdf, index):
        # Each worker owns its reader; mutable PDF reader state is never shared.
        from pypdf import PdfReader
        return PdfReader(pdf).pages[index].extract_text() or ""


# Components from text_chunker.py
class TextChunker:
    def __init__(self, size=1200, overlap=200):
        if not 0 <= overlap < size:
            raise ValueError("Require 0 <= overlap < size")
        self.size = size
        self.overlap = overlap

    def split(self, text):
        """Return exact character spans; chunks never cross a page boundary."""
        size, overlap = self.size, self.overlap
        if not 0 <= overlap < size:
            raise ValueError("Require 0 <= overlap < size")
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if text[start:end].strip():
                chunks.append({"start": start, "end": end, "text": text[start:end]})
            if end == len(text):
                break
            start = end - overlap
        return chunks


# Components from file_store.py
class FileStore:
    @staticmethod
    def write_json(path, value):
        """Replace metadata atomically so failed runs do not leave truncated JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def snapshot_pdf(source, settings):
        data = source.read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        config = f"{checksum}:{settings.chunk_size}:{settings.overlap}:{settings.embedding_model}:v1"
        document_id = hashlib.sha256(config.encode()).hexdigest()
        folder = settings.output.resolve() / document_id
        folder.mkdir(parents=True, exist_ok=True)
        snapshot = folder / "source.pdf"
        snapshot.write_bytes(data)
        return document_id, checksum, folder, snapshot


# Components from async_executor.py
class AsyncExecutor:
    @staticmethod
    async def bounded_map(items, workers, operation):
        """Fixed workers avoid allocating one task per chunk for large PDFs."""
        iterator = iter(items)
        results = []

        async def worker():
            for item in iterator:
                results.append(await operation(item))

        async with asyncio.TaskGroup() as group:
            for _ in range(workers):
                group.create_task(worker())
        return results


# Components from embedding_service.py
class EmbeddingService:
    async def open(self, model):
        from openai import AsyncOpenAI
        self.model = model
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=3, timeout=60)

    async def embed(self, text):
        result = await self.client.embeddings.create(model=self.model, input=text)
        return result.data[0].embedding

    async def close(self):
        if hasattr(self, "client"):
            await self.client.close()


# Components from neo4j_store.py
class Neo4jGraphStore:
    async def open(self):
        from neo4j import AsyncGraphDatabase
        self.driver = AsyncGraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
        )
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        await self.driver.verify_connectivity()
        for label in ("Document", "Page", "Chunk"):
            await self.query(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")

    async def query(self, cypher, **parameters):
        return await self.driver.execute_query(
            cypher, parameters_=parameters, database_=self.database
        )

    async def document(self, metadata):
        await self.query("MERGE (d:Document {id: $meta.id}) SET d += $meta", meta=metadata)

    async def page(self, document_id, metadata):
        await self.query(
            "MATCH (d:Document {id:$doc}) MERGE (p:Page {id:$meta.id}) "
            "SET p += $meta MERGE (d)-[:HAS_PAGE]->(p)", doc=document_id, meta=metadata
        )

    async def chunk(self, document_id, page_id, metadata):
        await self.query(
            "MATCH (d:Document {id:$doc}), (p:Page {id:$page}) "
            "MERGE (c:Chunk {id:$meta.id}) SET c += $meta "
            "MERGE (d)-[:HAS_CHUNK]->(c) MERGE (p)-[:HAS_CHUNK]->(c)",
            doc=document_id, page=page_id, meta=metadata,
        )

    async def link(self, pairs):
        # Avoid oversized transactions on very large documents.
        for start in range(0, len(pairs), 1000):
            await self.query(
                "UNWIND $pairs AS pair MATCH (a:Chunk {id:pair[0]}), "
                "(b:Chunk {id:pair[1]}) MERGE (a)-[:NEXT_CHUNK]->(b)",
                pairs=pairs[start:start + 1000]
            )

    async def close(self):
        if hasattr(self, "driver"):
            await self.driver.close()


# Components from pinecone_store.py
class PineconeVectorStore:
    async def open(self):
        from pinecone import AsyncPinecone
        self.client = AsyncPinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = await self.client.index(host=os.environ["PINECONE_INDEX_HOST"])
        self.namespace = os.getenv("PINECONE_NAMESPACE", "pdf_chunks")
        self.stats = await self.index.describe_index_stats()

    async def upsert(self, metadata, text, values):
        if len(values) != self.stats.dimension:
            raise ValueError(f"Pinecone index dimension {self.stats.dimension} does not match embedding dimension {len(values)}")
        await self.index.upsert(
            vectors=[{"id": metadata["id"], "values": values,
                      "metadata": {**metadata, "text": text}}],
            namespace=self.namespace,
        )

    async def close(self):
        try:
            if hasattr(self, "index"):
                await self.index.close()
        finally:
            if hasattr(self, "client"):
                await self.client.close()


# Components from stores.py
class Stores:
    def __init__(self, graph=None, embeddings=None, vectors=None):
        self.graph = graph if graph is not None else Neo4jGraphStore()
        self.embeddings = embeddings if embeddings is not None else EmbeddingService()
        self.vectors = vectors if vectors is not None else PineconeVectorStore()

    async def open(self, settings):
        await self.graph.open()
        await self.embeddings.open(settings.embedding_model)
        await self.vectors.open()

    async def document(self, metadata):
        await self.graph.document(metadata)

    async def page(self, document_id, metadata):
        await self.graph.page(document_id, metadata)

    async def chunk(self, document_id, page_id, metadata, text):
        values = await self.embeddings.embed(text)
        await self.vectors.upsert(metadata, text, values)
        await self.graph.chunk(document_id, page_id, metadata)

    async def link(self, pairs):
        await self.graph.link(pairs)

    async def close(self):
        # Attempt every cleanup, including when initialization partially fails.
        errors = []
        for component in (self.vectors, self.embeddings, self.graph):
            try:
                await component.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Failed to close RAG clients", errors)


# Components from chunk_processor.py
@dataclass
class ChunkProcessor:
    settings: Settings
    stores: object
    files: object
    chunk_slots: asyncio.Semaphore
    document_id: str
    page_id: str
    page_folder: Path
    number: int

    async def process(self, item):
        settings, stores, chunk_slots = self.settings, self.stores, self.chunk_slots
        document_id, page_id = self.document_id, self.page_id
        page_folder, number = self.page_folder, self.number
        position, chunk = item
        async with chunk_slots:
            chunk_id = f"{page_id}:c{position:04d}"
            chunk_folder = page_folder / "chunks"
            chunk_folder.mkdir(exist_ok=True)
            chunk_path = chunk_folder / f"chunk_{position:04d}.txt"
            chunk_meta = {
                "id": chunk_id, "document_id": document_id, "page_id": page_id,
                "page_number": number, "chunk_number": position,
                "start_char": chunk["start"], "end_char": chunk["end"],
                "text_path": str(chunk_path), "embedding_model": settings.embedding_model,
            }
            await asyncio.to_thread(chunk_path.write_text, chunk["text"], encoding="utf-8")
            await asyncio.to_thread(self.files.write_json, chunk_path.with_suffix(".json"), chunk_meta)
            if stores:
                await stores.chunk(document_id, page_id, chunk_meta, chunk["text"])
            return chunk_meta


# Components from page_processor.py
@dataclass
class PageProcessor:
    settings: Settings
    stores: object
    extractor: object
    chunker: object
    files: object
    executor: object
    document_id: str
    folder: Path
    snapshot: Path
    count: int
    chunk_slots: asyncio.Semaphore

    async def process(self, index):
        settings, stores = self.settings, self.stores
        document_id, folder, snapshot, count = self.document_id, self.folder, self.snapshot, self.count
        number = index + 1
        text = await asyncio.to_thread(self.extractor.extract_page, snapshot, index)
        page_folder = folder / "pages" / f"page_{number:04d}"
        page_folder.mkdir(parents=True, exist_ok=True)
        text_path = page_folder / "page.txt"
        await asyncio.to_thread(text_path.write_text, text, encoding="utf-8")
        chunks = self.chunker.split(text)
        page_id = f"{document_id}:p{number:04d}"
        page_meta = {
            "id": page_id, "document_id": document_id, "page_number": number,
            "text_path": str(text_path), "chunk_count": len(chunks),
            "empty_text": not bool(text.strip()),
        }
        if stores:
            await stores.page(document_id, page_meta)

        processor = ChunkProcessor(settings, stores, self.files, self.chunk_slots, document_id, page_id, page_folder, number)

        chunk_metadata = await self.executor.bounded_map(
            enumerate(chunks, start=1), settings.chunk_workers, processor.process
        )
        chunk_metadata.sort(key=lambda chunk: chunk["chunk_number"])
        await asyncio.to_thread(self.files.write_json, page_folder / "page.json", page_meta)
        print(f"Page {number}/{count}: {len(chunks)} chunks", flush=True)
        return {"page": page_meta, "chunks": chunk_metadata}


# Components from pipeline.py
class RagPipeline:
    def __init__(self, settings, stores=None, extractor=None, chunker=None, files=None, executor=None):
        settings.validate()
        self.settings = settings
        self.stores = stores
        self.extractor = extractor if extractor is not None else PDFExtractor()
        self.chunker = chunker if chunker is not None else TextChunker(settings.chunk_size, settings.overlap)
        self.files = files if files is not None else FileStore()
        self.executor = executor if executor is not None else AsyncExecutor()

    async def ingest(self, source):
        settings, stores = self.settings, self.stores
        settings.validate()
        document_id, checksum, folder, snapshot = await asyncio.to_thread(self.files.snapshot_pdf, source, settings)
        count = await asyncio.to_thread(self.extractor.page_count, snapshot)
        if count == 0:
            raise ValueError("PDF has no pages")
        metadata = {
            "id": document_id, "source_name": source.name, "sha256": checksum,
            "source_path": str(snapshot), "page_count": count, "status": "ingesting",
            "chunk_size": settings.chunk_size, "overlap": settings.overlap,
            "embedding_model": settings.embedding_model,
        }
        manifest = folder / "document.json"
        await asyncio.to_thread(self.files.write_json, manifest, metadata)
        chunk_slots = asyncio.Semaphore(settings.chunk_workers)

        page_processor = PageProcessor(
            settings, stores, self.extractor, self.chunker, self.files, self.executor,
            document_id, folder, snapshot, count, chunk_slots,
        )

        try:
            if stores:
                await stores.document(metadata)
            pages = await self.executor.bounded_map(range(count), settings.page_workers, page_processor.process)
            pages.sort(key=lambda page: page["page"]["page_number"])
            ordered = [chunk for page in pages for chunk in page["chunks"]]
            pairs = [[a["id"], b["id"]] for a, b in zip(ordered, ordered[1:])]
            if stores:
                await stores.link(pairs)
            # Persist the same ordering references locally, including page boundaries.
            for index, chunk in enumerate(ordered):
                chunk["previous_chunk_id"] = ordered[index - 1]["id"] if index else None
                chunk["next_chunk_id"] = ordered[index + 1]["id"] if index + 1 < len(ordered) else None
                await asyncio.to_thread(self.files.write_json, Path(chunk["text_path"]).with_suffix(".json"), chunk)
            metadata.update(
                status="completed" if stores else "files_only",
                chunk_count=len(ordered), empty_pages=[p["page"]["page_number"] for p in pages if p["page"]["empty_text"]]
            )
            if stores:
                # Neo4j properties cannot contain maps; empty_pages is an integer list.
                await stores.document(metadata)
            await asyncio.to_thread(self.files.write_json, manifest, {**metadata, "pages": pages})
            return metadata
        except BaseException:
            metadata["status"] = "failed"
            await asyncio.to_thread(self.files.write_json, manifest, metadata)
            if stores:
                try:
                    await stores.document(metadata)
                except Exception:
                    pass  # Preserve the original ingestion error.
            raise


# Components from cli.py
async def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / "rag" / ".env")
    load_dotenv(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("pdf_data"))
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--page-workers", type=int, default=5)
    parser.add_argument("--chunk-workers", type=int, default=8)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--files-only", action="store_true", help="Skip Neo4j and embeddings")
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error("PDF file does not exist")
    settings = Settings(args.output, args.chunk_size, args.overlap, args.page_workers, args.chunk_workers, args.embedding_model)
    settings.validate()
    stores = None
    if not args.files_only:
        required = ("OPENAI_API_KEY", "NEO4J_URI", "NEO4J_PASSWORD", "PINECONE_API_KEY", "PINECONE_INDEX_HOST")
        missing = [key for key in required if not os.getenv(key) or os.getenv(key).startswith("your-")]
        if missing:
            parser.error("Missing configuration: " + ", ".join(missing))
        stores = Stores()
    try:
        if stores:
            await stores.open(settings)
        result = await RagPipeline(settings, stores).ingest(args.pdf.resolve())
        print(json.dumps(result, indent=2))
    finally:
        if stores:
            await stores.close()


if __name__ == "__main__":
    asyncio.run(main())
