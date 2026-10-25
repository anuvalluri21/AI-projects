"""Orchestrate asynchronous page/chunk processing and ordering links."""
import asyncio
from pathlib import Path

from .async_executor import AsyncExecutor
from .file_store import FileStore
from .pdf_extractor import PDFExtractor
from .text_chunker import TextChunker
from .page_processor import PageProcessor
from .page_renderer import PageRenderer


class RagPipeline:
    def __init__(self, settings, stores=None, extractor=None, chunker=None, files=None, executor=None, renderer=None):
        settings.validate()
        self.settings = settings
        self.stores = stores
        self.extractor = extractor if extractor is not None else PDFExtractor()
        self.chunker = chunker if chunker is not None else TextChunker(settings.chunk_size, settings.overlap)
        self.files = files if files is not None else FileStore()
        self.executor = executor if executor is not None else AsyncExecutor()
        self.renderer = renderer if renderer is not None else PageRenderer()

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
                "embedding_provider": settings.embedding_provider,
                **({"embedding_dimension": settings.embedding_dimension} if settings.embedding_dimension is not None else {}),
            "render_dpi": settings.render_dpi, "entity_extraction": bool(stores and settings.extract_entities),
            "entity_model": settings.entity_model,
        }
        manifest = folder / "document.json"
        await asyncio.to_thread(self.files.write_json, manifest, metadata)
        chunk_slots = asyncio.Semaphore(settings.chunk_workers)

        page_processor = PageProcessor(
            settings, stores, self.extractor, self.chunker, self.files, self.executor,
            document_id, folder, snapshot, count, chunk_slots, self.renderer,
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
        finally:
            await self.renderer.close()
