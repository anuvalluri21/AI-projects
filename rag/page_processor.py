"""Extract a page and process its chunks asynchronously."""
import asyncio
from dataclasses import dataclass
from pathlib import Path
from .chunk_processor import ChunkProcessor
from .settings import Settings


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
