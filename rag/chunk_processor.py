"""Save a chunk and coordinate embedding/vector/graph writes."""
import asyncio
from dataclasses import dataclass
from pathlib import Path
from .settings import Settings


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
                "embedding_provider": settings.embedding_provider,
                **({"embedding_dimension": settings.embedding_dimension} if settings.embedding_dimension is not None else {}),
                "page_image_path": str(page_folder / "page.png"),
                "page_knowledge_path": str(page_folder / "knowledge.json"),
            }
            await asyncio.to_thread(chunk_path.write_text, chunk["text"], encoding="utf-8")
            await asyncio.to_thread(self.files.write_json, chunk_path.with_suffix(".json"), chunk_meta)
            if stores:
                await stores.chunk(document_id, page_id, chunk_meta, chunk["text"])
            return chunk_meta
