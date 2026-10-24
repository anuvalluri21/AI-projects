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
    renderer: object

    async def process(self, index):
        settings, stores = self.settings, self.stores
        document_id, folder, snapshot, count = self.document_id, self.folder, self.snapshot, self.count
        number = index + 1
        text = await asyncio.to_thread(self.extractor.extract_page, snapshot, index)
        page_folder = folder / "pages" / f"page_{number:04d}"
        page_folder.mkdir(parents=True, exist_ok=True)
        text_path = page_folder / "page.txt"
        await asyncio.to_thread(text_path.write_text, text, encoding="utf-8")
        image_path = page_folder / "page.png"
        image_info = await self.renderer.render(snapshot, index, image_path, settings.render_dpi)
        page_id = f"{document_id}:p{number:04d}"
        knowledge_path = page_folder / "knowledge.json"
        page_meta = {
            "id": page_id, "document_id": document_id, "page_number": number,
            "text_path": str(text_path), "image_path": str(image_path),
            "image_width": image_info["width"], "image_height": image_info["height"],
            "knowledge_path": str(knowledge_path), "chunk_count": 0,
            "knowledge_status": "pending" if stores and settings.extract_entities else "skipped",
            "text_source": "pdf_text",
            "empty_text": not bool(text.strip()),
        }
        await asyncio.to_thread(self.files.write_json, page_folder / "page.json", page_meta)
        if stores:
            await stores.page(document_id, page_meta)

        if stores and settings.extract_entities:
            try:
                knowledge = await stores.page_knowledge(page_id, image_path, text)
                await asyncio.to_thread(self.files.write_json, knowledge_path, knowledge)
                page_meta.update(knowledge_status="extracted",
                                 entity_count=len(knowledge["entities"]),
                                 relationship_count=len(knowledge["relationships"]))
                if not text.strip() and knowledge["transcribed_text"].strip():
                    text = knowledge["transcribed_text"]
                    page_meta.update(text_source="vision_transcription", empty_text=False)
                    await asyncio.to_thread(text_path.write_text, text, encoding="utf-8")
            except Exception as error:
                page_meta["knowledge_status"] = "failed"
                await asyncio.to_thread(self.files.write_json, page_folder / "page.json", page_meta)
                await asyncio.to_thread(self.files.write_json, knowledge_path,
                                        {"page_id": page_id, "status": "failed", "error_type": type(error).__name__})
                try:
                    await stores.page(document_id, page_meta)
                except Exception:
                    pass
                raise
        else:
            await asyncio.to_thread(self.files.write_json, knowledge_path,
                                    {"page_id": page_id, "status": "skipped",
                                     "reason": "files-only mode or --skip-entities"})

        chunks = self.chunker.split(text)
        page_meta["chunk_count"] = len(chunks)
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
