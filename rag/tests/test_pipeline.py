"""Offline tests: ordering, concurrency bounds, recovery, and text coverage."""

import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag import Settings, RagPipeline
from rag.text_chunker import TextChunker


async def ingest(source, settings, stores=None):
    return await RagPipeline(settings, stores, renderer=FakeRenderer()).ingest(source)


class FakeRenderer:
    async def render(self, pdf, index, destination, dpi):
        destination.write_bytes(b"fake test image")
        return {"width": 10, "height": 10}

    async def close(self):
        pass


def split_text(text, size, overlap):
    return TextChunker(size, overlap).split(text)


class FakeStores:
    def __init__(self, fail=False):
        self.active = self.peak = 0
        self.chunks = {}
        self.documents = []
        self.pairs = []
        self.fail = fail

    async def document(self, metadata):
        self.documents.append(dict(metadata))

    async def page(self, document_id, metadata):
        pass

    async def page_knowledge(self, page_id, image_path, text):
        return {"entities": [], "relationships": [], "transcribed_text": "", "status": "extracted"}

    async def chunk(self, document_id, page_id, metadata, text):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.001 * (4 - metadata["page_number"] % 4))
            if self.fail:
                raise RuntimeError("embedding service unavailable")
            self.chunks[metadata["id"]] = (metadata, text)
        finally:
            self.active -= 1

    async def link(self, pairs):
        self.pairs = pairs


class ChunkTests(unittest.TestCase):
    def test_offsets_overlap_and_full_coverage(self):
        text = "0123456789" * 5
        chunks = split_text(text, 12, 3)
        reconstructed = chunks[0]["text"]
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous["end"] - current["start"], 3)
            reconstructed += current["text"][3:]
        self.assertEqual(reconstructed, text)
        self.assertTrue(all(c["text"] == text[c["start"]:c["end"]] for c in chunks))
        self.assertEqual(split_text("   ", 12, 3), [])
        with self.assertRaises(ValueError):
            split_text(text, 12, 12)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_50_page_pdf_files_only(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = PdfWriter()
            font = DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            })
            font_ref = writer._add_object(font)
            for number in range(1, 51):
                page = writer.add_blank_page(width=612, height=792)
                page[NameObject("/Resources")] = DictionaryObject({
                    NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})
                })
                stream = DecodedStreamObject()
                stream.set_data(f"BT /F1 12 Tf 72 720 Td (Page {number}: knowledge graph example.) Tj ET".encode())
                page[NameObject("/Contents")] = writer._add_object(stream)
            pdf = root / "fifty_pages.pdf"
            writer.write(pdf)
            with contextlib.redirect_stdout(io.StringIO()):
                result = await RagPipeline(Settings(root / "data", render_dpi=72)).ingest(pdf)
            self.assertEqual(result["page_count"], 50)
            self.assertEqual(result["chunk_count"], 50)
            self.assertEqual(result["status"], "files_only")
            self.assertEqual(result["empty_pages"], [])
            text_path = root / "data" / result["id"] / "pages" / "page_0050" / "page.txt"
            self.assertIn("Page 50", text_path.read_text())
            import pymupdf
            image = pymupdf.Pixmap(str(text_path.with_name("page.png")))
            self.assertEqual((image.width, image.height), (612, 792))
            page_meta = json.loads(text_path.with_name("page.json").read_text())
            self.assertTrue(Path(page_meta["image_path"]).exists())
            self.assertEqual(page_meta["knowledge_status"], "skipped")
            self.assertEqual(len(list((root / "data" / result["id"]).glob("pages/*/page.png"))), 50)

    async def test_50_pages_ordering_and_idempotent_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "sample.pdf"
            pdf.write_bytes(b"offline PDF extraction fixture")
            settings = Settings(root / "data", chunk_size=12, overlap=3, page_workers=5, chunk_workers=3)
            stores = FakeStores()
            # Retain page 25 but skip its empty text in NEXT_CHUNK ordering.
            def extract(path, index):
                return "" if index == 24 else f"Page {index + 1}: " + "abcdefghij" * 3
            with patch("rag.pdf_extractor.PDFExtractor.page_count", return_value=50), patch("rag.pdf_extractor.PDFExtractor.extract_page", side_effect=extract), contextlib.redirect_stdout(io.StringIO()):
                result = await ingest(pdf, settings, stores)
                again = await ingest(pdf, settings, FakeStores())
            self.assertEqual(result["id"], again["id"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["empty_pages"], [25])
            self.assertLessEqual(stores.peak, 3)
            self.assertGreater(stores.peak, 1)
            self.assertEqual(len(stores.pairs), result["chunk_count"] - 1)
            ordered = sorted(stores.chunks, key=lambda key: (stores.chunks[key][0]["page_number"], stores.chunks[key][0]["chunk_number"]))
            self.assertEqual(stores.pairs, [[a, b] for a, b in zip(ordered, ordered[1:])])
            folder = settings.output / result["id"]
            self.assertEqual(len(list((folder / "pages").glob("page_*"))), 50)
            first = json.loads(Path(stores.chunks[ordered[0]][0]["text_path"]).with_suffix(".json").read_text())
            last = json.loads(Path(stores.chunks[ordered[-1]][0]["text_path"]).with_suffix(".json").read_text())
            self.assertIsNone(first["previous_chunk_id"])
            self.assertEqual(first["next_chunk_id"], ordered[1])
            self.assertIsNone(last["next_chunk_id"])

    async def test_failed_ingestion_is_not_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "sample.pdf"
            pdf.write_bytes(b"fixture")
            stores = FakeStores(fail=True)
            with patch("rag.pdf_extractor.PDFExtractor.page_count", return_value=2), patch("rag.pdf_extractor.PDFExtractor.extract_page", return_value="text"), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ExceptionGroup):
                    await ingest(pdf, Settings(root / "data"), stores)
            self.assertEqual(stores.documents[-1]["status"], "failed")
            manifest = next((root / "data").glob("*/document.json"))
            self.assertEqual(json.loads(manifest.read_text())["status"], "failed")
            self.assertEqual(stores.active, 0)
            self.assertEqual(stores.pairs, [])


if __name__ == "__main__":
    unittest.main()
