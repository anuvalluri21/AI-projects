"""Offline coverage for actual Trustcall graph execution and page provenance."""
import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError
from rag.page_knowledge import PageKnowledge, PageKnowledgeExtractor
from rag.neo4j_store import Neo4jGraphStore
from rag.pipeline import RagPipeline
from rag.settings import Settings
from rag.tests.test_pipeline import FakeRenderer, FakeStores


def payload():
    return {
        "transcribed_text": "Alice works at Acme.",
        "entities": [
            {"local_id": "e1", "name": "Alice", "entity_type": "Person", "evidence": "Alice"},
            {"local_id": "e2", "name": "Acme", "entity_type": "Organization", "evidence": "Acme"},
        ],
        "relationships": [{"source": "e1", "target": "e2", "predicate": "WORKS_AT",
                           "evidence": "Alice works at Acme.", "confidence": 0.95}],
    }


class KnowledgeSchemaTests(unittest.TestCase):
    def test_rejects_dangling_endpoints_and_duplicate_ids(self):
        data = payload()
        data["relationships"][0]["target"] = "missing"
        with self.assertRaises(ValidationError):
            PageKnowledge.model_validate(data)
        data = payload()
        data["entities"][1]["local_id"] = "e1"
        with self.assertRaises(ValidationError):
            PageKnowledge.model_validate(data)

    def test_ids_are_stable_and_scoped_to_page(self):
        knowledge = PageKnowledge.model_validate(payload())
        first = PageKnowledgeExtractor.attach_references(knowledge, "doc:p1", Path("page.png"), "offline")
        again = PageKnowledgeExtractor.attach_references(knowledge, "doc:p1", Path("page.png"), "offline")
        second = PageKnowledgeExtractor.attach_references(knowledge, "doc:p2", Path("page.png"), "offline")
        self.assertEqual(first, again)
        self.assertNotEqual(first["entities"][0]["id"], second["entities"][0]["id"])
        self.assertEqual(first["relationships"][0]["source_id"], first["entities"][0]["id"])


class PageKnowledgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_trustcall_graph_with_fake_model(self):
        from trustcall import create_extractor
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage

        class ToolModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        invalid = payload()
        invalid["relationships"][0]["target"] = "missing"
        model = ToolModel(responses=[
            AIMessage(content="", tool_calls=[
                {"name": "PageKnowledge", "args": invalid, "id": "call_offline"}
            ]),
            AIMessage(content="", tool_calls=[{
                "name": "PatchFunctionErrors", "id": "repair_offline",
                "args": {"json_doc_id": "call_offline", "planned_edits": "Reference the existing target entity e2",
                         "patches": [{"op": "replace", "path": "/relationships/0/target", "value": "e2"}]},
            }]),
        ])
        service = PageKnowledgeExtractor()
        service.model_name = "offline-fake-model"
        service.extractor = create_extractor(model, tools=[PageKnowledge], tool_choice="PageKnowledge")
        with tempfile.TemporaryDirectory() as temporary:
            png = Path(temporary) / "page.png"
            png.write_bytes(b"fake-image-for-mocked-model")
            result = await service.extract("doc:p1", png, "Alice works at Acme.")
        self.assertEqual(len(result["entities"]), 2)
        self.assertEqual(result["relationships"][0]["predicate"], "WORKS_AT")
        self.assertEqual(result["page_id"], "doc:p1")

    async def test_scanned_page_uses_transcription_and_keeps_image_references(self):
        class KnowledgeStores(FakeStores):
            async def page_knowledge(self, page_id, image_path, text):
                return PageKnowledgeExtractor.attach_references(
                    PageKnowledge.model_validate(payload()), page_id, image_path, "offline"
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "test.pdf"
            pdf.write_bytes(b"fixture")
            stores = KnowledgeStores()
            with patch("rag.pdf_extractor.PDFExtractor.page_count", return_value=1), patch("rag.pdf_extractor.PDFExtractor.extract_page", return_value=""), contextlib.redirect_stdout(io.StringIO()):
                result = await RagPipeline(Settings(root / "data"), stores, renderer=FakeRenderer()).ingest(pdf)
            folder = root / "data" / result["id"] / "pages/page_0001"
            meta = json.loads((folder / "page.json").read_text())
            self.assertEqual(meta["text_source"], "vision_transcription")
            self.assertEqual(meta["relationship_count"], 1)
            self.assertEqual((folder / "page.txt").read_text(), "Alice works at Acme.")
            chunk_meta, _ = next(iter(stores.chunks.values()))
            self.assertEqual(Path(chunk_meta["page_image_path"]).resolve(), (folder / "page.png").resolve())
            self.assertTrue(Path(chunk_meta["page_knowledge_path"]).exists())

    async def test_extraction_failure_keeps_png_and_marks_page_failed(self):
        stores = FakeStores()
        stores.page_knowledge = AsyncMock(side_effect=ValueError("bad extraction"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "test.pdf"
            pdf.write_bytes(b"fixture")
            with patch("rag.pdf_extractor.PDFExtractor.page_count", return_value=1), patch("rag.pdf_extractor.PDFExtractor.extract_page", return_value="text"), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ExceptionGroup):
                    await RagPipeline(Settings(root / "data"), stores, renderer=FakeRenderer()).ingest(pdf)
            page = next((root / "data").glob("*/pages/page_0001/page.json"))
            meta = json.loads(page.read_text())
            self.assertEqual(meta["knowledge_status"], "failed")
            self.assertTrue(Path(meta["image_path"]).exists())
            self.assertEqual(stores.documents[-1]["status"], "failed")

    async def test_neo4j_replaces_page_graph_in_one_transaction(self):
        graph = Neo4jGraphStore()
        graph.database = "neo4j"
        graph.driver = MagicMock()
        session, transaction = AsyncMock(), AsyncMock()
        graph.driver.session.return_value.__aenter__ = AsyncMock(return_value=session)

        async def execute_write(callback):
            await callback(transaction)

        session.execute_write.side_effect = execute_write
        data = PageKnowledgeExtractor.attach_references(PageKnowledge.model_validate(payload()), "doc:p1", Path("page.png"), "offline")
        await graph.page_knowledge("doc:p1", data)
        session.execute_write.assert_awaited_once()
        self.assertEqual(transaction.run.await_count, 4)
        arguments = transaction.run.await_args_list
        self.assertIn("DETACH DELETE", arguments[0].args[0])
        self.assertEqual(arguments[2].kwargs["entities"][0]["page_id"], "doc:p1")
        self.assertIn("HAS_RELATION", arguments[3].args[0])

    async def test_page_context_returns_saved_provenance(self):
        graph = Neo4jGraphStore()
        graph.query = AsyncMock(return_value=([{"context": {"id": "doc:p1", "image_path": "page.png"}}], None, None))
        result = await graph.page_context("doc:p1")
        self.assertEqual(result["image_path"], "page.png")
