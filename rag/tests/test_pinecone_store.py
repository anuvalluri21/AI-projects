"""Verify document API payloads and dimension guards."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from rag.pinecone_store import PineconeVectorStore

class PineconeDocumentTests(unittest.IsolatedAsyncioTestCase):
    def store(self):
        store = PineconeVectorStore()
        store.stats = SimpleNamespace(dimension=2)
        store.namespace = 'pdf_chunks'
        store.index = SimpleNamespace(documents=SimpleNamespace(upsert=AsyncMock()), upsert=AsyncMock())
        return store

    async def test_document_preserves_vector_text_and_provenance(self):
        store = self.store()
        metadata = {'id':'doc:p1:c1','page_number':1,'page_image_path':'page.png','document_id':'doc'}
        await store.upsert(metadata, 'page content', [0.1, 0.2])
        store.index.documents.upsert.assert_awaited_once_with(namespace='pdf_chunks', documents=[{**metadata, '_id':metadata['id'], 'embedding':[0.1,0.2], 'chunk_text':'page content'}])
        store.index.upsert.assert_not_awaited()
        self.assertNotIn('_id', metadata)

    async def test_wrong_dimension_never_writes(self):
        store = self.store()
        with self.assertRaisesRegex(ValueError, 'dimension'):
            await store.upsert({'id':'chunk'}, 'text', [0.1])
        store.index.documents.upsert.assert_not_awaited()

    async def test_document_write_failure_propagates(self):
        store = self.store()
        store.index.documents.upsert.side_effect = RuntimeError('write failed')
        with self.assertRaisesRegex(RuntimeError, 'write failed'):
            await store.upsert({'id':'chunk'}, 'text', [0.1,0.2])
