"""Verify integration order and cleanup without external services."""
import unittest
from unittest.mock import AsyncMock

from rag.stores import Stores


class StoreIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_chunk_embeds_then_writes_vector_then_graph(self):
        events = []
        graph, embeddings, vectors = AsyncMock(), AsyncMock(), AsyncMock()

        async def embed(text):
            events.append("embed")
            return [0.1, 0.2]

        async def upsert(metadata, text, values):
            events.append("vector")
            self.assertEqual(values, [0.1, 0.2])

        async def graph_chunk(document_id, page_id, metadata):
            events.append("graph")
            self.assertEqual(metadata["id"], "chunk-1")

        embeddings.embed.side_effect = embed
        vectors.upsert.side_effect = upsert
        graph.chunk.side_effect = graph_chunk
        stores = Stores(graph, embeddings, vectors)
        await stores.chunk("doc-1", "page-1", {"id": "chunk-1"}, "text")
        self.assertEqual(events, ["embed", "vector", "graph"])

    async def test_vector_failure_does_not_record_completed_graph_chunk(self):
        graph, embeddings, vectors = AsyncMock(), AsyncMock(), AsyncMock()
        embeddings.embed.return_value = [0.1]
        vectors.upsert.side_effect = RuntimeError("vector service unavailable")
        stores = Stores(graph, embeddings, vectors)
        with self.assertRaises(RuntimeError):
            await stores.chunk("doc", "page", {"id": "chunk"}, "text")
        graph.chunk.assert_not_awaited()

    async def test_cleanup_attempts_every_component_if_one_fails(self):
        graph, embeddings, vectors = AsyncMock(), AsyncMock(), AsyncMock()
        vectors.close.side_effect = RuntimeError("close failed")
        with self.assertRaises(ExceptionGroup):
            await Stores(graph, embeddings, vectors).close()
        for component in (graph, embeddings, vectors):
            component.close.assert_awaited_once()
