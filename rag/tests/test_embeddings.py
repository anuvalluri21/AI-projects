"""Provider routing and vector compatibility without remote API calls."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from rag.embedding_service import EmbeddingService
from rag.settings import Settings
from pathlib import Path

class EmbeddingTests(unittest.IsolatedAsyncioTestCase):
    def service(self, provider, dimension):
        service = EmbeddingService()
        service.provider, service.model, service.dimension = provider, 'test-model', dimension
        service.client = SimpleNamespace(inference=SimpleNamespace(embed=AsyncMock(return_value=[SimpleNamespace(values=[1., 2.])])), embeddings=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[1., 2.])]))))
        return service

    async def test_pinecone_passage_and_query(self):
        service = self.service('pinecone', 2)
        self.assertEqual(await service.embed('document'), [1., 2.])
        await service.embed('question', input_type='query')
        self.assertEqual(service.client.inference.embed.await_args.kwargs['parameters'], {'input_type':'query', 'dimension':2})
        self.assertEqual(service.client.inference.embed.await_args_list[0].kwargs['parameters']['input_type'], 'passage')
        service.client.embeddings.create.assert_not_awaited()

    async def test_openai_default_dimensions(self):
        service = self.service('openai', None)
        await service.embed('document')
        service.client.embeddings.create.assert_awaited_once_with(model='test-model', input='document')
        service.client.inference.embed.assert_not_awaited()

    async def test_dimension_mismatch(self):
        service = self.service('pinecone', 1024)
        with self.assertRaisesRegex(ValueError, 'Expected 1024'):
            await service.embed('document')

    def test_llama_requires_pinecone(self):
        with self.assertRaisesRegex(ValueError, 'requires'):
            Settings(Path('.'), embedding_model='llama-text-embed-v2').validate()
