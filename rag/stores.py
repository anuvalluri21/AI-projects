"""Integrate graph storage, embeddings, and vector storage."""
from .embedding_service import EmbeddingService
from .neo4j_store import Neo4jGraphStore
from .pinecone_store import PineconeVectorStore


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
