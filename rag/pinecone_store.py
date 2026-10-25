"""Asynchronous Pinecone document storage with vectors and shared graph IDs."""
import os


class PineconeVectorStore:
    async def open(self):
        from pinecone import AsyncPinecone
        self.client = AsyncPinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = await self.client.index(host=os.environ["PINECONE_INDEX_HOST"])
        self.namespace = os.getenv("PINECONE_NAMESPACE", "pdf_chunks")
        self.stats = await self.index.describe_index_stats()

    async def upsert(self, metadata, text, values):
        if len(values) != self.stats.dimension:
            raise ValueError(f"Pinecone index dimension {self.stats.dimension} does not match embedding dimension {len(values)}")
        await self.index.documents.upsert(
            documents=[{**metadata, "_id": metadata["id"],
                        "embedding": values, "chunk_text": text}],
            namespace=self.namespace,
        )

    async def close(self):
        try:
            if hasattr(self, "index"):
                await self.index.close()
        finally:
            if hasattr(self, "client"):
                await self.client.close()
