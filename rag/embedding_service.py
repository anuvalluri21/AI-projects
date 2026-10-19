"""Asynchronous OpenAI embedding generation."""
import os


class EmbeddingService:
    async def open(self, model):
        from openai import AsyncOpenAI
        self.model = model
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=3, timeout=60)

    async def embed(self, text):
        result = await self.client.embeddings.create(model=self.model, input=text)
        return result.data[0].embedding

    async def close(self):
        if hasattr(self, "client"):
            await self.client.close()
