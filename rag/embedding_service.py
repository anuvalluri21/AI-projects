"""Asynchronous OpenAI or Pinecone Inference embeddings."""
import os


class EmbeddingService:
    async def open(self, model, provider="openai", dimension=None):
        self.model = model
        self.provider = provider
        self.dimension = dimension
        if provider == "pinecone":
            from pinecone import AsyncPinecone
            self.client = AsyncPinecone(api_key=os.environ["PINECONE_API_KEY"])
        elif provider == "openai":
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=3, timeout=60)
        else:
            raise ValueError("Embedding provider must be openai or pinecone")

    async def embed(self, text, input_type="passage"):
        if input_type not in ("passage", "query"):
            raise ValueError("input_type must be passage or query")
        if self.provider == "pinecone":
            parameters = {"input_type": input_type}
            if self.dimension is not None:
                parameters["dimension"] = self.dimension
            result = await self.client.inference.embed(model=self.model, inputs=[text], parameters=parameters)
            values = result[0].values
        else:
            options = {"dimensions": self.dimension} if self.dimension is not None else {}
            result = await self.client.embeddings.create(model=self.model, input=text, **options)
            values = result.data[0].embedding
        if self.dimension is not None and len(values) != self.dimension:
            raise ValueError(f"Expected {self.dimension} embedding dimensions; received {len(values)}")
        return values

    async def close(self):
        if hasattr(self, "client"):
            await self.client.close()
