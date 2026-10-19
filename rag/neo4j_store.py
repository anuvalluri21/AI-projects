"""Neo4j document, page, chunk, and ordering relationships."""
import os

class Neo4jGraphStore:
    async def open(self):
        from neo4j import AsyncGraphDatabase
        self.driver = AsyncGraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
        )
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        await self.driver.verify_connectivity()
        for label in ("Document", "Page", "Chunk"):
            await self.query(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")

    async def query(self, cypher, **parameters):
        return await self.driver.execute_query(
            cypher, parameters_=parameters, database_=self.database
        )

    async def document(self, metadata):
        await self.query("MERGE (d:Document {id: $meta.id}) SET d += $meta", meta=metadata)

    async def page(self, document_id, metadata):
        await self.query(
            "MATCH (d:Document {id:$doc}) MERGE (p:Page {id:$meta.id}) "
            "SET p += $meta MERGE (d)-[:HAS_PAGE]->(p)", doc=document_id, meta=metadata
        )

    async def chunk(self, document_id, page_id, metadata):
        await self.query(
            "MATCH (d:Document {id:$doc}), (p:Page {id:$page}) "
            "MERGE (c:Chunk {id:$meta.id}) SET c += $meta "
            "MERGE (d)-[:HAS_CHUNK]->(c) MERGE (p)-[:HAS_CHUNK]->(c)",
            doc=document_id, page=page_id, meta=metadata,
        )

    async def link(self, pairs):
        # Avoid oversized transactions on very large documents.
        for start in range(0, len(pairs), 1000):
            await self.query(
                "UNWIND $pairs AS pair MATCH (a:Chunk {id:pair[0]}), "
                "(b:Chunk {id:pair[1]}) MERGE (a)-[:NEXT_CHUNK]->(b)",
                pairs=pairs[start:start + 1000]
            )

    async def close(self):
        if hasattr(self, "driver"):
            await self.driver.close()
