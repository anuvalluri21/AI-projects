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
        for label in ("Document", "Page", "Chunk", "PageEntity", "PageRelation"):
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

    async def page_knowledge(self, page_id, knowledge):
        """Replace this page's extracted graph atomically, avoiding stale retry data."""
        entities = [{**entity, "page_id": page_id, "image_path": knowledge["image_path"],
                     "model": knowledge["model"]} for entity in knowledge["entities"]]
        relations = [{**relation, "page_id": page_id, "image_path": knowledge["image_path"],
                      "model": knowledge["model"]} for relation in knowledge["relationships"]]

        async def replace(tx):
            queries = [
                ("MATCH (r:PageRelation {page_id:$page}) DETACH DELETE r", {}),
                ("MATCH (e:PageEntity {page_id:$page}) DETACH DELETE e", {}),
                ("MATCH (p:Page {id:$page}) UNWIND $entities AS meta "
                 "MERGE (e:PageEntity {id:meta.id}) SET e += meta "
                 "MERGE (p)-[:MENTIONS]->(e)", {"entities": entities}),
                ("MATCH (p:Page {id:$page}) UNWIND $relations AS meta "
                 "MATCH (a:PageEntity {id:meta.source_id}), (b:PageEntity {id:meta.target_id}) "
                 "MERGE (r:PageRelation {id:meta.id}) SET r += meta "
                 "MERGE (p)-[:HAS_RELATION]->(r) "
                 "MERGE (r)-[:SUBJECT]->(a) MERGE (r)-[:OBJECT]->(b)", {"relations": relations}),
            ]
            for query, params in queries:
                result = await tx.run(query, page=page_id, **params)
                await result.consume()

        async with self.driver.session(database=self.database) as session:
            await session.execute_write(replace)

    async def page_context(self, page_id):
        records, _, _ = await self.query(
            "MATCH (p:Page {id:$page}) "
            "OPTIONAL MATCH (p)-[:MENTIONS]->(e:PageEntity) "
            "OPTIONAL MATCH (p)-[:HAS_RELATION]->(r:PageRelation) "
            "RETURN p{.*, entities:collect(DISTINCT e{.*}), "
            "relationships:collect(DISTINCT r{.*})} AS context", page=page_id
        )
        return records[0]["context"] if records else None

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
