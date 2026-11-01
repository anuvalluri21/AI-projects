"""Pinecone vector retrieval plus source-backed Neo4j graph traversal.
python hybrid_search.py ingest
python hybrid_search.py search
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

DOC_ID = "security-policy-001"
ROOT = Path(__file__).resolve().parent
QUESTION = ("Can an external contractor download customer data to a personal "
            "laptop, and who can approve an exception?")

# Each fact has an explicit source. Never infer a fact just from co-occurrence.
FACTS = [
    ("external contractor", "classified_as", "third-party user", 8),
    ("customer data", "classified_as", "restricted data", 24),
    ("third-party user", "subject_to", "device restriction", 51),
    ("restricted data", "download_governed_by", "device restriction", 51),
    ("device restriction", "requires_device", "company-managed device", 51),
    ("device restriction", "exception_requires_approval_from", "Security Director", 79),
]
ALIASES = {
    "external contractor": ["external contractor", "external contractors", "contractor"],
    "customer data": ["customer data"],
    "third-party user": ["third-party user", "third-party users"],
    "restricted data": ["restricted data"],
    "device restriction": ["device restriction", "device restrictions"],
    "company-managed device": ["company-managed device", "company-managed devices"],
    "Security Director": ["security director"],
}


def read_chunks(path):
    text = path.read_text(encoding="utf-8-sig")
    assert f"Document ID: {DOC_ID}" in text, "Unexpected document"
    parts = re.split(r"(?m)^CHUNK (\d+)\s*$", text)
    chunks = {int(parts[i]): parts[i + 1].strip()
              for i in range(1, len(parts), 2)}
    assert set(chunks) == {8, 24, 51, 79}
    return chunks


def link_entities(text):
    # Controlled aliases for this tiny example; use an entity linker at scale.
    return {entity for entity, names in ALIASES.items()
            if any(re.search(r"\b" + re.escape(name) + r"\b", text,
                             flags=re.IGNORECASE) for name in names)}



MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION = 1024
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
NAMESPACE = DOC_ID

# All nodes and relationships are scoped to a document, including every hop.
TRAVERSE = """
MATCH (seed:PolicyEntity {document_id: $doc})
WHERE seed.name IN $seeds
MATCH p=(seed)-[:POLICY_FACT*1..3]-(other:PolicyEntity)
WHERE all(n IN nodes(p) WHERE n.document_id = $doc)
  AND all(r IN relationships(p) WHERE r.document_id = $doc)
UNWIND relationships(p) AS r
WITH DISTINCT r
RETURN startNode(r).name AS source, r.predicate AS relation,
       endNode(r).name AS target, r.chunk_id AS chunk_id
ORDER BY chunk_id, source, relation, target
"""


def settings():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    required = ["PINECONE_API_KEY", "PINECONE_INDEX_HOST", "NEO4J_URI",
                "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing = [key for key in required
               if not os.getenv(key) or os.getenv(key) == "replace_me"]
    if missing:
        raise ValueError("Set these values in .env: " + ", ".join(missing))
    return {key: os.environ[key] for key in required}


def connect(config):
    from pinecone import Pinecone
    from neo4j import GraphDatabase
    pc = Pinecone(api_key=config["PINECONE_API_KEY"])
    index = pc.index(host=config["PINECONE_INDEX_HOST"])
    stats = index.describe_index_stats()
    if stats.dimension != EMBEDDING_DIMENSION:
        raise ValueError("Use a 1024-dimensional dense Pinecone index with cosine metric")
    driver = GraphDatabase.driver(config["NEO4J_URI"], auth=(
        config["NEO4J_USERNAME"], config["NEO4J_PASSWORD"]))
    driver.verify_connectivity()
    return index, driver


def query(driver, cypher, **params):
    records, _, _ = driver.execute_query(
        cypher, parameters_=params,
        database_=os.getenv("NEO4J_DATABASE", "neo4j"))
    return [r.data() for r in records]


def ingest(index, driver, model):
    chunks = read_chunks(ROOT / "data" / "security_policy.txt")
    query(driver, """CREATE CONSTRAINT policy_entity_key IF NOT EXISTS
        FOR (e:PolicyEntity) REQUIRE (e.document_id, e.name) IS UNIQUE""")
    query(driver, """CREATE CONSTRAINT policy_chunk_key IF NOT EXISTS
        FOR (c:PolicyChunk) REQUIRE (c.document_id, c.chunk_id) IS UNIQUE""")
    query(driver, """
        UNWIND $chunks AS item
        MERGE (c:PolicyChunk {document_id: $doc, chunk_id: item.chunk_id})
        SET c.text = item.text
        """, doc=DOC_ID,
        chunks=[{"chunk_id": c, "text": text} for c, text in chunks.items()])
    # Manually curated facts across ALL chunks; ingestion is idempotent for
    # this fixed sample. It does not remove obsolete facts after policy edits.
    query(driver, """
        UNWIND $facts AS f
        MERGE (a:PolicyEntity {document_id: $doc, name: f.source})
        MERGE (b:PolicyEntity {document_id: $doc, name: f.target})
        MERGE (a)-[r:POLICY_FACT {document_id: $doc,
                                 predicate: f.relation,
                                 chunk_id: f.chunk_id}]->(b)
        WITH a, b, f
        MATCH (c:PolicyChunk {document_id: $doc, chunk_id: f.chunk_id})
        MERGE (c)-[:MENTIONS]->(a)
        MERGE (c)-[:MENTIONS]->(b)
        """, doc=DOC_ID, facts=[
            {"source": a, "relation": r, "target": b, "chunk_id": c}
            for a, r, b, c in FACTS])
    ids = sorted(chunks)
    vectors = model.encode([chunks[c] for c in ids], normalize_embeddings=True)
    index.documents.upsert(namespace=NAMESPACE, documents=[
        {"_id": str(c), "embedding": vector.tolist(),
         "document_id": DOC_ID, "chunk_id": c, "text": chunks[c],
         "embedding_model": MODEL}
        for c, vector in zip(ids, vectors)])
    # Pinecone is eventually consistent. Check visibility with a bounded wait.
    deadline = time.monotonic() + 60
    while True:
        response = index.documents.fetch(ids=[str(c) for c in ids], namespace=NAMESPACE)
        if len(response.documents) == len(ids):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Writes submitted; wait before running search")
        time.sleep(2)
    print("Ingested 4 chunks into Pinecone and 6 source-backed facts into Neo4j.")
    print("If search has no matches yet, retry after a few seconds.")


def search(index, driver, model, question, top_k):
    vector = model.encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()
    response = index.documents.search(namespace=NAMESPACE, top_k=top_k,
        score_by=[{"type": "dense_vector", "field": "embedding", "values": vector}],
        include_fields=["chunk_id", "text"], filter={
            "document_id": {"$eq": DOC_ID},
            "embedding_model": {"$eq": MODEL}})
    if not response.matches:
        raise ValueError("No matches. Run ingest first, then wait and retry.")
    hits = [{"chunk_id": int(m.chunk_id), "score": m.score}
            for m in response.matches]
    seeds = link_entities(question)
    for match in response.matches:
        seeds.update(link_entities(match.text))
    # Expand in Neo4j, not in Python. Undirected paths discover evidence;
    # startNode/endNode preserve original fact direction in the returned data.
    facts = query(driver, TRAVERSE, doc=DOC_ID, seeds=sorted(seeds))
    vector_ids = {hit["chunk_id"] for hit in hits}
    graph_ids = {fact["chunk_id"] for fact in facts}
    combined = sorted(vector_ids | graph_ids)
    context = query(driver, """
        MATCH (c:PolicyChunk {document_id: $doc})
        WHERE c.chunk_id IN $ids
        RETURN c.chunk_id AS chunk_id, c.text AS text ORDER BY chunk_id
        """, doc=DOC_ID, ids=combined)
    if {c["chunk_id"] for c in context} != set(combined):
        raise ValueError("Missing Neo4j source chunks; rerun ingestion")
    return {"document_id": DOC_ID, "question": question,
            "vector_hits": hits, "seed_entities": sorted(seeds),
            "graph_facts": facts,
            "graph_added_chunk_ids": sorted(graph_ids - vector_ids),
            "context": context}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["ingest", "search", "check"])
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--top-k", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    config = settings()
    index, driver = connect(config)
    try:
        if args.command == "check":
            # Live service check for the ingested graph and vector evidence.
            facts = query(driver, TRAVERSE, doc=DOC_ID, seeds=["customer data"])
            assert {24, 51, 79} <= {f["chunk_id"] for f in facts}
            assert any(f["target"] == "Security Director" for f in facts)
            assert len(index.documents.fetch(ids=["8", "24", "51", "79"],
                                   namespace=NAMESPACE).documents) == 4
            print("PASS: Pinecone records and Neo4j three-hop evidence path")
            return
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL, cache_folder=os.getenv("EMBEDDING_CACHE_DIR", str(ROOT / ".model_cache")))
        if model.get_sentence_embedding_dimension() != EMBEDDING_DIMENSION:
            raise ValueError("Embedding model must output 1024 dimensions")
        if args.command == "ingest":
            ingest(index, driver, model)
        else:
            print(json.dumps(search(index, driver, model, args.question,
                                    args.top_k), indent=2))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
