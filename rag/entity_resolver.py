"""Indexed exact-match canonical hubs; page mentions remain evidence spokes."""
import hashlib
import uuid


class EntityConflictError(ValueError):
    """The configured identity rules cannot safely select one canonical hub."""


class EntityResolver:
    def __init__(self, schema):
        self.schema = schema

    def index_queries(self):
        queries = []
        seen = set()
        for label, definition in self.schema.node_types.items():
            for indexed_label in (label, definition.hub):
                for rule in self.schema.dedup[definition.hub].any_of:
                    key = (indexed_label, tuple(rule.fields))
                    if key in seen:
                        continue
                    seen.add(key)
                    suffix = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
                    fields = ", ".join(f"n.`{field}`" for field in ["scope", *rule.fields])
                    queries.append(f"CREATE INDEX rag_identity_{suffix} IF NOT EXISTS "
                                   f"FOR (n:`{indexed_label}`) ON ({fields})")
        return queries

    async def acquire_locks(self, tx, entities):
        # All ingestion processes must use these locks. Sorted order avoids lock cycles.
        hubs = sorted({self.schema.node_types[e["entity_type"]].hub for e in entities})
        for hub in hubs:
            result = await tx.run(
                "MERGE (l:EntityWriteLock {id:$id}) SET l.version=coalesce(l.version,0)+1",
                id=self.schema.namespace + ":" + hub,
            )
            await result.consume()

    async def resolve(self, tx, mention):
        label = mention["entity_type"]
        if label not in self.schema.node_types:
            raise ValueError(f"Unconfigured entity label: {label}")
        definition = self.schema.node_types[label]
        hub = definition.hub
        rules = self.schema.dedup[hub]
        props = self.schema.canonical_properties(mention)
        candidates = {}
        for rule in rules.any_of:
            if not all(field in props for field in rule.fields):
                continue
            predicate = " AND ".join(f"c.`{field}` = ${field}" for field in rule.fields)
            index_fields = ", ".join(f"`{field}`" for field in ["scope", *rule.fields])
            result = await tx.run(
                f"MATCH (c:CanonicalEntity:`{hub}`) USING INDEX c:`{hub}`({index_fields}) "
                f"WHERE c.scope=$scope AND {predicate} "
                "RETURN c{.*} AS entity LIMIT 2",
                scope=self.schema.namespace, **{field: props[field] for field in rule.fields},
            )
            for record in await result.data():
                candidate = record["entity"]
                candidates[candidate["id"]] = candidate
        if len(candidates) > 1:
            raise EntityConflictError("Dedup rules matched multiple canonical entities; review the page and rules")
        candidate = next(iter(candidates.values()), None)
        if candidate:
            for field in rules.conflict_fields:
                if field in props and candidate.get(field) is not None and candidate[field] != props[field]:
                    raise EntityConflictError(f"Conflicting {field} on a dedup match; no nodes were merged")
        canonical_id = candidate["id"] if candidate else "entity:" + uuid.uuid4().hex
        result = await tx.run(
            f"MERGE (c:CanonicalEntity:`{hub}` {{id:$id}}) "
            f"ON CREATE SET c.created_at=datetime() SET c:`{label}`, c.scope=$scope, "
            "c += $props, c.display_name=$name, c.updated_at=datetime() "
            "WITH c MATCH (m:PageEntity {id:$mention}) SET m.canonical_id=c.id "
            "MERGE (m)-[:REFERS_TO]->(c)",
            id=canonical_id, scope=self.schema.namespace, props=props,
            name=mention["name"], mention=mention["id"],
        )
        await result.consume()
        return canonical_id
