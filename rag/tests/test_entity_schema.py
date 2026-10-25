"""Offline model selection, dedup correctness, and parameterized query tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock
from pydantic import ValidationError
from rag.entity_schema import EntitySchemaConfig
from rag.entity_resolver import EntityResolver, EntityConflictError
from rag.page_knowledge import PageEntity, PageKnowledge


class SchemaTests(unittest.TestCase):
    def test_create_model_union_selects_employee_and_rejects_unknown_label(self):
        model = EntitySchemaConfig.default().build_extraction_model(PageEntity, PageKnowledge)
        entity = {"local_id": "e1", "name": "Anu", "entity_type": "Employee", "email": "anu@example.com", "evidence": "Anu is an employee"}
        result = model.model_validate({"transcribed_text": "", "entities": [entity], "relationships": []})
        self.assertEqual(type(result.entities[0]).__name__, "ExtractedEmployee")
        self.assertEqual(result.entities[0].email, "anu@example.com")
        self.assertEqual(len(model.model_json_schema()["properties"]["entities"]["items"]["oneOf"]), 4)
        entity["entity_type"] = "InventedLabel"
        with self.assertRaises(ValidationError):
            model.model_validate({"transcribed_text": "", "entities": [entity], "relationships": []})

    def test_load_config_and_normalize_identifiers(self):
        schema = EntitySchemaConfig.default()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.json"
            path.write_text(schema.model_dump_json())
            self.assertEqual(EntitySchemaConfig.load(path), schema)
        normalized = schema.canonical_properties({"entity_type": "Person", "name": " Anu   Valluri ",
                                                  "email": "ANU@Example.com ", "phone": "+1 (732) 491-6652"})
        self.assertEqual(normalized, {"name": "anu valluri", "email": "anu@example.com", "phone": "+17324916652"})
        self.assertIsNone(schema.normalize("732-491-6652", "phone"))

    def test_invalid_config_identifiers_or_fields_rejected(self):
        data = EntitySchemaConfig.default().model_dump()
        data["node_types"]["Bad` MATCH (n)"] = data["node_types"].pop("Equipment")
        with self.assertRaises(ValidationError):
            EntitySchemaConfig.model_validate(data)
        data = EntitySchemaConfig.default().model_dump()
        data["dedup"]["Person"]["any_of"] = [{"fields": ["unknown"]}]
        with self.assertRaises(ValidationError):
            EntitySchemaConfig.model_validate(data)

    def test_configurable_composite_rules_create_matching_indexes(self):
        data = EntitySchemaConfig.default().model_dump()
        data["dedup"]["Person"]["any_of"] = [{"fields": ["name", "email"]}]
        resolver = EntityResolver(EntitySchemaConfig.model_validate(data))
        statements = resolver.index_queries()
        self.assertTrue(any('FOR (n:`Person`) ON (n.`scope`, n.`name`, n.`email`)' in q for q in statements))
        self.assertTrue(any('FOR (n:`Employee`)' in q for q in statements))


class MemoryTransaction:
    """Small behavioral fake for resolver MATCH/MERGE, not a Cypher engine."""
    def __init__(self):
        self.nodes = {}
        self.links = {}
        self.calls = []

    async def run(self, query, **params):
        self.calls.append((query, params))
        records = []
        if query.startswith("MATCH (c:CanonicalEntity"):
            for node in self.nodes.values():
                if node["scope"] == params["scope"] and all(node.get(k) == v for k, v in params.items() if k != "scope"):
                    records.append({"entity": dict(node)})
        elif query.startswith("MERGE (c:CanonicalEntity"):
            node = self.nodes.setdefault(params["id"], {"id": params["id"], "scope": params["scope"]})
            node.update(params["props"])
            self.links[params["mention"]] = params["id"]
        result = AsyncMock()
        result.data.return_value = records
        return result


class ResolverTests(unittest.IsolatedAsyncioTestCase):
    def mention(self, local_id, **fields):
        return {"id": local_id, "name": "Anu Valluri", "entity_type": "Person", **fields}

    async def test_different_models_same_email_share_hub_and_update(self):
        resolver = EntityResolver(EntitySchemaConfig.default())
        tx = MemoryTransaction()
        first = await resolver.resolve(tx, self.mention("p1:e1", email="ANU@example.com"))
        second = await resolver.resolve(tx, self.mention("p2:e1", entity_type="Employee", email="anu@example.com", phone="+17324916652"))
        self.assertEqual(first, second)
        self.assertEqual(len(tx.nodes), 1)
        self.assertEqual(tx.nodes[first]["phone"], "+17324916652")
        self.assertEqual(tx.links, {"p1:e1": first, "p2:e1": first})

    async def test_phone_and_name_fallback(self):
        resolver = EntityResolver(EntitySchemaConfig.default())
        tx = MemoryTransaction()
        first = await resolver.resolve(tx, self.mention("a", phone="+1 (732) 491-6652"))
        second = await resolver.resolve(tx, self.mention("b", phone="+17324916652"))
        third = await resolver.resolve(tx, self.mention("c"))
        self.assertEqual(first, second)
        self.assertEqual(first, third)

    async def test_conflicting_email_does_not_silently_merge_by_name(self):
        resolver = EntityResolver(EntitySchemaConfig.default())
        tx = MemoryTransaction()
        await resolver.resolve(tx, self.mention("a", email="anu@example.com"))
        with self.assertRaises(EntityConflictError):
            await resolver.resolve(tx, self.mention("b", email="someoneelse@example.com"))
        self.assertNotIn("b", tx.links)

    async def test_multiple_matches_are_ambiguous(self):
        resolver = EntityResolver(EntitySchemaConfig.default())
        tx = MemoryTransaction()
        tx.nodes = {"a": {"id": "a", "scope": resolver.schema.namespace, "email": "anu@example.com"},
                    "b": {"id": "b", "scope": resolver.schema.namespace, "phone": "+17324916652"}}
        with self.assertRaises(EntityConflictError):
            await resolver.resolve(tx, self.mention("c", email="anu@example.com", phone="+17324916652"))

    async def test_can_disable_name_rule_and_values_remain_parameters(self):
        config = EntitySchemaConfig.default().model_dump()
        config["dedup"]["Person"]["any_of"] = [{"fields": ["email"]}, {"fields": ["phone"]}]
        resolver = EntityResolver(EntitySchemaConfig.model_validate(config))
        tx = MemoryTransaction()
        first = await resolver.resolve(tx, self.mention("a", email="anu@example.com"))
        second = await resolver.resolve(tx, self.mention("b", email="other@example.com"))
        self.assertNotEqual(first, second)
        self.assertTrue(all("anu@example.com" not in query for query, _ in tx.calls))
