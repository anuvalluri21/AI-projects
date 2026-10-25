# Configurable node schemas and canonical entity hubs

Both `rag_single_file.py` and `python -m rag` load `rag/graph_schema.json`
automatically in this repository. Override it with `--schema-config path.json`.
The standalone file has built-in defaults if copied without the config; put
`graph_schema.json` beside it to customize without the `rag` package.

```powershell
python rag_single_file.py document.pdf --schema-config rag/graph_schema.json
```

## Dynamic Pydantic models and LLM label selection

`EntitySchemaConfig.build_extraction_model()` calls `pydantic.create_model()`
for every configured node type. The default output is conceptually:

```python
Entity = Annotated[
    Union[ExtractedPerson, ExtractedEmployee, ExtractedOrganization, ExtractedEquipment],
    Field(discriminator="entity_type"),
]
```

Trustcall receives the generated `ConfiguredPageKnowledge` schema. The LLM
selects an allowed `entity_type` and supplies that type's fields. Pydantic
rejects unconfigured labels, unknown properties, wrong field types, dangling
relationship endpoints, and invalid IDs. No LLM-generated labels or predicates
are directly interpolated into Cypher: graph labels and field names come only
from validated configuration, and entity values are parameters.

To add a type, edit `node_types` in the JSON. Every type has a canonical `hub`,
an optional description, and a `properties` dictionary. Supported property
types are `string`, `integer`, `number`, and `boolean`. Set `required: true`
to require a value. All types inherit local_id, name, entity_type, and evidence.

## Person identity rules

This section of the default config defines alternative exact-match keys:

```json
"Person": {
  "any_of": [
    {"fields": ["email"]},
    {"fields": ["phone"]},
    {"fields": ["name"]}
  ],
  "conflict_fields": ["email", "phone"]
}
```

Each `any_of` entry is a separate rule; every field inside an entry must
match. To require name AND email, use `{"fields": ["name", "email"]}`.
To disable name-only matching, remove `{"fields": ["name"]}`. Names are weak
identifiers: two people may share a name. Name matching is enabled here per
the requested configuration; use email/phone-only rules for stricter identity.

All available rules are checked. A match from one rule is not accepted if
another rule identifies a different canonical entity. Conflicting nonempty
email/phone values also raise `EntityConflictError`. The page transaction is
rolled back and ingestion fails for review; the system never silently merges
two existing canonical hubs. To change contact information deliberately,
review the conflict and update identity data explicitly.

Normalization:

| Mode | Behavior |
| --- | --- |
| `name` | Unicode NFKC, trim/collapse whitespace, casefold |
| `email` | Trim, casefold, basic address-format check |
| `phone` | Remove separators; accept only `+countrycode...` international numbers |
| `casefold` | Unicode NFKC, trim/collapse whitespace, casefold |
| `exact` | Trim Unicode-normalized strings; retain other primitive values |

Missing, empty, or invalid contact keys are never matched as identities.
Country codes are not guessed. If no configured key is available, create a
new hub rather than matching nulls. Phone normalization does not resolve local
numbers or extensions; provide a full international number for phone matching.

## Indexes and concurrency

Startup creates a composite Neo4j range index for each node label and each
dedup rule, including its canonical hub label. For example:

```cypher
CREATE INDEX ... IF NOT EXISTS FOR (n:Person) ON (n.scope, n.email);
CREATE INDEX ... IF NOT EXISTS FOR (n:Person) ON (n.scope, n.phone);
CREATE INDEX ... IF NOT EXISTS FOR (n:Employee) ON (n.scope, n.email);
```

It waits for indexes to become online. Identity lookups use equality predicates
with matching `USING INDEX` hints. `scope` is the config's `namespace`, which
separates projects/tenants. Use distinct namespaces for unrelated datasets.
The executing Neo4j account needs schema/index creation privileges.

Indexes accelerate matching; they do not alone guarantee OR-rule uniqueness.
The resolver acquires a transaction-held `EntityWriteLock` for each namespace
and hub type before checking identities. Locks are acquired in sorted order
to reduce deadlock risk. Ingestion workers using this resolver are serialized
within a hub type while resolving identities; unrelated types may proceed.
External writers must follow the same protocol. ID uniqueness constraints
protect canonical IDs and lock IDs; arbitrary property writes bypassing the
resolver are not covered by this application-level identity rule.

## Hub and spoke graph

The config maps both Person and Employee to the Person hub. For example,
an Employee mention and a Person mention with the same normalized email
refer to one canonical person:

```text
Page 1 --MENTIONS--> PageEntity(type=Person)   --REFERS_TO--> CanonicalEntity:Person
Page 2 --MENTIONS--> PageEntity(type=Employee) --REFERS_TO--> the same hub
```

The canonical node gains observed subtype labels (such as Employee).
`MERGE` creates it if missing; non-null normalized properties are updated
on a match, and omitted fields are preserved. Original names, emails, phones,
evidence, page images, and relationships stay on page-specific mention nodes.
Mention JSON includes `canonical_id`, linking local files to the graph hub.

Cross-model sameness is explicitly configured through `hub` and supported by
exact identity evidence. The program does not assume semantic similarity or
embedding proximity proves two people are identical. Add domain-specific
hub mappings and matching keys when modeling additional types.

Canonical nodes are retained when a page is reprocessed; the page's old
spokes/relations are replaced. Unreferenced canonical nodes are not garbage-
collected automatically. Historical aliases and fuzzy entity resolution are
not implemented.

Retrieve a hub's page evidence:

```python
context = await stores.graph.canonical_context(canonical_id)
```

```cypher
MATCH (p:Page)-[:MENTIONS]->(m:PageEntity)-[:REFERS_TO]->(c:CanonicalEntity:Person)
RETURN c.id, c.display_name, labels(c), p.page_number, p.image_path, m.evidence;
```

Config fingerprints are included in document IDs so changed extraction or
dedup settings create a new ingestion version. Existing data is not migrated
automatically; canonical matching still uses the configured namespace.

## Validation

Offline tests cover generated union schemas, rejected labels/configuration,
contact normalization, composite index statements, create/update behavior,
shared hubs across subtypes, ambiguous identity conflicts, and the actual
Trustcall graph's repair of invalid relationship references with a fake LLM.
Resolver database operations are tested with behavioral fakes, not a live
Neo4j server; deployment requires live integration verification.
