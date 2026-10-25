"""Configuration-driven Pydantic unions and exact identity normalization."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator


class EntityProperty(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "integer", "number", "boolean"] = "string"
    normalizer: Literal["name", "email", "phone", "casefold", "exact"] = "exact"
    required: bool = False


class EntityType(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hub: str
    description: str = ""
    properties: dict[str, EntityProperty] = Field(default_factory=dict)


class DedupRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fields: list[str] = Field(min_length=1)


class HubRules(BaseModel):
    model_config = ConfigDict(extra="forbid")
    any_of: list[DedupRule] = Field(min_length=1)
    conflict_fields: list[str] = Field(default_factory=list)


class EntitySchemaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    namespace: str = Field(min_length=1, max_length=100)
    node_types: dict[str, EntityType] = Field(min_length=1, max_length=30)
    dedup: dict[str, HubRules]

    @model_validator(mode="after")
    def validate_schema(self):
        reserved = {"id", "scope", "display_name", "created_at", "updated_at", "local_id", "evidence", "entity_type"}
        reserved_labels = {"Document", "Page", "Chunk", "PageEntity", "PageRelation", "CanonicalEntity", "EntityWriteLock"}
        for label, definition in self.node_types.items():
            for identifier in (label, definition.hub, *definition.properties):
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", identifier):
                    raise ValueError(f"Invalid Cypher identifier: {identifier}")
            if label in reserved_labels or definition.hub in reserved_labels:
                raise ValueError("Entity labels must not use reserved graph labels")
            if reserved.intersection(definition.properties):
                raise ValueError("Entity properties must not use reserved metadata fields")
            if "name" in definition.properties:
                raise ValueError("name is a built-in required field")
            if any(p.normalizer != "exact" and p.type != "string" for p in definition.properties.values()):
                raise ValueError("Text normalizers require string properties")
            if definition.hub not in self.dedup:
                raise ValueError(f"Missing dedup rules for {definition.hub}")
            available = {"name", *definition.properties}
            rules = self.dedup[definition.hub]
            for fields in [r.fields for r in rules.any_of] + [rules.conflict_fields]:
                if len(fields) != len(set(fields)) or not set(fields) <= available:
                    raise ValueError(f"Invalid dedup fields for {label}")
        if set(self.dedup) != {n.hub for n in self.node_types.values()}:
            raise ValueError("Dedup hubs must match configured node hubs")
        # A shared hub must use the same field representation for every subtype.
        for hub in self.dedup:
            members = [v for v in self.node_types.values() if v.hub == hub]
            fields = {f for r in self.dedup[hub].any_of for f in r.fields} | set(self.dedup[hub].conflict_fields)
            for field in fields - {"name"}:
                representations = {(m.properties[field].type, m.properties[field].normalizer) for m in members}
                if len(representations) != 1:
                    raise ValueError(f"Shared hub {hub} needs consistent normalization for {field}")
        return self

    @classmethod
    def default(cls):
        return cls.model_validate({
            "namespace": "project_knowledge",
            "node_types": {
                "Person": {"hub": "Person", "description": "A person mentioned in the document", "properties": {
                    "email": {"normalizer": "email"}, "phone": {"normalizer": "phone"}}},
                "Employee": {"hub": "Person", "description": "A person explicitly identified as an employee", "properties": {
                    "email": {"normalizer": "email"}, "phone": {"normalizer": "phone"}}},
                "Organization": {"hub": "Organization", "properties": {"domain": {"normalizer": "casefold"}}},
                "Equipment": {"hub": "Equipment", "properties": {"asset_tag": {"normalizer": "casefold"}}},
            },
            "dedup": {
                "Person": {"any_of": [{"fields": ["email"]}, {"fields": ["phone"]}, {"fields": ["name"]}],
                           "conflict_fields": ["email", "phone"]},
                "Organization": {"any_of": [{"fields": ["name"]}]},
                "Equipment": {"any_of": [{"fields": ["asset_tag"]}, {"fields": ["name"]}]},
            },
        })

    @classmethod
    def load(cls, path=None):
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8")) if path else cls.default()

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()

    def build_extraction_model(self, entity_base, knowledge_base):
        """LLM selects a typed node through the entity_type discriminator."""
        types = {"string": str, "integer": int, "number": float, "boolean": bool}
        models = []
        for label, definition in self.node_types.items():
            fields = {"entity_type": (Literal[label], Field(..., description=definition.description or label))}
            for name, prop in definition.properties.items():
                annotation = types[prop.type]
                fields[name] = (annotation if prop.required else annotation | None,
                                Field(... if prop.required else None, description=f"Observed {name}; do not infer missing values"))
            models.append(create_model(f"Extracted{label}", __base__=entity_base, **fields))
        entity_union = Annotated[Union[tuple(models)], Field(discriminator="entity_type")] if len(models) > 1 else models[0]
        return create_model("ConfiguredPageKnowledge", __base__=knowledge_base,
                            entities=(list[entity_union], Field(..., max_length=100)))

    @staticmethod
    def normalize(value, mode):
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        value = unicodedata.normalize("NFKC", value).strip()
        if not value:
            return None
        if mode == "phone":
            # No guessed country codes: only complete international numbers match.
            value = re.sub(r"[\s().-]", "", value)
            return value if re.fullmatch(r"\+[1-9][0-9]{6,14}", value) else None
        if mode == "email":
            return value.casefold() if re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value) else None
        return " ".join(value.split()).casefold() if mode in ("name", "casefold") else value

    def canonical_properties(self, entity):
        definition = self.node_types[entity["entity_type"]]
        props = {"name": self.normalize(entity["name"], "name")}
        for name, prop in definition.properties.items():
            value = self.normalize(entity.get(name), prop.normalizer)
            if value is not None:
                props[name] = value
        return {key: value for key, value in props.items() if value is not None}
