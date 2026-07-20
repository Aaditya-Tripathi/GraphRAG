import re

from app.llm import generate_structured_data
from app.schemas import KnowledgeExtraction


ALLOWED_ENTITY_TYPES = {
    "Person",
    "Organization",
    "Technology",
    "Concept",
    "Product",
    "Project",
    "Place",
    "Event",
    "Process",
    "Document",
    "Device",
    "System",
    "Metric",
    "Other",
}


KNOWLEDGE_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 200,
                    },
                    "type": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 50,
                    },
                },
                "required": [
                    "name",
                    "type",
                ],
                "additionalProperties": False,
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string"
                    },
                    "predicate": {
                        "type": "string"
                    },
                    "target": {
                        "type": "string"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": [
                    "source",
                    "predicate",
                    "target",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "entities",
        "relationships",
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """
You extract a compact knowledge graph from a text chunk.

Rules:

1. Extract only facts explicitly supported by the text.
2. Do not add outside knowledge.
3. Use short, meaningful entity names.
4. Entity names must use the same spelling everywhere.
5. Prefer these entity types: Person, Organization, Technology,
   Concept, Product, Project, Place, Event, Process, Document,
   Device, System, Metric, or Other. When unsure, use Other.
6. Every relationship source and target must appear in entities.
7. Relationship predicates must be concise uppercase snake case.
8. Examples of predicates:
   SUPPORTS
   USES
   PART_OF
   STORES
   PREVENTS
   CONNECTS_TO
   REPRESENTED_AS
9. Confidence must be between 0 and 1.
10. Extract no more than 15 entities.
11. Extract no more than 20 relationships.
12. Return empty arrays when no reliable facts are present.
"""


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name for comparison."""

    normalized = name.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(
        r"[^\w\s-]",
        "",
        normalized,
    )

    return normalized


def normalize_entity_type(value: str) -> str:
    """Map model-suggested categories into a controlled set."""

    cleaned = value.strip().title()

    aliases = {
        "Company": "Organization",
        "Business": "Organization",
        "Location": "Place",
        "City": "Place",
        "Country": "Place",
        "Software": "Technology",
        "Database": "Technology",
        "Tool": "Technology",
        "Machine": "Device",
        "Robot": "Device",
        "Application": "System",
        "Platform": "System",
        "Measurement": "Metric",
        "Statistic": "Metric",
    }

    normalized = aliases.get(cleaned, cleaned)

    if normalized not in ALLOWED_ENTITY_TYPES:
        return "Other"

    return normalized


def extract_knowledge(
    chunk_text: str,
    provider: str = "groq",
) -> KnowledgeExtraction:
    """Extract entities and relationships from one chunk."""

    if not isinstance(chunk_text, str):
        raise TypeError(
            "chunk_text must be a string."
        )

    chunk_text = chunk_text.strip()

    if not chunk_text:
        raise ValueError(
            "chunk_text cannot be empty."
        )

    result = generate_structured_data(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            "Extract entities and relationships from "
            "this text:\n\n"
            f"{chunk_text}"
        ),
        schema_name="knowledge_extraction",
        json_schema=KNOWLEDGE_EXTRACTION_SCHEMA,
        response_model=KnowledgeExtraction,
        provider=provider,
    )

    if not isinstance(result, KnowledgeExtraction):
        result = KnowledgeExtraction.model_validate(
            result
        )

    valid_entities = []
    entity_names = set()

    for entity in result.entities:
        normalized_name = normalize_entity_name(
            entity.name
        )

        if normalized_name and normalized_name not in entity_names:
            valid_entities.append(entity)
            entity_names.add(normalized_name)

        if len(valid_entities) == 15:
            break

    valid_relationships = []

    for relationship in result.relationships:
        source_name = normalize_entity_name(
            relationship.source
        )

        target_name = normalize_entity_name(
            relationship.target
        )

        if (
            source_name in entity_names
            and target_name in entity_names
            and source_name != target_name
        ):
            valid_relationships.append(relationship)

        if len(valid_relationships) == 20:
            break

    return KnowledgeExtraction(
        entities=valid_entities,
        relationships=valid_relationships,
    )
