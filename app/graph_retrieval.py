from typing import Any

from app.config import NEO4J_DATABASE
from app.constants import (
    MAX_CONVERSATION_ID_LENGTH,
)
from app.database import get_driver
from app.validation import validate_required_text


def validate_chunk_ids(
    chunk_ids: list[str],
) -> list[str]:
    if not isinstance(chunk_ids, list):
        raise TypeError(
            "chunk_ids must be a list."
        )

    cleaned_ids = []

    for chunk_id in chunk_ids:
        if not isinstance(chunk_id, str):
            raise TypeError(
                "Every chunk ID must be a string."
            )

        cleaned_id = chunk_id.strip()

        if cleaned_id:
            cleaned_ids.append(cleaned_id)

    unique_ids = list(dict.fromkeys(cleaned_ids))

    if not unique_ids:
        raise ValueError(
            "At least one chunk ID is required."
        )

    return unique_ids


def retrieve_graph_context(
    conversation_id: str,
    chunk_ids: list[str],
    max_facts: int = 20,
) -> dict[str, Any]:
    """
    Traverse from retrieved chunks to mentioned entities
    and one-hop RELATED_TO graph facts.
    """

    conversation_id = validate_required_text(
        conversation_id,
        "conversation_id",
        max_length=MAX_CONVERSATION_ID_LENGTH,
    )

    chunk_ids = validate_chunk_ids(
        chunk_ids
    )

    if not isinstance(max_facts, int):
        raise TypeError(
            "max_facts must be an integer."
        )

    if max_facts < 1 or max_facts > 100:
        raise ValueError(
            "max_facts must be between 1 and 100."
        )

    query = """
    UNWIND $chunk_ids AS requested_chunk_id

    MATCH
        (document:Document {
            conversation_id: $conversation_id
        })
        -[:HAS_CHUNK]->
        (chunk:Chunk {
            id: requested_chunk_id,
            conversation_id: $conversation_id
        })

    OPTIONAL MATCH
        (chunk)
        -[:MENTIONS]->
        (entity:Entity {
            conversation_id: $conversation_id
        })

    OPTIONAL MATCH
        (entity)
        -[fact:RELATED_TO {
            conversation_id: $conversation_id
        }]-
        (related:Entity {
            conversation_id: $conversation_id
        })

    WITH
        document,
        chunk,

        collect(
            DISTINCT
            CASE
                WHEN entity IS NULL
                THEN null

                ELSE {
                    id: entity.id,
                    name: entity.name,
                    normalized_name:
                        entity.normalized_name,
                    type: entity.type
                }
            END
        ) AS entities,

        collect(
            DISTINCT
            CASE
                WHEN related IS NULL
                THEN null

                ELSE {
                    id: related.id,
                    name: related.name,
                    normalized_name:
                        related.normalized_name,
                    type: related.type
                }
            END
        ) AS related_entities,

        collect(
            DISTINCT
            CASE
                WHEN fact IS NULL
                THEN null

                ELSE {
                    id: fact.id,
                    source:
                        startNode(fact).name,
                    predicate:
                        fact.predicate,
                    target:
                        endNode(fact).name,
                    confidence:
                        coalesce(
                            fact.confidence,
                            0.0
                        ),
                    source_chunk_id:
                        fact.source_chunk_id,
                    document_id:
                        fact.document_id,
                    evidence:
                        fact.evidence
                }
            END
        ) AS facts

    RETURN
        chunk.id AS chunk_id,
        chunk.chunk_index AS chunk_index,
        document.id AS document_id,
        document.title AS document_title,
        entities + related_entities AS entities,
        facts

    ORDER BY chunk.chunk_index
    """

    records, _, _ = get_driver().execute_query(
        query,
        conversation_id=conversation_id,
        chunk_ids=chunk_ids,
        database_=NEO4J_DATABASE,
    )

    per_chunk: list[dict[str, Any]] = []

    unique_entities: dict[str, dict] = {}
    unique_facts: dict[str, dict] = {}
    document_titles: dict[str, str] = {}

    for record in records:
        row = record.data()

        entities = [
            dict(entity)
            for entity in row["entities"]
            if entity is not None
        ]

        facts = [
            dict(fact)
            for fact in row["facts"]
            if fact is not None
        ]

        document_titles[
            row["chunk_id"]
        ] = row["document_title"]

        for entity in entities:
            entity_key = (
                entity.get("id")
                or entity.get("normalized_name")
                or entity.get("name")
            )

            if entity_key:
                unique_entities[
                    str(entity_key)
                ] = entity

        for fact in facts:
            fact_key = fact.get("id")

            if not fact_key:
                fact_key = "|".join(
                    [
                        str(fact.get("source", "")),
                        str(fact.get("predicate", "")),
                        str(fact.get("target", "")),
                        str(
                            fact.get(
                                "source_chunk_id",
                                "",
                            )
                        ),
                    ]
                )

            unique_facts[str(fact_key)] = fact

        per_chunk.append(
            {
                "chunk_id": row["chunk_id"],
                "chunk_index": row["chunk_index"],
                "document_id": row["document_id"],
                "document_title": row["document_title"],
                "entities": entities,
                "facts": facts,
            }
        )

    requested_chunk_ids = set(chunk_ids)

    sorted_facts = sorted(
        unique_facts.values(),
        key=lambda fact: (
            fact.get("source_chunk_id")
            not in requested_chunk_ids,
            -float(fact.get("confidence", 0.0)),
        ),
    )

    return {
        "conversation_id": conversation_id,
        "chunks": per_chunk,
        "entities": list(
            unique_entities.values()
        ),
        "facts": sorted_facts[:max_facts],
        "document_titles": document_titles,
    }
