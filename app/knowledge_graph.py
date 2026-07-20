import hashlib
import logging
from typing import Any

from app.config import NEO4J_DATABASE
from app.constants import (
    MAX_CONVERSATION_ID_LENGTH,
)
from app.database import get_driver
from app.entity_extraction import (
    extract_knowledge,
    normalize_entity_name,
    normalize_entity_type,
)
from app.validation import validate_required_text


logger = logging.getLogger(__name__)


def create_entity_id(
    conversation_id: str,
    normalized_name: str,
) -> str:
    raw_value = (
        f"{conversation_id}|{normalized_name}"
    )

    return hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()[:32]


def create_fact_id(
    conversation_id: str,
    source_name: str,
    predicate: str,
    target_name: str,
    source_chunk_id: str,
) -> str:
    raw_value = "|".join(
        [
            conversation_id,
            source_name,
            predicate,
            target_name,
            source_chunk_id,
        ]
    )

    return hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()[:32]


def get_conversation_chunks(
    conversation_id: str,
) -> list[dict[str, Any]]:
    """Retrieve every chunk stored for one conversation."""

    query = """
    MATCH (chunk:Chunk {
        conversation_id: $conversation_id
    })

    RETURN
        chunk.id AS chunk_id,
        chunk.document_id AS document_id,
        chunk.chunk_index AS chunk_index,
        chunk.text AS text

    ORDER BY
        chunk.document_id,
        chunk.chunk_index
    """

    records, _, _ = get_driver().execute_query(
        query,
        conversation_id=conversation_id,
        database_=NEO4J_DATABASE,
    )

    return [
        record.data()
        for record in records
    ]


def get_document_chunks(
    conversation_id: str,
    document_id: str,
) -> list[dict[str, Any]]:
    """Retrieve only the chunks belonging to one document."""

    query = """
    MATCH
        (document:Document {
            id: $document_id,
            conversation_id: $conversation_id
        })
        -[:HAS_CHUNK]->
        (chunk:Chunk {
            conversation_id: $conversation_id
        })

    RETURN
        chunk.id AS chunk_id,
        chunk.document_id AS document_id,
        chunk.chunk_index AS chunk_index,
        chunk.text AS text

    ORDER BY chunk.chunk_index
    """

    records, _, _ = get_driver().execute_query(
        query,
        conversation_id=conversation_id,
        document_id=document_id,
        database_=NEO4J_DATABASE,
    )

    return [
        record.data()
        for record in records
    ]


def prepare_entities(
    conversation_id: str,
    extracted_entities,
) -> list[dict[str, Any]]:
    """Normalize and deduplicate entities before storage."""

    prepared_entities: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for entity in extracted_entities:
        normalized_name = normalize_entity_name(
            entity.name
        )
        normalized_type = normalize_entity_type(
            entity.type
        )

        if not normalized_name:
            continue

        if normalized_name in seen_names:
            continue

        seen_names.add(normalized_name)

        prepared_entities.append(
            {
                "id": create_entity_id(
                    conversation_id,
                    normalized_name,
                ),
                "name": entity.name.strip(),
                "normalized_name": normalized_name,
                "type": normalized_type,
            }
        )

    return prepared_entities


def prepare_relationships(
    conversation_id: str,
    chunk_id: str,
    chunk_text: str,
    extracted_relationships,
    available_entity_names: set[str],
) -> list[dict[str, Any]]:
    """Normalize and validate relationships before storage."""

    prepared_relationships: list[
        dict[str, Any]
    ] = []

    seen_facts: set[
        tuple[str, str, str]
    ] = set()

    evidence_excerpt = chunk_text.strip()

    for relationship in extracted_relationships:
        source_name = normalize_entity_name(
            relationship.source
        )

        target_name = normalize_entity_name(
            relationship.target
        )

        predicate = relationship.predicate.strip().upper()

        if source_name not in available_entity_names:
            continue

        if target_name not in available_entity_names:
            continue

        if source_name == target_name:
            continue

        fact_key = (
            source_name,
            predicate,
            target_name,
        )

        if fact_key in seen_facts:
            continue

        seen_facts.add(fact_key)

        prepared_relationships.append(
            {
                "id": create_fact_id(
                    conversation_id,
                    source_name,
                    predicate,
                    target_name,
                    chunk_id,
                ),
                "source_normalized_name": source_name,
                "target_normalized_name": target_name,
                "predicate": predicate,
                "confidence": float(
                    relationship.confidence
                ),
                "evidence": evidence_excerpt,
            }
        )

    return prepared_relationships


def store_entities_and_mentions(
    conversation_id: str,
    document_id: str,
    chunk_id: str,
    entities: list[dict[str, Any]],
) -> int:
    """
    Store Entity nodes and connect the chunk to them
    using MENTIONS relationships.
    """

    if not entities:
        return 0

    query = """
    UNWIND $entities AS entity_data

    MATCH (chunk:Chunk {
        id: $chunk_id,
        conversation_id: $conversation_id
    })

    MERGE (entity:Entity {
        conversation_id: $conversation_id,
        normalized_name:
            entity_data.normalized_name
    })

    ON CREATE SET
        entity.id = entity_data.id,
        entity.name = entity_data.name,
        entity.type = entity_data.type,
        entity.created_at = datetime()

    ON MATCH SET
        entity.updated_at = datetime(),
        entity.type =
            CASE
                WHEN entity.type = 'Other'
                     AND entity_data.type <> 'Other'
                THEN entity_data.type
                ELSE entity.type
            END

    MERGE
        (chunk)
        -[mention:MENTIONS]->
        (entity)

    ON CREATE SET
        mention.created_at = datetime()

    SET
        mention.conversation_id =
            $conversation_id,
        mention.document_id =
            $document_id,
        mention.source_chunk_id =
            $chunk_id,
        mention.updated_at =
            datetime()

    RETURN count(mention) AS mentions_processed
    """

    records, _, _ = get_driver().execute_query(
        query,
        conversation_id=conversation_id,
        document_id=document_id,
        chunk_id=chunk_id,
        entities=entities,
        database_=NEO4J_DATABASE,
    )

    if not records:
        return 0

    return int(
        records[0]["mentions_processed"]
    )


def store_graph_facts(
    conversation_id: str,
    document_id: str,
    chunk_id: str,
    relationships: list[dict[str, Any]],
) -> int:
    """
    Store graph facts using the controlled
    RELATED_TO relationship type.
    """

    if not relationships:
        return 0

    query = """
    UNWIND $relationships AS relationship_data

    MATCH (source:Entity {
        conversation_id: $conversation_id,
        normalized_name:
            relationship_data.source_normalized_name
    })

    MATCH (target:Entity {
        conversation_id: $conversation_id,
        normalized_name:
            relationship_data.target_normalized_name
    })

    MERGE
        (source)
        -[fact:RELATED_TO {
            conversation_id:
                $conversation_id,
            source_chunk_id:
                $chunk_id,
            predicate:
                relationship_data.predicate
        }]->
        (target)

    ON CREATE SET
        fact.id =
            relationship_data.id,
        fact.created_at =
            datetime()

    SET
        fact.document_id =
            $document_id,
        fact.confidence =
            CASE
                WHEN fact.confidence IS NULL
                     OR relationship_data.confidence
                        > fact.confidence
                THEN relationship_data.confidence
                ELSE fact.confidence
            END,
        fact.evidence =
            relationship_data.evidence,
        fact.updated_at =
            datetime()

    RETURN count(fact) AS facts_processed
    """

    records, _, _ = get_driver().execute_query(
        query,
        conversation_id=conversation_id,
        document_id=document_id,
        chunk_id=chunk_id,
        relationships=relationships,
        database_=NEO4J_DATABASE,
    )

    if not records:
        return 0

    return int(
        records[0]["facts_processed"]
    )


def build_knowledge_graph(
    conversation_id: str,
    document_id: str | None = None,
    provider: str = "groq",
) -> dict[str, Any]:
    """
    Extract and store knowledge for one document, or rebuild
    every chunk in a conversation when no document is supplied.
    """

    conversation_id = validate_required_text(
        conversation_id,
        "conversation_id",
        max_length=MAX_CONVERSATION_ID_LENGTH,
    )

    if document_id is not None:
        if not isinstance(document_id, str):
            raise TypeError(
                "document_id must be a string."
            )

        document_id = document_id.strip()

        if not document_id:
            raise ValueError(
                "document_id cannot be empty."
            )

        chunks = get_document_chunks(
            conversation_id=conversation_id,
            document_id=document_id,
        )

    else:
        chunks = get_conversation_chunks(
            conversation_id
        )

    if not chunks:
        if document_id:
            raise ValueError(
                "No chunks were found for this document "
                "inside the selected conversation."
            )

        raise ValueError(
            "No chunks were found for this conversation."
        )

    total_entities_extracted = 0
    total_relationships_extracted = 0
    total_mentions_processed = 0
    total_facts_processed = 0
    chunk_results: list[dict[str, Any]] = []

    for position, chunk in enumerate(
        chunks,
        start=1,
    ):
        logger.info(
            "Processing knowledge graph chunk %s of %s.",
            position,
            len(chunks),
        )

        extraction = extract_knowledge(
            chunk["text"],
            provider=provider,
        )

        entities = prepare_entities(
            conversation_id,
            extraction.entities,
        )

        available_entity_names = {
            entity["normalized_name"]
            for entity in entities
        }

        relationships = prepare_relationships(
            conversation_id=conversation_id,
            chunk_id=chunk["chunk_id"],
            chunk_text=chunk["text"],
            extracted_relationships=(
                extraction.relationships
            ),
            available_entity_names=(
                available_entity_names
            ),
        )

        mentions_processed = store_entities_and_mentions(
            conversation_id=conversation_id,
            document_id=chunk["document_id"],
            chunk_id=chunk["chunk_id"],
            entities=entities,
        )

        facts_processed = store_graph_facts(
            conversation_id=conversation_id,
            document_id=chunk["document_id"],
            chunk_id=chunk["chunk_id"],
            relationships=relationships,
        )

        total_entities_extracted += len(entities)
        total_relationships_extracted += len(
            relationships
        )
        total_mentions_processed += mentions_processed
        total_facts_processed += facts_processed

        chunk_results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "entities_extracted": len(entities),
                "relationships_extracted": len(
                    relationships
                ),
                "mentions_processed": mentions_processed,
                "facts_processed": facts_processed,
            }
        )

    return {
        "conversation_id": conversation_id,
        "document_id": document_id,
        "chunks_processed": len(chunks),
        "entities_extracted": total_entities_extracted,
        "relationships_extracted": (
            total_relationships_extracted
        ),
        "mentions_processed": total_mentions_processed,
        "facts_processed": total_facts_processed,
        "chunks": chunk_results,
    }
