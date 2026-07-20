from uuid import uuid4

from app.config import NEO4J_DATABASE
from app.constants import (
    MAX_CONVERSATION_ID_LENGTH,
    MAX_DOCUMENT_CHARACTERS,
    MAX_DOCUMENT_TITLE_LENGTH,
)
from app.database import get_driver
from app.embeddings import embed_texts
from app.text_processing import clean_text, split_text
from app.validation import validate_required_text


def ingest_text(
    conversation_id: str,
    document_title: str,
    text: str,
) -> dict:
    """
    Split text, create embeddings, and store the resulting
    conversation, document, and chunks in Neo4j.
    """

    conversation_id = validate_required_text(
        conversation_id,
        "conversation_id",
        max_length=MAX_CONVERSATION_ID_LENGTH,
    )

    document_title = validate_required_text(
        document_title,
        "document_title",
        max_length=MAX_DOCUMENT_TITLE_LENGTH,
    )

    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError(
            "The supplied document text is empty."
        )

    if len(cleaned_text) > MAX_DOCUMENT_CHARACTERS:
        raise ValueError(
            "Document text cannot exceed "
            f"{MAX_DOCUMENT_CHARACTERS:,} characters."
        )

    chunks = split_text(cleaned_text)
    embeddings = embed_texts(chunks)

    if len(chunks) != len(embeddings):
        raise RuntimeError(
            "The number of chunks does not match "
            "the number of embeddings."
        )

    document_id = str(uuid4())
    chunk_records: list[dict] = []

    for index, (chunk_text, embedding) in enumerate(
        zip(chunks, embeddings)
    ):
        chunk_records.append(
            {
                "id": str(uuid4()),
                "text": chunk_text,
                "chunk_index": index,
                "embedding": embedding,
            }
        )

    chunk_pairs = [
        {
            "current_id": chunk_records[index]["id"],
            "next_id": chunk_records[index + 1]["id"],
        }
        for index in range(len(chunk_records) - 1)
    ]

    create_nodes_query = """
    MERGE (
        conversation:Conversation {
            id: $conversation_id
        }
    )

    ON CREATE SET
        conversation.created_at = datetime()

    SET
        conversation.conversation_id = $conversation_id,
        conversation.updated_at = datetime()

    CREATE (
        document:Document {
            id: $document_id,
            conversation_id: $conversation_id,
            title: $document_title,
            source_type: 'text',
            character_count: $character_count,
            chunk_count: size($chunks),
            created_at: datetime()
        }
    )

    CREATE
        (conversation)
        -[:HAS_DOCUMENT {
            conversation_id: $conversation_id,
            document_id: $document_id,
            created_at: datetime()
        }]->
        (document)

    WITH document

    UNWIND $chunks AS chunk_data

    CREATE (
        chunk:Chunk {
            id: chunk_data.id,
            conversation_id: $conversation_id,
            document_id: $document_id,
            chunk_index: chunk_data.chunk_index,
            text: chunk_data.text,
            embedding: chunk_data.embedding,
            created_at: datetime()
        }
    )

    CREATE
        (document)
        -[:HAS_CHUNK {
            conversation_id: $conversation_id,
            document_id: $document_id,
            chunk_id: chunk_data.id,
            created_at: datetime()
        }]->
        (chunk)

    WITH
        document,
        count(chunk) AS chunks_created

    RETURN
        document.id AS document_id,
        document.title AS document_title,
        chunks_created
    """

    create_sequence_query = """
    UNWIND $chunk_pairs AS pair

    MATCH (
        current_chunk:Chunk {
            id: pair.current_id
        }
    )

    MATCH (
        next_chunk:Chunk {
            id: pair.next_id
        }
    )

    CREATE
        (current_chunk)
        -[:NEXT_CHUNK {
            conversation_id: $conversation_id,
            source_chunk_id: pair.current_id,
            target_chunk_id: pair.next_id,
            created_at: datetime()
        }]->
        (next_chunk)

    RETURN
        count(*) AS relationships_created
    """

    def store_document(transaction):
        record = transaction.run(
            create_nodes_query,
            conversation_id=conversation_id,
            document_id=document_id,
            document_title=document_title,
            character_count=len(cleaned_text),
            chunks=chunk_records,
        ).single()

        if record is None:
            raise RuntimeError(
                "Neo4j did not return an ingestion result."
            )

        if chunk_pairs:
            transaction.run(
                create_sequence_query,
                conversation_id=conversation_id,
                chunk_pairs=chunk_pairs,
            ).consume()

        return record.data()

    with get_driver().session(
        database=NEO4J_DATABASE
    ) as session:
        database_result = session.execute_write(
            store_document
        )

    return {
        "conversation_id": conversation_id,
        "document_id": database_result["document_id"],
        "document_title": database_result[
            "document_title"
        ],
        "chunks_created": database_result[
            "chunks_created"
        ],
        "next_chunk_relationships": len(chunk_pairs),
        "embedding_dimension": len(embeddings[0]),
    }


def delete_document(
    conversation_id: str,
    document_id: str,
) -> None:
    """Remove a newly ingested document after graph-build failure."""

    conversation_id = validate_required_text(
        conversation_id,
        "conversation_id",
        max_length=MAX_CONVERSATION_ID_LENGTH,
    )
    document_id = validate_required_text(
        document_id,
        "document_id",
    )

    queries = [
        """
        MATCH ()-[fact:RELATED_TO {
            conversation_id: $conversation_id,
            document_id: $document_id
        }]->()
        DELETE fact
        """,
        """
        MATCH
            (document:Document {
                id: $document_id,
                conversation_id: $conversation_id
            })
            -[:HAS_CHUNK]->
            (chunk:Chunk)
        DETACH DELETE chunk
        """,
        """
        MATCH (document:Document {
            id: $document_id,
            conversation_id: $conversation_id
        })
        DETACH DELETE document
        """,
        """
        MATCH (entity:Entity {
            conversation_id: $conversation_id
        })
        WHERE
            NOT EXISTS {
                MATCH ()-[:MENTIONS]->(entity)
            }
            AND NOT EXISTS {
                MATCH (entity)-[:RELATED_TO]-()
            }
        DETACH DELETE entity
        """,
        """
        MATCH (conversation:Conversation {
            id: $conversation_id
        })
        WHERE NOT EXISTS {
            MATCH (conversation)-[:HAS_DOCUMENT]->()
        }
        DETACH DELETE conversation
        """,
    ]

    driver = get_driver()

    for query in queries:
        driver.execute_query(
            query,
            conversation_id=conversation_id,
            document_id=document_id,
            database_=NEO4J_DATABASE,
        )
