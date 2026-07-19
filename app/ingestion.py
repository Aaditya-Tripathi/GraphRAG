import hashlib
from uuid import uuid4

from app.config import NEO4J_DATABASE
from app.database import create_driver
from app.embeddings import embed_texts
from app.text_processing import clean_text, split_text


def validate_required_text(
    value: str,
    field_name: str,
) -> str:
    """Validate and normalize a required string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    cleaned_value = value.strip()

    if not cleaned_value:
        raise ValueError(f"{field_name} is required.")

    return cleaned_value


def create_content_hash(text: str) -> str:
    """Create a stable fingerprint for document content."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


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
    )

    document_title = validate_required_text(
        document_title,
        "document_title",
    )

    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError(
            "The supplied document text is empty."
        )

    chunks = split_text(cleaned_text)
    embeddings = embed_texts(chunks)

    if len(chunks) != len(embeddings):
        raise RuntimeError(
            "The number of chunks does not match "
            "the number of embeddings."
        )

    document_id = str(uuid4())
    content_hash = create_content_hash(cleaned_text)

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
            content_hash: $content_hash,
            character_count: $character_count,
            chunk_count: size($chunks),
            created_at: datetime()
        }
    )

    CREATE
        (conversation)
        -[:HAS_DOCUMENT]->
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
        -[:HAS_CHUNK]->
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

    MERGE
        (current_chunk)
        -[:NEXT_CHUNK]->
        (next_chunk)

    RETURN
        count(*) AS relationships_created
    """

    with create_driver() as driver:
        records, _, _ = driver.execute_query(
            create_nodes_query,
            conversation_id=conversation_id,
            document_id=document_id,
            document_title=document_title,
            content_hash=content_hash,
            character_count=len(cleaned_text),
            chunks=chunk_records,
            database_=NEO4J_DATABASE,
        )

        if not records:
            raise RuntimeError(
                "Neo4j did not return an ingestion result."
            )

        driver.execute_query(
            create_sequence_query,
            chunk_pairs=chunk_pairs,
            database_=NEO4J_DATABASE,
        )

    database_result = records[0]

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