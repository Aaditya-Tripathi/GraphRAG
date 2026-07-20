from app.config import NEO4J_DATABASE
from app.constants import (
    EMBEDDING_DIMENSION,
    VECTOR_INDEX_NAME,
)
from app.database import create_driver


SCHEMA_QUERIES = [
    """
    CREATE CONSTRAINT conversation_id_unique IF NOT EXISTS
    FOR (conversation:Conversation)
    REQUIRE conversation.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT document_id_unique IF NOT EXISTS
    FOR (document:Document)
    REQUIRE document.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
    FOR (chunk:Chunk)
    REQUIRE chunk.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT entity_identity_unique IF NOT EXISTS
    FOR (entity:Entity)
    REQUIRE (
        entity.conversation_id,
        entity.normalized_name
    ) IS UNIQUE
    """,
    """
    CREATE INDEX chunk_conversation_id_index IF NOT EXISTS
    FOR (chunk:Chunk)
    ON (chunk.conversation_id)
    """,
    """
    CREATE INDEX document_conversation_id_index IF NOT EXISTS
    FOR (document:Document)
    ON (document.conversation_id)
    """,
]


VECTOR_INDEX_QUERY = f"""
CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
FOR (chunk:Chunk)
ON (chunk.embedding)
OPTIONS {{
    indexConfig: {{
        `vector.dimensions`: {EMBEDDING_DIMENSION},
        `vector.similarity_function`: 'cosine'
    }}
}}
"""


def setup_database() -> None:
    """Create the Neo4j schema required by the application."""

    with create_driver() as driver:
        for query in SCHEMA_QUERIES:
            driver.execute_query(
                query,
                database_=NEO4J_DATABASE,
            )

        driver.execute_query(
            VECTOR_INDEX_QUERY,
            database_=NEO4J_DATABASE,
        )
        driver.execute_query(
            "CALL db.awaitIndex($index_name, 60)",
            index_name=VECTOR_INDEX_NAME,
            database_=NEO4J_DATABASE,
        )

    print("Neo4j constraints and indexes are ready.")


if __name__ == "__main__":
    setup_database()
