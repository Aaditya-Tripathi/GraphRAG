import re

from neo4j.exceptions import ClientError

from app.config import NEO4J_DATABASE
from app.database import create_driver


CONSTRAINT_QUERIES = [
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
    CREATE CONSTRAINT message_id_unique IF NOT EXISTS
    FOR (message:Message)
    REQUIRE message.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT entity_identity_unique IF NOT EXISTS
    FOR (entity:Entity)
    REQUIRE (
        entity.conversation_id,
        entity.normalized_name
    ) IS UNIQUE
    """,
]


RANGE_INDEX_QUERIES = [
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


FILTERABLE_VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
FOR (chunk:Chunk)
ON (chunk.embedding)
WITH [chunk.conversation_id]
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: 'cosine'
    }
}
"""


BASIC_VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
FOR (chunk:Chunk)
ON (chunk.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: 'cosine'
    }
}
"""


def get_server_version(driver) -> str:
    records, _, _ = driver.execute_query(
        """
        CALL dbms.components()
        YIELD versions
        RETURN versions[0] AS version
        """,
        database_=NEO4J_DATABASE,
    )

    if not records:
        return "unknown"

    return str(records[0]["version"])


def supports_filterable_vector_index(version: str) -> bool:
    match = re.match(r"(\d{4})\.(\d{1,2})", version)

    if not match:
        return False

    year = int(match.group(1))
    month = int(match.group(2))

    return (year, month) >= (2026, 1)


def run_query(driver, query: str) -> None:
    driver.execute_query(
        query,
        database_=NEO4J_DATABASE,
    )


def setup_database() -> None:
    with create_driver() as driver:
        version = get_server_version(driver)
        print(f"Neo4j server version: {version}")

        print("\nCreating constraints...")

        for query in CONSTRAINT_QUERIES:
            run_query(driver, query)

        print("Constraints created.")

        print("\nCreating normal indexes...")

        for query in RANGE_INDEX_QUERIES:
            run_query(driver, query)

        print("Normal indexes created.")

        print("\nCreating vector index...")

        if supports_filterable_vector_index(version):
            try:
                run_query(
                    driver,
                    FILTERABLE_VECTOR_INDEX_QUERY,
                )
                print(
                    "Created vector index with "
                    "conversation filtering support."
                )
            except ClientError as error:
                print(
                    "Filterable vector index was unavailable. "
                    "Using the compatible version instead."
                )
                print(f"Neo4j message: {error.message}")

                run_query(
                    driver,
                    BASIC_VECTOR_INDEX_QUERY,
                )
        else:
            run_query(
                driver,
                BASIC_VECTOR_INDEX_QUERY,
            )
            print(
                "Created standard vector index. "
                "Conversation filtering will be handled "
                "by the retrieval code."
            )

        print("\nDatabase setup completed successfully.")


if __name__ == "__main__":
    setup_database()
