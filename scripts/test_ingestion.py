from pathlib import Path

from app.config import (
    EMBEDDING_DIMENSION,
    NEO4J_DATABASE,
)
from app.database import create_driver
from app.ingestion import ingest_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILE = PROJECT_ROOT / "sample_data" / "sample.txt"

TEST_CONVERSATION_ID = "step9-demo"


def remove_previous_test_data() -> None:
    """
    Delete data from an earlier run of this test so that
    the test can be repeated safely.
    """

    with create_driver() as driver:
        driver.execute_query(
            """
            MATCH (node)
            WHERE node.conversation_id = $conversation_id
            DETACH DELETE node
            """,
            conversation_id=TEST_CONVERSATION_ID,
            database_=NEO4J_DATABASE,
        )


def verify_stored_data() -> dict:
    with create_driver() as driver:
        records, _, _ = driver.execute_query(
            """
            MATCH
                (conversation:Conversation {
                    id: $conversation_id
                })
                -[:HAS_DOCUMENT]->
                (document:Document)
                -[:HAS_CHUNK]->
                (chunk:Chunk)

            WITH
                conversation,
                collect(DISTINCT document) AS documents,
                collect(DISTINCT chunk) AS chunks

            OPTIONAL MATCH
                (first:Chunk {
                    conversation_id: $conversation_id
                })
                -[next_relationship:NEXT_CHUNK]->
                (second:Chunk {
                    conversation_id: $conversation_id
                })

            RETURN
                conversation.id AS conversation_id,
                size(documents) AS document_count,
                size(chunks) AS chunk_count,
                count(DISTINCT next_relationship)
                    AS next_chunk_relationship_count,
                [chunk IN chunks | size(chunk.embedding)]
                    AS embedding_dimensions
            """,
            conversation_id=TEST_CONVERSATION_ID,
            database_=NEO4J_DATABASE,
        )

    if not records:
        raise RuntimeError(
            "The stored graph could not be found."
        )

    return records[0].data()


def main() -> None:
    remove_previous_test_data()

    sample_text = SAMPLE_FILE.read_text(
        encoding="utf-8-sig",
    )

    print("Starting text ingestion...")

    ingestion_result = ingest_text(
        conversation_id=TEST_CONVERSATION_ID,
        document_title="GraphRAG Sample Document",
        text=sample_text,
    )

    print()
    print("INGESTION RESULT")
    print("-" * 60)

    for key, value in ingestion_result.items():
        print(f"{key}: {value}")

    verification = verify_stored_data()

    print()
    print("NEO4J VERIFICATION")
    print("-" * 60)

    for key, value in verification.items():
        print(f"{key}: {value}")

    expected_chunks = ingestion_result["chunks_created"]

    if verification["document_count"] != 1:
        raise RuntimeError(
            "Expected exactly one stored document."
        )

    if verification["chunk_count"] != expected_chunks:
        raise RuntimeError(
            "The stored chunk count is incorrect."
        )

    expected_next_relationships = max(
        expected_chunks - 1,
        0,
    )

    if (
        verification["next_chunk_relationship_count"]
        != expected_next_relationships
    ):
        raise RuntimeError(
            "The NEXT_CHUNK relationship count is incorrect."
        )

    if not all(
        dimension == EMBEDDING_DIMENSION
        for dimension in verification[
            "embedding_dimensions"
        ]
    ):
        raise RuntimeError(
            "At least one stored embedding has "
            "the wrong dimension."
        )

    print()
    print("All ingestion checks passed.")


if __name__ == "__main__":
    main()
