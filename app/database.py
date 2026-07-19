from neo4j import Driver, GraphDatabase

from app.config import (
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
)


def create_driver() -> Driver:
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )


def verify_connection() -> dict:
    with create_driver() as driver:
        driver.verify_connectivity()

        records, _, _ = driver.execute_query(
            """
            RETURN
                $message AS message,
                1 + 1 AS result
            """,
            message="Python connected to Neo4j",
            database_=NEO4J_DATABASE,
        )

        if not records:
            raise RuntimeError("Neo4j returned no records.")

        return records[0].data()
