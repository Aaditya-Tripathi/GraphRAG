from threading import Lock

from neo4j import Driver, GraphDatabase

from app.config import (
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
)


_driver: Driver | None = None
_driver_lock = Lock()


def create_driver() -> Driver:
    """Create an independent Neo4j driver."""

    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )


def get_driver() -> Driver:
    """Return the application-wide Neo4j connection pool."""

    global _driver

    if _driver is None:
        with _driver_lock:
            if _driver is None:
                _driver = create_driver()

    return _driver


def close_driver() -> None:
    """Close the shared Neo4j driver during application shutdown."""

    global _driver

    with _driver_lock:
        driver = _driver
        _driver = None

    if driver is not None:
        driver.close()


def verify_connection() -> dict:
    driver = get_driver()
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
