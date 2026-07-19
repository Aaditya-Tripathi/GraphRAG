from typing import Any

import neo4j
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.types import RetrieverResultItem

from app.config import (
    NEO4J_DATABASE,
    VECTOR_INDEX_NAME,
)
from app.database import create_driver
from app.embeddings import get_embedder


def validate_required_text(
    value: str,
    field_name: str,
) -> str:
    """Validate a required text value."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    cleaned_value = value.strip()

    if not cleaned_value:
        raise ValueError(f"{field_name} is required.")

    return cleaned_value


def format_vector_result(
    record: neo4j.Record,
) -> RetrieverResultItem:
    """
    Convert a Neo4j vector-search record into a clean result.
    """

    node = record["node"]
    score = float(record["score"])

    metadata: dict[str, Any] = {
        "score": score,
        "chunk_id": node.get("id"),
        "conversation_id": node.get("conversation_id"),
        "document_id": node.get("document_id"),
        "chunk_index": node.get("chunk_index"),
    }

    return RetrieverResultItem(
        content=node.get("text", ""),
        metadata=metadata,
    )


def retrieve_chunks(
    conversation_id: str,
    question: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve up to top_k semantically relevant chunks belonging
    only to the selected conversation.
    """

    conversation_id = validate_required_text(
        conversation_id,
        "conversation_id",
    )

    question = validate_required_text(
        question,
        "question",
    )

    if not isinstance(top_k, int):
        raise TypeError("top_k must be an integer.")

    if top_k < 1 or top_k > 20:
        raise ValueError(
            "top_k must be between 1 and 20."
        )

    with create_driver() as driver:
        retriever = VectorRetriever(
            driver=driver,
            index_name=VECTOR_INDEX_NAME,
            embedder=get_embedder(),
            result_formatter=format_vector_result,
            neo4j_database=NEO4J_DATABASE,
        )

        search_result = retriever.search(
            query_text=question,
            top_k=top_k,
            filters={
                "conversation_id": {
                    "$eq": conversation_id,
                }
            },
        )

    results: list[dict] = []

    for item in search_result.items:
        metadata = item.metadata or {}

        results.append(
            {
                "chunk_id": metadata.get("chunk_id"),
                "conversation_id": metadata.get(
                    "conversation_id"
                ),
                "document_id": metadata.get("document_id"),
                "chunk_index": metadata.get("chunk_index"),
                "text": str(item.content),
                "score": float(
                    metadata.get("score", 0.0)
                ),
            }
        )

    return results
