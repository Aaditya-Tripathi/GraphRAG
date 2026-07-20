from typing import Any

import neo4j
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.types import RetrieverResultItem

from app.config import NEO4J_DATABASE
from app.constants import (
    MAX_CONVERSATION_ID_LENGTH,
    MAX_QUESTION_LENGTH,
    MAX_TOP_K,
    VECTOR_INDEX_NAME,
)
from app.database import get_driver
from app.embeddings import get_embedder
from app.validation import validate_required_text


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
        max_length=MAX_CONVERSATION_ID_LENGTH,
    )

    question = validate_required_text(
        question,
        "question",
        max_length=MAX_QUESTION_LENGTH,
    )

    if not isinstance(top_k, int):
        raise TypeError("top_k must be an integer.")

    if top_k < 1 or top_k > MAX_TOP_K:
        raise ValueError(
            f"top_k must be between 1 and {MAX_TOP_K}."
        )

    retriever = VectorRetriever(
        driver=get_driver(),
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
