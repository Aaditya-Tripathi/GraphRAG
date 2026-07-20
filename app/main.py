import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import (
    FastAPI,
    HTTPException,
    status,
)
from neo4j.exceptions import (
    ServiceUnavailable,
    SessionExpired,
)

from app.api_schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    TextIngestRequest,
    TextIngestResponse,
)
from app.config import GROQ_MODEL, OPENROUTER_MODEL
from app.constants import (
    EMBEDDING_MODEL,
)
from app.database import close_driver, verify_connection
from app.ingestion import delete_document, ingest_text
from app.knowledge_graph import (
    build_knowledge_graph,
)
from app.llm import LLMServiceError
from app.rag import answer_question


logger = logging.getLogger(__name__)


def model_error_detail(
    error: LLMServiceError,
) -> str:
    """Return a safe, useful message for an LLM failure."""

    provider_name = (
        "OpenRouter"
        if error.provider == "openrouter"
        else "Groq"
    )

    if (
        error.code == "rate_limit"
        and error.retry_after_seconds is not None
    ):
        if error.retry_after_seconds >= 120:
            wait_value = (
                error.retry_after_seconds + 59
            ) // 60
            wait_text = f"{wait_value} minutes"
        else:
            wait_text = (
                f"{error.retry_after_seconds} seconds"
            )

        return (
            f"{provider_name}'s request limit was reached. "
            f"Try again in about {wait_text}."
        )

    messages = {
        "rate_limit": (
            f"{provider_name}'s request limit was reached. "
            "Wait briefly and try again."
        ),
        "connection": (
            f"{provider_name} could not be reached. Check your "
            "internet connection and try again."
        ),
        "timeout": (
            f"{provider_name} took too long to respond. "
            "Please try again."
        ),
        "structured_output": (
            f"{provider_name} could not extract valid graph data "
            "from one section. Please try again."
        ),
        "configuration": (
            f"{provider_name} is not configured. Add its API key "
            "to the .env file and restart the backend."
        ),
    }

    return messages.get(
        error.code,
        f"{provider_name} could not complete the request. Please try again.",
    )


def model_http_exception(
    error: LLMServiceError,
) -> HTTPException:
    """Map a Groq failure to the appropriate HTTP response."""

    if error.code == "rate_limit":
        headers = None

        if error.retry_after_seconds is not None:
            headers = {
                "Retry-After": str(
                    error.retry_after_seconds
                )
            }

        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=model_error_detail(error),
            headers=headers,
        )

    if error.code == "configuration":
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=model_error_detail(error),
        )

    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=model_error_detail(error),
    )


def database_unavailable_detail() -> str:
    return (
        "Neo4j is temporarily unavailable. Check your "
        "internet connection and AuraDB status, then try again."
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Release the shared Neo4j connection pool on shutdown."""

    yield
    close_driver()


app = FastAPI(
    title="Knowledge Graph RAG API",
    description=(
        "Conversation-isolated GraphRAG using "
        "Neo4j vector retrieval, graph traversal, "
        "and Groq."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def run_ingestion_pipeline(
    *,
    conversation_id: str,
    document_title: str,
    text: str,
    provider: str = "groq",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Store one document and compensate if graph creation fails."""

    ingestion_result = ingest_text(
        conversation_id=conversation_id,
        document_title=document_title,
        text=text,
    )

    try:
        graph_result = build_knowledge_graph(
            conversation_id=conversation_id,
            document_id=ingestion_result["document_id"],
            provider=provider,
        )
    except Exception:
        try:
            delete_document(
                conversation_id=conversation_id,
                document_id=ingestion_result["document_id"],
            )
        except Exception as cleanup_error:
            logger.error(
                "Ingestion rollback failed (%s).",
                type(cleanup_error).__name__,
            )
        raise

    return ingestion_result, graph_result


def format_ingestion_response(
    ingestion_result: dict[str, Any],
    graph_result: dict[str, Any],
    provider: str,
) -> dict[str, Any]:
    """Build the response for text ingestion."""

    return {
        "conversation_id": ingestion_result["conversation_id"],
        "document_id": ingestion_result["document_id"],
        "document_title": ingestion_result["document_title"],
        "chunks_created": ingestion_result["chunks_created"],
        "next_chunk_relationships": ingestion_result[
            "next_chunk_relationships"
        ],
        "embedding_dimension": ingestion_result[
            "embedding_dimension"
        ],
        "provider": provider,
        "knowledge_graph": {
            "chunks_processed": graph_result["chunks_processed"],
            "entities_extracted": graph_result[
                "entities_extracted"
            ],
            "relationships_extracted": graph_result[
                "relationships_extracted"
            ],
            "mentions_processed": graph_result[
                "mentions_processed"
            ],
            "facts_processed": graph_result["facts_processed"],
        },
    }


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Knowledge Graph RAG API",
        "documentation": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health_check() -> dict:
    """
    Verify that the API and Neo4j connection are working.

    This endpoint does not call Groq, so it does not
    consume an LLM request.
    """

    try:
        verify_connection()

        return {
            "status": "ok",
            "neo4j": "connected",
            "embedding_model": EMBEDDING_MODEL,
            "groq_model": GROQ_MODEL,
            "openrouter_model": OPENROUTER_MODEL,
        }

    except Exception as error:
        logger.error(
            "Health check failed (%s).",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail="Neo4j connection failed.",
        ) from error


@app.post(
    "/api/ingest/text",
    response_model=TextIngestResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_text_endpoint(
    request: TextIngestRequest,
) -> dict:
    """
    Ingest text, generate embeddings, store chunks,
    extract knowledge, and build the Neo4j graph.
    """

    try:
        ingestion_result, graph_result = run_ingestion_pipeline(
            conversation_id=request.conversation_id,
            document_title=request.document_title,
            text=request.text,
            provider=request.provider,
        )

        return format_ingestion_response(
            ingestion_result,
            graph_result,
            request.provider,
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=str(error),
        ) from error

    except LLMServiceError as error:
        logger.warning(
            "Text ingestion model request failed (%s).",
            error.code,
        )
        raise model_http_exception(error) from error

    except (ServiceUnavailable, SessionExpired) as error:
        logger.warning(
            "Text ingestion database request failed (%s).",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=database_unavailable_detail(),
        ) from error

    except Exception as error:
        logger.error(
            "Text ingestion failed (%s).",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Text ingestion failed. "
                "Check the API server logs."
            ),
        ) from error


@app.post(
    "/api/query",
    response_model=QueryResponse,
)
def query_endpoint(
    request: QueryRequest,
) -> dict:
    """
    Run vector retrieval, graph traversal, and
    grounded answer generation.
    """

    try:
        return answer_question(
            conversation_id=request.conversation_id,
            question=request.question,
            top_k=request.top_k,
            provider=request.provider,
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=str(error),
        ) from error

    except LLMServiceError as error:
        logger.warning(
            "Query model request failed (%s).",
            error.code,
        )
        raise model_http_exception(error) from error

    except (ServiceUnavailable, SessionExpired) as error:
        logger.warning(
            "Query database request failed (%s).",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=database_unavailable_detail(),
        ) from error

    except Exception as error:
        logger.error(
            "GraphRAG query failed (%s).",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "GraphRAG query failed. "
                "Check the API server logs."
            ),
        ) from error
