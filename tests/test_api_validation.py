import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import app


def test_invalid_query_requests_return_422() -> None:
    invalid_payloads = [
        {},
        {
            "conversation_id": "",
            "question": "What is Neo4j?",
            "top_k": 5,
        },
        {
            "conversation_id": "demo",
            "question": "",
            "top_k": 5,
        },
        {
            "conversation_id": "demo",
            "question": "What is Neo4j?",
            "top_k": 6,
        },
        {
            "conversation_id": "demo",
            "question": "What is Neo4j?",
            "top_k": 5,
            "provider": "unsupported-provider",
        },
    ]

    with TestClient(app) as client:
        for payload in invalid_payloads:
            response = client.post(
                "/api/query",
                json=payload,
            )
            assert response.status_code == 422


def test_failed_graph_build_rolls_back_new_document(
    monkeypatch,
) -> None:
    deleted: dict[str, str] = {}

    monkeypatch.setattr(
        main,
        "ingest_text",
        lambda **kwargs: {
            "conversation_id": "conversation-a",
            "document_id": "document-a",
        },
    )
    monkeypatch.setattr(
        main,
        "build_knowledge_graph",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("upstream model failed")
        ),
    )
    monkeypatch.setattr(
        main,
        "delete_document",
        lambda **kwargs: deleted.update(kwargs),
    )

    with pytest.raises(RuntimeError, match="upstream model failed"):
        main.run_ingestion_pipeline(
            conversation_id="conversation-a",
            document_title="Example",
            text="Example text",
        )

    assert deleted == {
        "conversation_id": "conversation-a",
        "document_id": "document-a",
    }
