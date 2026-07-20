from app import rag


def test_empty_conversation_returns_no_information(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        rag,
        "retrieve_chunks",
        lambda **kwargs: [],
    )

    response = rag.answer_question(
        conversation_id="empty-conversation",
        question="What is Neo4j?",
    )

    assert "don't know" in response["answer"].lower()
    assert response["results"] == []
    assert response["facts"] == []


def test_basic_rag_response_contains_evidence(
    monkeypatch,
) -> None:
    selected_provider: dict[str, str] = {}
    monkeypatch.setattr(
        rag,
        "retrieve_chunks",
        lambda **kwargs: [
            {
                "chunk_id": "chunk-1",
                "conversation_id": "conversation-a",
                "document_id": "document-a",
                "chunk_index": 0,
                "text": "Neo4j stores connected data.",
                "score": 0.92,
            }
        ],
    )
    monkeypatch.setattr(
        rag,
        "retrieve_graph_context",
        lambda **kwargs: {
            "document_titles": {
                "chunk-1": "Neo4j Notes"
            },
            "entities": [
                {
                    "name": "Neo4j",
                    "type": "Technology",
                }
            ],
            "facts": [
                {
                    "source": "Neo4j",
                    "predicate": "STORES",
                    "target": "Connected Data",
                    "confidence": 0.9,
                    "source_chunk_id": "chunk-1",
                }
            ],
        },
    )
    monkeypatch.setattr(
        rag,
        "generate_text",
        lambda **kwargs: (
            selected_provider.update(
                provider=kwargs["provider"]
            )
            or "Neo4j stores connected data [Chunk 1] [Fact 1]."
        ),
    )

    response = rag.answer_question(
        conversation_id="conversation-a",
        question="What does Neo4j store?",
        provider="openrouter",
    )

    assert "[Chunk 1]" in response["answer"]
    assert len(response["results"]) == 1
    assert len(response["facts"]) == 1
    assert response["provider"] == "openrouter"
    assert selected_provider["provider"] == "openrouter"
