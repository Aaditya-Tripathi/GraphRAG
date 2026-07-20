from types import SimpleNamespace

import pytest

from app import retrieval


class FakeDriver:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def test_retrieval_uses_conversation_filter_and_top_five(
    monkeypatch,
) -> None:
    captured: dict = {}

    class FakeVectorRetriever:
        def __init__(self, **kwargs):
            captured["database"] = kwargs["neo4j_database"]

        def search(self, **kwargs):
            captured.update(kwargs)
            items = [
                SimpleNamespace(
                    content=f"Chunk {index}",
                    metadata={
                        "chunk_id": f"chunk-{index}",
                        "conversation_id": "conversation-a",
                        "document_id": "document-a",
                        "chunk_index": index,
                        "score": 0.9 - (index * 0.01),
                    },
                )
                for index in range(5)
            ]
            return SimpleNamespace(items=items)

    monkeypatch.setattr(
        retrieval,
        "get_driver",
        lambda: FakeDriver(),
    )
    monkeypatch.setattr(
        retrieval,
        "get_embedder",
        lambda: object(),
    )
    monkeypatch.setattr(
        retrieval,
        "VectorRetriever",
        FakeVectorRetriever,
    )

    results = retrieval.retrieve_chunks(
        conversation_id="conversation-a",
        question="What is GraphRAG?",
        top_k=5,
    )

    assert len(results) == 5
    assert captured["top_k"] == 5
    assert captured["filters"] == {
        "conversation_id": {"$eq": "conversation-a"}
    }
    assert all(
        result["conversation_id"] == "conversation-a"
        for result in results
    )

    with pytest.raises(ValueError, match="between 1 and 5"):
        retrieval.retrieve_chunks(
            conversation_id="conversation-a",
            question="What is GraphRAG?",
            top_k=6,
        )
