from app import graph_retrieval


class FakeRecord(dict):
    def data(self) -> dict:
        return dict(self)


class FakeDriver:
    def __init__(self) -> None:
        self.query = ""
        self.parameters: dict = {}

    def execute_query(self, query: str, **kwargs):
        self.query = query
        self.parameters = kwargs

        return (
            [
                FakeRecord(
                    chunk_id="chunk-1",
                    chunk_index=0,
                    document_id="document-1",
                    document_title="Example",
                    entities=[
                        {
                            "id": "entity-a",
                            "name": "Project Atlas",
                            "type": "Project",
                        },
                        {
                            "id": "entity-b",
                            "name": "Robot",
                            "type": "Device",
                        },
                    ],
                    facts=[
                        {
                            "id": "fact-direct",
                            "source": "Project Atlas",
                            "predicate": "USES",
                            "target": "Robot",
                            "confidence": 0.7,
                            "source_chunk_id": "chunk-1",
                        },
                        {
                            "id": "fact-indirect",
                            "source": "Robot",
                            "predicate": "USES",
                            "target": "Scanner",
                            "confidence": 0.99,
                            "source_chunk_id": "other-chunk",
                        },
                    ],
                )
            ],
            None,
            None,
        )


def test_graph_context_is_scoped_and_prefers_direct_facts(
    monkeypatch,
) -> None:
    driver = FakeDriver()
    monkeypatch.setattr(
        graph_retrieval,
        "get_driver",
        lambda: driver,
    )

    result = graph_retrieval.retrieve_graph_context(
        conversation_id="conversation-a",
        chunk_ids=["chunk-1"],
        max_facts=2,
    )

    assert driver.parameters["conversation_id"] == "conversation-a"
    assert "related_entities" in driver.query
    assert {entity["id"] for entity in result["entities"]} == {
        "entity-a",
        "entity-b",
    }
    assert result["facts"][0]["id"] == "fact-direct"
