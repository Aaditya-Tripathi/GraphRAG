from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from groq import BadRequestError

from app import llm
from app.main import model_http_exception
from app.schemas import KnowledgeExtraction


def test_structured_generation_falls_back_to_json_object(
    monkeypatch,
) -> None:
    body = {
        "error": {
            "code": "json_validate_failed",
        }
    }
    response = httpx.Response(
        400,
        request=httpx.Request(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
        ),
        json=body,
    )
    strict_error = BadRequestError(
        "Structured generation failed.",
        response=response,
        body=body,
    )
    fallback_completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"entities":[{"name":"Neo4j",'
                        '"type":"Technology"}]}'
                    )
                )
            )
        ]
    )
    invalid_completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"entities":"invalid",'
                        '"relationships":[]}'
                    )
                )
            )
        ]
    )
    create = Mock(
        side_effect=[
            strict_error,
            invalid_completion,
            fallback_completion,
        ]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=create,
            )
        )
    )
    monkeypatch.setattr(
        llm,
        "get_groq_client",
        lambda: client,
    )

    result = llm.generate_structured_data(
        system_prompt="Extract entities.",
        user_prompt="Neo4j is a graph database.",
        schema_name="knowledge_extraction",
        json_schema={
            "type": "object",
            "properties": {
                "entities": {"type": "array"},
                "relationships": {"type": "array"},
            },
            "required": [
                "entities",
                "relationships",
            ],
            "additionalProperties": False,
        },
        response_model=KnowledgeExtraction,
    )

    assert result.entities[0].name == "Neo4j"
    assert create.call_count == 3
    assert create.call_args_list[0].kwargs[
        "response_format"
    ]["type"] == "json_schema"
    for call in create.call_args_list[1:]:
        assert call.kwargs[
            "response_format"
        ] == {"type": "json_object"}

    rate_error = llm.LLMServiceError(
        "Rate limited.",
        code="rate_limit",
        retry_after_seconds=1778,
    )
    http_error = model_http_exception(rate_error)

    assert http_error.status_code == 429
    assert http_error.headers == {
        "Retry-After": "1778"
    }
    assert "30 minutes" in http_error.detail


def test_openrouter_text_request_uses_free_router_and_reasoning(
    monkeypatch,
) -> None:
    captured: dict = {}
    request_count = 0

    def fake_post(url, **kwargs):
        nonlocal request_count
        request_count += 1
        captured["url"] = url
        captured["payload"] = kwargs["json"]

        if request_count == 1:
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "error": {
                        "code": 502,
                        "message": "Temporary free-router failure",
                    }
                },
            )

        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {
                        "message": {
                            "content": "Grounded answer."
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(
        llm,
        "get_openrouter_api_key",
        lambda: "test-key",
    )
    monkeypatch.setattr(llm.httpx, "post", fake_post)

    answer = llm.generate_text(
        system_prompt="Use only the evidence.",
        user_prompt="Answer the question.",
        provider="openrouter",
    )

    assert answer == "Grounded answer."
    assert request_count == 2
    assert captured["url"] == llm.OPENROUTER_API_URL
    assert captured["payload"]["model"] == (
        "openai/gpt-oss-20b:free"
    )
    assert captured["payload"]["reasoning"] == {
        "effort": "low",
        "exclude": True,
    }


def test_openrouter_structured_output_retries_invalid_json(
    monkeypatch,
) -> None:
    responses = iter(
        [
            "not-json",
            '{"entities":[],"relationships":[]}',
        ]
    )
    formats: list[dict | None] = []

    def fake_completion(**kwargs):
        formats.append(kwargs.get("response_format"))
        return next(responses)

    monkeypatch.setattr(
        llm,
        "create_openrouter_completion",
        fake_completion,
    )

    result = llm.generate_structured_data(
        system_prompt="Extract facts.",
        user_prompt="There are no reliable facts.",
        schema_name="knowledge_extraction",
        json_schema={
            "type": "object",
            "properties": {
                "entities": {"type": "array"},
                "relationships": {"type": "array"},
            },
            "required": ["entities", "relationships"],
        },
        response_model=KnowledgeExtraction,
        provider="openrouter",
    )

    assert result.entities == []
    assert formats == [{"type": "json_object"}, None]

    tolerant_result = llm.validate_structured_content(
        """
        {
          "nodes": [
            {
              "name": "Neo4j",
              "category": "Technology",
              "description": "Ignored extra field"
            }
          ],
          "edges": [
            {
              "from": "Neo4j",
              "type": "stores",
              "to": "Graph Data"
            }
          ]
        }
        """,
        KnowledgeExtraction,
        provider="openrouter",
    )

    assert tolerant_result.entities[0].type == "Technology"
    assert tolerant_result.relationships[0].predicate == "STORES"
    assert tolerant_result.relationships[0].confidence == 0.5
