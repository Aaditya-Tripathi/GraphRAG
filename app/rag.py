import json
from typing import Any

from app.graph_retrieval import (
    retrieve_graph_context,
)
from app.constants import (
    MAX_CONVERSATION_ID_LENGTH,
    MAX_QUESTION_LENGTH,
)
from app.llm import generate_text
from app.retrieval import retrieve_chunks
from app.validation import validate_required_text


RAG_SYSTEM_PROMPT = """
You are a grounded Knowledge Graph RAG answer generator.

You will receive:

1. Retrieved text chunks.
2. Extracted knowledge-graph facts.

Rules:

- Answer using only the supplied evidence.
- Do not use outside knowledge.
- Prefer explicit chunk text if a graph fact appears
  uncertain or conflicts with the chunks.
- Graph facts are machine-extracted and may be imperfect.
- Cite supporting chunks using [Chunk 1], [Chunk 2], etc.
- Cite supporting graph facts using [Fact 1], [Fact 2], etc.
- Do not cite evidence that does not support the claim.
- If the supplied evidence is insufficient, respond:
  "I don't know based on the stored information."
- Keep the answer concise but complete.
"""


def prepare_prompt_context(
    chunks: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> dict[str, Any]:
    prepared_chunks = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        prepared_chunks.append(
            {
                "reference": f"Chunk {index}",
                "chunk_id": chunk["chunk_id"],
                "document_title": chunk.get(
                    "document_title"
                ),
                "similarity_score": round(
                    float(chunk["score"]),
                    4,
                ),
                "text": chunk["text"],
            }
        )

    prepared_facts = []

    for index, fact in enumerate(
        facts,
        start=1,
    ):
        evidence = str(
            fact.get("evidence", "")
        ).strip()

        prepared_facts.append(
            {
                "reference": f"Fact {index}",
                "source": fact.get("source"),
                "predicate": fact.get("predicate"),
                "target": fact.get("target"),
                "confidence": round(
                    float(
                        fact.get(
                            "confidence",
                            0.0,
                        )
                    ),
                    4,
                ),
                "source_chunk_id": fact.get(
                    "source_chunk_id"
                ),
                "evidence": evidence,
            }
        )

    return {
        "chunks": prepared_chunks,
        "graph_facts": prepared_facts,
    }


def answer_question(
    conversation_id: str,
    question: str,
    top_k: int = 5,
    provider: str = "groq",
) -> dict[str, Any]:
    """
    Run vector retrieval, graph traversal, and grounded
    answer generation.
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

    vector_results = retrieve_chunks(
        conversation_id=conversation_id,
        question=question,
        top_k=top_k,
    )

    if not vector_results:
        return {
            "conversation_id": conversation_id,
            "question": question,
            "provider": provider,
            "answer": (
                "I don't know based on the stored "
                "information for this conversation."
            ),
            "results": [],
            "entities": [],
            "facts": [],
        }

    chunk_ids = [
        result["chunk_id"]
        for result in vector_results
    ]

    graph_context = retrieve_graph_context(
        conversation_id=conversation_id,
        chunk_ids=chunk_ids,
        max_facts=20,
    )

    document_titles = graph_context[
        "document_titles"
    ]

    enriched_results = []

    for result in vector_results:
        enriched_result = dict(result)

        enriched_result["document_title"] = (
            document_titles.get(
                result["chunk_id"]
            )
        )

        enriched_results.append(
            enriched_result
        )

    prompt_context = prepare_prompt_context(
        chunks=enriched_results,
        facts=graph_context["facts"],
    )

    serialized_context = json.dumps(
        prompt_context,
        indent=2,
        ensure_ascii=False,
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        "Retrieved evidence:\n"
        f"{serialized_context}"
    )

    answer = generate_text(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        provider=provider,
    )

    return {
        "conversation_id": conversation_id,
        "question": question,
        "provider": provider,
        "answer": answer,
        "results": enriched_results,
        "entities": graph_context["entities"],
        "facts": graph_context["facts"],
    }
