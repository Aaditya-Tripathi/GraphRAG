from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from app.constants import (
    MAX_CONVERSATION_ID_LENGTH,
    MAX_DOCUMENT_CHARACTERS,
    MAX_DOCUMENT_TITLE_LENGTH,
    MAX_QUESTION_LENGTH,
    MAX_TOP_K,
)


class TextIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(
        min_length=1,
        max_length=MAX_CONVERSATION_ID_LENGTH,
    )

    document_title: str = Field(
        min_length=1,
        max_length=MAX_DOCUMENT_TITLE_LENGTH,
    )

    text: str = Field(
        min_length=1,
        max_length=MAX_DOCUMENT_CHARACTERS,
    )

    provider: Literal["groq", "openrouter"] = "groq"

    @field_validator(
        "conversation_id",
        "document_title",
        "text",
    )
    @classmethod
    def strip_required_text(
        cls,
        value: str,
    ) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "Value cannot be empty."
            )

        return cleaned


class KnowledgeGraphBuildSummary(BaseModel):
    chunks_processed: int
    entities_extracted: int
    relationships_extracted: int
    mentions_processed: int
    facts_processed: int


class TextIngestResponse(BaseModel):
    conversation_id: str
    document_id: str
    document_title: str
    chunks_created: int
    next_chunk_relationships: int
    embedding_dimension: int
    provider: Literal["groq", "openrouter"]
    knowledge_graph: KnowledgeGraphBuildSummary


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(
        min_length=1,
        max_length=MAX_CONVERSATION_ID_LENGTH,
    )

    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=MAX_TOP_K,
    )

    provider: Literal["groq", "openrouter"] = "groq"

    @field_validator(
        "conversation_id",
        "question",
    )
    @classmethod
    def strip_query_text(
        cls,
        value: str,
    ) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "Value cannot be empty."
            )

        return cleaned


class SupportingChunk(BaseModel):
    chunk_id: str
    conversation_id: str
    document_id: str
    document_title: str | None = None
    chunk_index: int
    text: str
    score: float


class GraphEntity(BaseModel):
    id: str | None = None
    name: str
    normalized_name: str | None = None
    type: str


class GraphFact(BaseModel):
    id: str | None = None
    source: str
    predicate: str
    target: str
    confidence: float
    source_chunk_id: str | None = None
    document_id: str | None = None
    evidence: str | None = None


class QueryResponse(BaseModel):
    conversation_id: str
    question: str
    provider: Literal["groq", "openrouter"]
    answer: str
    results: list[SupportingChunk]
    entities: list[GraphEntity]
    facts: list[GraphFact]


class HealthResponse(BaseModel):
    status: str
    neo4j: str
    embedding_model: str
    groq_model: str
    openrouter_model: str
