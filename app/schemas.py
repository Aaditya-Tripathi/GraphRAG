from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class EntityItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=200)
    type: str = Field(
        default="Other",
        min_length=1,
        max_length=50,
        validation_alias=AliasChoices(
            "type",
            "category",
            "label",
        ),
    )

    @field_validator("name", "type")
    @classmethod
    def clean_text(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())

        if not cleaned:
            raise ValueError("Value cannot be empty.")

        return cleaned


class RelationshipItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "source",
            "from",
        ),
    )
    predicate: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "predicate",
            "relation",
            "relationship",
            "type",
        ),
    )
    target: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "target",
            "to",
        ),
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    @field_validator("source", "target")
    @classmethod
    def clean_entity_name(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())

        if not cleaned:
            raise ValueError("Entity name cannot be empty.")

        return cleaned

    @field_validator("predicate")
    @classmethod
    def normalize_predicate(cls, value: str) -> str:
        cleaned = "_".join(
            value.strip().upper().split()
        )

        if not cleaned:
            raise ValueError("Predicate cannot be empty.")

        if not all(
            character.isalnum() or character == "_"
            for character in cleaned
        ):
            raise ValueError(
                "Predicate must contain only letters, "
                "numbers, and underscores."
            )

        return cleaned


class KnowledgeExtraction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entities: list[EntityItem] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "entities",
            "nodes",
        ),
    )
    relationships: list[RelationshipItem] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "relationships",
            "edges",
            "relations",
        ),
    )
