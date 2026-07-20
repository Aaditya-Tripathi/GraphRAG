from app.entity_extraction import KNOWLEDGE_EXTRACTION_SCHEMA
from app.knowledge_graph import prepare_entities
from app.schemas import EntityItem


def test_entity_types_are_open_in_schema_and_normalized_for_storage() -> None:
    type_schema = KNOWLEDGE_EXTRACTION_SCHEMA["properties"][
        "entities"
    ]["items"]["properties"]["type"]

    assert "enum" not in type_schema

    entities = prepare_entities(
        "type-normalization-test",
        [
            EntityItem(name="Project Atlas", type="Project"),
            EntityItem(name="Routing Platform", type="Software"),
            EntityItem(
                name="Warehouse Component",
                type="WarehouseComponent",
            ),
        ],
    )

    stored_types = {
        entity["name"]: entity["type"]
        for entity in entities
    }

    assert stored_types == {
        "Project Atlas": "Project",
        "Routing Platform": "Technology",
        "Warehouse Component": "Other",
    }
