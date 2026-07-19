import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


def require_environment_variable(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Missing environment variable: {name}. "
            f"Check the .env file at {ENV_FILE}"
        )

    return value


NEO4J_URI = require_environment_variable("NEO4J_URI")
NEO4J_USERNAME = require_environment_variable("NEO4J_USERNAME")
NEO4J_PASSWORD = require_environment_variable("NEO4J_PASSWORD")
NEO4J_DATABASE = require_environment_variable("NEO4J_DATABASE")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
VECTOR_INDEX_NAME = "chunk_embedding_index"