import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = Path(
    os.getenv(
        "GRAPHRAG_ENV_FILE",
        PROJECT_ROOT / ".env",
    )
)

load_dotenv(ENV_FILE, override=True)


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

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-oss-20b:free",
)


def get_groq_api_key() -> str:
    """Load the Groq key only when an LLM request is made."""

    return require_environment_variable("GROQ_API_KEY")


def get_openrouter_api_key() -> str:
    """Load the OpenRouter key only when it is selected."""

    return require_environment_variable("OPENROUTER_API_KEY")
