import os
from pathlib import Path


os.environ["GRAPHRAG_ENV_FILE"] = str(
    Path(__file__).with_name(".env.test")
)
os.environ.setdefault(
    "NEO4J_URI",
    "neo4j://localhost:7687",
)
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("NEO4J_DATABASE", "neo4j")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("GROQ_MODEL", "openai/gpt-oss-20b")
