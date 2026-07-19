from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

from app.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
)


_embedder: SentenceTransformerEmbeddings | None = None


def get_embedder() -> SentenceTransformerEmbeddings:
    """Create the embedding model once and reuse it."""

    global _embedder

    if _embedder is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")

        _embedder = SentenceTransformerEmbeddings(
            model=EMBEDDING_MODEL,
        )

    return _embedder


def embed_text(text: str) -> list[float]:
    """Convert one piece of text into an embedding vector."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError("Cannot create an embedding for empty text.")

    vector = get_embedder().embed_query(cleaned_text)
    vector = [float(value) for value in vector]

    if len(vector) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            "Unexpected embedding dimension. "
            f"Expected {EMBEDDING_DIMENSION}, received {len(vector)}."
        )

    return vector


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert multiple text chunks into embedding vectors."""

    if not texts:
        raise ValueError("The text list cannot be empty.")

    return [
        embed_text(text)
        for text in texts
    ]
