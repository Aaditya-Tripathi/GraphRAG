import math
from pathlib import Path

from app.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL
from app.embeddings import embed_text, embed_texts
from app.text_processing import split_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILE = PROJECT_ROOT / "sample_data" / "sample.txt"


def cosine_similarity(
    first: list[float],
    second: list[float],
) -> float:
    dot_product = sum(
        first_value * second_value
        for first_value, second_value in zip(first, second)
    )

    first_length = math.sqrt(
        sum(value * value for value in first)
    )

    second_length = math.sqrt(
        sum(value * value for value in second)
    )

    if first_length == 0 or second_length == 0:
        raise ValueError("Cannot compare an empty vector.")

    return dot_product / (first_length * second_length)


def main() -> None:
    sample_text = SAMPLE_FILE.read_text(
        encoding="utf-8-sig",
    )

    chunks = split_text(sample_text)
    embeddings = embed_texts(chunks)

    print()
    print("Embedding generation successful")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(
        "First five values:",
        [round(value, 6) for value in embeddings[0][:5]],
    )

    assert len(embeddings) == len(chunks)

    assert all(
        len(vector) == EMBEDDING_DIMENSION
        for vector in embeddings
    )

    first = embed_text(
        "Neo4j stores information using nodes and relationships."
    )

    similar = embed_text(
        "A graph database represents data as connected nodes."
    )

    unrelated = embed_text(
        "A chef prepares vegetables in a restaurant kitchen."
    )

    similar_score = cosine_similarity(first, similar)
    unrelated_score = cosine_similarity(first, unrelated)

    print()
    print("Similarity test")
    print(
        f"Graph-related sentence: {similar_score:.4f}"
    )
    print(
        f"Unrelated sentence: {unrelated_score:.4f}"
    )

    if similar_score <= unrelated_score:
        raise RuntimeError(
            "The semantic similarity test produced an "
            "unexpected result."
        )

    print()
    print("All embedding checks passed.")


if __name__ == "__main__":
    main()
