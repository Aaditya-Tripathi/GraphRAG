from app.text_processing import (
    CHUNK_OVERLAP,
    clean_text,
    split_text,
)


def test_clean_text_normalizes_whitespace() -> None:
    raw_text = "  Neo4j    stores graphs.\r\n\r\n\r\nGraphRAG works.  "

    assert clean_text(raw_text) == (
        "Neo4j stores graphs.\n\nGraphRAG works."
    )


def test_split_text_creates_overlapping_chunks() -> None:
    text = " ".join(
        f"token-{index:04d}"
        for index in range(500)
    )
    chunks = split_text(text)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 800 for chunk in chunks)

    for first, second in zip(chunks, chunks[1:]):
        shared_length = max(
            (
                length
                for length in range(
                    1,
                    min(len(first), len(second)) + 1,
                )
                if first.endswith(second[:length])
            ),
            default=0,
        )
        assert shared_length >= CHUNK_OVERLAP - 1
