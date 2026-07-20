import re


CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def clean_text(text: str) -> str:
    """Clean unnecessary whitespace from input text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def split_text(text: str) -> list[str]:
    """Divide cleaned text into overlapping chunks."""

    cleaned = clean_text(text)

    if not cleaned:
        raise ValueError("The supplied text is empty.")

    chunks: list[str] = []
    start = 0

    while start < len(cleaned):
        end = min(start + CHUNK_SIZE, len(cleaned))

        if end < len(cleaned):
            search_start = start + (CHUNK_SIZE // 2)

            for separator in ("\n\n", "\n", ". ", " "):
                boundary = cleaned.rfind(
                    separator,
                    search_start,
                    end,
                )

                if boundary != -1:
                    end = boundary + len(separator)
                    break

        chunk = cleaned[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned):
            break

        start = max(
            end - CHUNK_OVERLAP,
            start + 1,
        )

    return chunks
