import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)


def clean_text(text: str) -> str:
    """Clean unnecessary whitespace from input text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Convert Windows and old Mac line endings to normal newlines.
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace repeated spaces and tabs with one space.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Remove spaces immediately after a newline.
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)

    # Keep at most one empty line between paragraphs.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def split_text(text: str) -> list[str]:
    """Clean text and divide it into overlapping chunks."""

    cleaned = clean_text(text)

    if not cleaned:
        raise ValueError("The supplied text is empty.")

    chunks = _text_splitter.split_text(cleaned)

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]
