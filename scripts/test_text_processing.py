from pathlib import Path

from app.text_processing import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    clean_text,
    split_text,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILE = PROJECT_ROOT / "sample_data" / "sample.txt"


def main() -> None:
    raw_text = SAMPLE_FILE.read_text(encoding="utf-8-sig")
    cleaned_text = clean_text(raw_text)
    chunks = split_text(raw_text)

    print("Text processing successful")
    print(f"Raw characters: {len(raw_text)}")
    print(f"Cleaned characters: {len(cleaned_text)}")
    print(f"Chunk size setting: {CHUNK_SIZE}")
    print(f"Chunk overlap setting: {CHUNK_OVERLAP}")
    print(f"Number of chunks: {len(chunks)}")

    if not chunks:
        raise RuntimeError("No chunks were generated.")

    for index, chunk in enumerate(chunks, start=1):
        print()
        print("=" * 60)
        print(f"CHUNK {index}")
        print(f"Characters: {len(chunk)}")
        print("-" * 60)
        print(chunk)


if __name__ == "__main__":
    main()
