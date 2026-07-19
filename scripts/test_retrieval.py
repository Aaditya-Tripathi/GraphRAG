from app.retrieval import retrieve_chunks


TEST_CONVERSATION_ID = "step9-demo"

QUESTION = (
    "How does the system prevent information from "
    "different conversations from being mixed?"
)


def main() -> None:
    print("Running vector retrieval...")
    print(f"Conversation: {TEST_CONVERSATION_ID}")
    print(f"Question: {QUESTION}")

    results = retrieve_chunks(
        conversation_id=TEST_CONVERSATION_ID,
        question=QUESTION,
        top_k=5,
    )

    print()
    print(f"Results returned: {len(results)}")

    if not results:
        raise RuntimeError(
            "No vector-search results were returned."
        )

    if len(results) > 5:
        raise RuntimeError(
            "More than five results were returned."
        )

    print()

    for index, result in enumerate(results, start=1):
        print("=" * 70)
        print(f"RESULT {index}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(
            f"Conversation ID: "
            f"{result['conversation_id']}"
        )
        print(f"Chunk index: {result['chunk_index']}")
        print("-" * 70)
        print(result["text"])
        print()

        if (
            result["conversation_id"]
            != TEST_CONVERSATION_ID
        ):
            raise RuntimeError(
                "A result from another conversation "
                "was returned."
            )

    scores = [
        result["score"]
        for result in results
    ]

    if scores != sorted(scores, reverse=True):
        raise RuntimeError(
            "Results are not ordered by similarity score."
        )

    missing_conversation_results = retrieve_chunks(
        conversation_id="conversation-that-does-not-exist",
        question=QUESTION,
        top_k=5,
    )

    if missing_conversation_results:
        raise RuntimeError(
            "Conversation filtering failed. Results were "
            "returned for a nonexistent conversation."
        )

    print("=" * 70)
    print("Conversation filter test passed.")
    print("All vector retrieval checks passed.")


if __name__ == "__main__":
    main()
