from app.database import verify_connection


def main() -> None:
    try:
        result = verify_connection()

        print("Connection successful")
        print(f"Message: {result['message']}")
        print(f"Test calculation: {result['result']}")

    except Exception as error:
        print("Connection failed")
        print(f"Error type: {type(error).__name__}")
        print(f"Details: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
