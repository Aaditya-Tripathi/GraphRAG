from app.config import NEO4J_DATABASE
from app.database import create_driver


def main() -> None:
    with create_driver() as driver:
        constraints, _, _ = driver.execute_query(
            """
            SHOW CONSTRAINTS
            YIELD name, type, labelsOrTypes, properties
            RETURN name, type, labelsOrTypes, properties
            ORDER BY name
            """,
            database_=NEO4J_DATABASE,
        )

        indexes, _, _ = driver.execute_query(
            """
            SHOW INDEXES
            YIELD name, type, state, labelsOrTypes, properties
            RETURN name, type, state, labelsOrTypes, properties
            ORDER BY name
            """,
            database_=NEO4J_DATABASE,
        )

        print("CONSTRAINTS")
        print("-" * 60)

        for record in constraints:
            print(
                f"{record['name']} | "
                f"{record['type']} | "
                f"{record['labelsOrTypes']} | "
                f"{record['properties']}"
            )

        print("\nINDEXES")
        print("-" * 60)

        for record in indexes:
            print(
                f"{record['name']} | "
                f"{record['type']} | "
                f"{record['state']} | "
                f"{record['labelsOrTypes']} | "
                f"{record['properties']}"
            )


if __name__ == "__main__":
    main()
