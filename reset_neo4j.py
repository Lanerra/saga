"""Fail-closed compatibility entrypoint for the retired database-wide reset."""

import argparse
from typing import NoReturn

RESET_REFUSAL = (
    "Database reset is disabled: this tool cannot prove project ownership or coordinate "
    "Neo4j data with project files and checkpoints. No data was changed. "
    "Resume the explicitly selected project; see README.md for recovery boundaries."
)


async def reset_neo4j_database_async(uri: str | None, user: str | None, password: str | None, confirm: bool = False) -> NoReturn:
    """Reject legacy callers, including confirmation bypasses, without connecting."""
    raise RuntimeError(RESET_REFUSAL)


def main() -> NoReturn:
    parser = argparse.ArgumentParser(description=RESET_REFUSAL)
    parser.parse_args()
    parser.exit(2, RESET_REFUSAL + "\n")


if __name__ == "__main__":
    main()
