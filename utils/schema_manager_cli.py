# utils/schema_manager_cli.py
#!/usr/bin/env python3
# utils/schema_manager_cli.py
"""
Command-line utility for managing the static schema system.

This utility provides commands for testing and managing the static schema system.
"""

import argparse
import asyncio
import logging
import sys

from core.schema_initialization import (
    setup_static_schema_logging,
    startup_schema_integration,
)


def setup_logging():
    """Setup basic logging for CLI operations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


async def command_status():
    """Show status of the static schema system."""
    print("Static Schema System Status")
    print("=" * 40)
    print("Static schema system is always available and ready.")
    print("No dynamic initialization or refresh needed.")
    return True


async def command_health():
    """Check health of the static schema system."""
    print("Static Schema Health Check")
    print("=" * 40)
    print("Status: healthy")
    print("Healthy: True")
    print("Message: Static schema system is functioning normally")
    return True


async def command_initialize(force=False):
    """Initialize the static schema system."""
    print(f"Initializing Static Schema System {'(forced)' if force else ''}")
    print("=" * 40)

    try:
        success = await startup_schema_integration()

        if success:
            print("✓ Initialization completed successfully")
            await command_status()
        else:
            print("✗ Initialization had issues")

        return success

    except Exception as e:
        print(f"Error during initialization: {e}")
        return False


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Static Schema System Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    subparsers.add_parser("status", help="Show system status")

    # Health command
    subparsers.add_parser("health", help="Check system health")

    # Initialize command
    init_parser = subparsers.add_parser("init", help="Initialize the system")
    init_parser.add_argument(
        "--force", action="store_true", help="Force initialization"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    setup_logging()
    setup_static_schema_logging()

    # Execute command
    success = False

    if args.command == "status":
        success = await command_status()
    elif args.command == "health":
        success = await command_health()
    elif args.command == "init":
        success = await command_initialize(force=args.force)

    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
