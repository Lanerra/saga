#!/usr/bin/env python3
# utils/schema_manager_cli.py
"""
Command-line utility for managing the dynamic schema system.

This utility provides commands for testing, monitoring, and managing
the dynamic schema system.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime

import config
from core.dynamic_schema_manager import dynamic_schema_manager
from core.schema_initialization import (
    get_schema_health_check,
    initialize_dynamic_schema_system,
    refresh_dynamic_schema_system,
    setup_dynamic_schema_logging,
)


def setup_logging():
    """Setup basic logging for CLI operations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


async def command_status():
    """Show status of the dynamic schema system."""
    print("Dynamic Schema System Status")
    print("=" * 40)

    try:
        status = await dynamic_schema_manager.get_system_status()

        # System info
        system = status.get("system", {})
        print(f"Initialized: {system.get('initialized', False)}")
        print(f"Learning Enabled: {system.get('learning_enabled', False)}")
        print(f"Auto Refresh: {system.get('auto_refresh_enabled', False)}")
        print(f"Fallback Enabled: {system.get('fallback_enabled', False)}")
        print(f"Last Update: {system.get('last_update', 'Never')}")
        print()

        # Type inference info
        type_info = status.get("type_inference", {})
        if type_info:
            print("Type Inference:")
            print(f"  Total Patterns: {type_info.get('total_patterns', 0)}")
            pattern_types = type_info.get("pattern_types", {})
            if pattern_types:
                for ptype, count in pattern_types.items():
                    print(f"  {ptype}: {count}")
            print()

        # Constraints info
        constraint_info = status.get("constraints", {})
        if constraint_info:
            print("Relationship Constraints:")
            print(f"  Total Constraints: {constraint_info.get('total_constraints', 0)}")
            print(f"  Total Samples: {constraint_info.get('total_samples', 0)}")
            print()

        # Schema info
        schema_info = status.get("schema", {})
        if schema_info and not schema_info.get("error"):
            print("Schema Discovery:")
            print(f"  Total Labels: {schema_info.get('total_labels', 0)}")
            print(
                f"  Total Relationship Types: {schema_info.get('total_relationship_types', 0)}"
            )
            print(f"  Total Nodes: {schema_info.get('total_nodes', 0)}")
            print()

            most_common = schema_info.get("most_common_labels", [])
            if most_common:
                print("Most Common Labels:")
                for label, count in most_common[:5]:
                    print(f"  {label}: {count}")
                print()

    except Exception as e:
        print(f"Error getting status: {e}")
        return False

    return True


async def command_health():
    """Check health of the dynamic schema system."""
    print("Dynamic Schema Health Check")
    print("=" * 40)

    try:
        health = await get_schema_health_check()

        print(f"Status: {health.get('status', 'unknown')}")
        print(f"Healthy: {health.get('healthy', False)}")
        print(f"Message: {health.get('message', 'No message')}")

        details = health.get("details", {})
        if details and isinstance(details, dict):
            print("\nDetails:")
            for key, value in details.items():
                print(f"  {key}: {value}")

        return health.get("healthy", False)

    except Exception as e:
        print(f"Error checking health: {e}")
        return False


async def command_initialize(force=False):
    """Initialize the dynamic schema system."""
    print(f"Initializing Dynamic Schema System {'(forced)' if force else ''}")
    print("=" * 40)

    try:
        success = await initialize_dynamic_schema_system(
            force_refresh=force,
            timeout_seconds=60,  # Longer timeout for CLI
        )

        if success:
            print("✓ Initialization completed successfully")
            await command_status()
        else:
            print("✗ Initialization failed or timed out")

        return success

    except Exception as e:
        print(f"Error during initialization: {e}")
        return False


async def command_refresh():
    """Refresh the dynamic schema system."""
    print("Refreshing Dynamic Schema System")
    print("=" * 40)

    try:
        success = await refresh_dynamic_schema_system()

        if success:
            print("✓ Refresh completed successfully")
        else:
            print("✗ Refresh failed")

        return success

    except Exception as e:
        print(f"Error during refresh: {e}")
        return False


async def command_test_inference():
    """Test type inference with sample data."""
    print("Testing Type Inference")
    print("=" * 40)

    test_cases = [
        ("King Arthur", "character", ""),
        ("Excalibur", "weapon", "A legendary sword"),
        ("Camelot", "location", "The legendary castle"),
        ("Magic System", "system", "The way magic works"),
        ("Dragon", "", "A fearsome creature"),
        ("Unknown Entity", "", ""),
    ]

    try:
        await dynamic_schema_manager.ensure_initialized()

        print("Testing type inference on sample entities:")
        print()

        for name, category, description in test_cases:
            try:
                inferred_type = await dynamic_schema_manager.infer_node_type(
                    name, category, description
                )
                print(f"'{name}' (cat: {category or 'none'}) -> {inferred_type}")
            except Exception as e:
                print(f"'{name}' -> ERROR: {e}")

        return True

    except Exception as e:
        print(f"Error testing inference: {e}")
        return False


async def command_export_status(filename=None):
    """Export status to JSON file."""
    if not filename:
        filename = f"schema_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print(f"Exporting status to {filename}")
    print("=" * 40)

    try:
        status = await dynamic_schema_manager.get_system_status()
        health = await get_schema_health_check()

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "enabled": config.settings.ENABLE_DYNAMIC_SCHEMA,
                "learning_enabled": config.settings.DYNAMIC_SCHEMA_LEARNING_ENABLED,
                "auto_refresh": config.settings.DYNAMIC_SCHEMA_AUTO_REFRESH,
            },
            "status": status,
            "health": health,
        }

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"✓ Status exported to {filename}")
        return True

    except Exception as e:
        print(f"Error exporting status: {e}")
        return False


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Dynamic Schema System Manager",
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
        "--force", action="store_true", help="Force refresh of cached data"
    )

    # Refresh command
    subparsers.add_parser("refresh", help="Refresh system data")

    # Test command
    subparsers.add_parser("test", help="Test type inference")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export status to JSON")
    export_parser.add_argument("--output", help="Output filename")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    setup_logging()
    setup_dynamic_schema_logging()

    # Execute command
    success = False

    if args.command == "status":
        success = await command_status()
    elif args.command == "health":
        success = await command_health()
    elif args.command == "init":
        success = await command_initialize(force=args.force)
    elif args.command == "refresh":
        success = await command_refresh()
    elif args.command == "test":
        success = await command_test_inference()
    elif args.command == "export":
        success = await command_export_status(args.output)

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
