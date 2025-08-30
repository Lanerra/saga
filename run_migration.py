#!/usr/bin/env python3

import sys
import os
import asyncio
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def main():
    try:
        # Import the migration module
        import importlib.util
        
        # Load the migration module
        migration_path = os.path.join(current_dir, 'migrations', '001_remove_dynamic_rel.py')
        spec = importlib.util.spec_from_file_location("migration_001", migration_path)
        migration_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration_module)
        
        logger.info("Starting migration to remove DYNAMIC_REL relationships...")
        
        # Run the migration
        asyncio.run(migration_module.main())
        
        logger.info("Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running migration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()