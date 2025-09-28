#!/usr/bin/env python3
"""
Test script to verify the simplified structlog output format.
"""
import sys
import os

# Add the project root to the path so we can import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from config.settings import logger
from orchestration import nana_orchestrator

def test_logging():
    """Test the new simplified logging format."""
    # Set up the logging system as the orchestrator does
    nana_orchestrator.setup_logging_nana()
    
    print("Testing simplified structlog output format...")
    print("=" * 50)
    
    # Test basic logging
    logger.info("Basic info message")
    logger.warning("Basic warning message")
    logger.error("Basic error message")
    
    # Test logging with context
    logger.info("Message with context", user_id=123, action="test", status="success")
    
    # Test logging with multiple context values
    logger.debug("Debug with multiple values", 
                chapter=1, 
                agent="narrative", 
                tokens_used=42, 
                processing_time=1.23)
    
    # Test error logging
    try:
        raise ValueError("Test error for logging")
    except Exception:
        logger.exception("Exception occurred during test")
    
    print("=" * 50)
    print("Logging test completed!")

if __name__ == "__main__":
    test_logging()
