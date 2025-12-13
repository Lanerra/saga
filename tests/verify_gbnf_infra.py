import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

# Mock structlog before importing modules that use it
sys.modules["structlog"] = MagicMock()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.llm_interface_refactored import create_llm_service
from prompts.grammar_loader import load_grammar


async def test_grammar_loader() -> None:
    print("\n--- Testing Grammar Loader ---")
    try:
        # Test loading initialization grammar (which should include common)
        grammar_text = load_grammar("initialization")

        print("Successfully loaded 'initialization.gbnf'")
        print(f"Total length: {len(grammar_text)} chars")

        if "root ::= global_outline" in grammar_text:
            print("✅ Found root definition from initialization.gbnf")
        else:
            print("❌ Missing root definition from initialization.gbnf")

        if "json_value ::= json_object" in grammar_text:
            print("✅ Found common definitions from common.gbnf")
        else:
            print("❌ Missing common definitions from common.gbnf")

        # Check that common.gbnf's root was filtered out
        # common.gbnf has "root ::= json_object"
        # initialization.gbnf has "root ::= global_outline ..."

        roots = [line for line in grammar_text.splitlines() if line.strip().startswith("root ::=")]
        print(f"Found {len(roots)} root definitions: {roots}")

        if len(roots) == 1 and "global_outline" in roots[0]:
            print("✅ Only specific grammar root is present")
        else:
            print("❌ Unexpected root definitions found")

    except Exception as e:
        print(f"❌ Grammar loader failed: {e}")
        import traceback

        traceback.print_exc()


async def test_service_propagation() -> None:
    print("\n--- Testing Service Parameter Propagation ---")

    # Create service
    llm_service = create_llm_service()

    # Mock the http client to intercept the call
    # We need to access the nested http client inside the completion service
    completion_client = llm_service._completion_service._completion_client
    http_client = completion_client._http_client

    # Mock post_json
    mock_post_json = AsyncMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"total_tokens": 10},
    }
    mock_post_json.return_value = mock_response
    http_client.post_json = mock_post_json  # type: ignore

    test_grammar = 'root ::= "test"'

    # Call the top-level service
    await llm_service.async_call_llm(model_name="test-model", prompt="Test prompt", grammar=test_grammar)

    # Verify the call reached the http client with the grammar in the payload
    call_args = mock_post_json.call_args
    if call_args:
        url, payload, headers = call_args[0]
        print("Called URL:", url)

        if "grammar" in payload:
            print("✅ 'grammar' field found in payload")
            if payload["grammar"] == test_grammar:
                print("✅ 'grammar' value matches input")
            else:
                print(f"❌ 'grammar' value mismatch. Got: {payload['grammar']}")
        else:
            print("❌ 'grammar' field MISSING in payload")
            print("Payload keys:", payload.keys())
    else:
        print("❌ HTTP client was not called")

    await http_client.aclose()


async def main() -> None:
    await test_grammar_loader()
    await test_service_propagation()


if __name__ == "__main__":
    asyncio.run(main())
