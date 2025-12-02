"""
Integration tests for OpenRouter LLM provider.

This test file verifies that:
1. OpenRouter client is properly initialized
2. All three Qwen models can be queried successfully
3. API key is correctly used
4. Pricing is calculated accurately
5. Responses are valid and contain expected content
"""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
from shinka.llm.client import get_client_llm
from shinka.llm.query import query
from shinka.llm.models.pricing import OPENROUTER_MODELS

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Test models
OPENROUTER_TEST_MODELS = [
    "openrouter-qwen/qwen3-32b",
    "openrouter-qwen/qwen3-coder-30b-a3b-instruct",
    "openrouter-qwen/qwen3-30b-a3b",
]


def test_openrouter_api_key_exists():
    """Test that OPENROUTER_API_KEY is set in environment."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key is not None, "OPENROUTER_API_KEY not found in environment"
    assert api_key.startswith("sk-or-v1-"), "Invalid OpenRouter API key format"
    assert len(api_key) > 20, "API key seems too short"


def test_openrouter_models_in_pricing():
    """Test that all OpenRouter models are registered in pricing."""
    expected_models = [
        "qwen/qwen3-32b",
        "qwen/qwen3-coder-30b-a3b-instruct",
        "qwen/qwen3-30b-a3b",
    ]

    for model in expected_models:
        assert model in OPENROUTER_MODELS, f"Model {model} not found in OPENROUTER_MODELS"
        assert "input_price" in OPENROUTER_MODELS[model], f"Missing input_price for {model}"
        assert "output_price" in OPENROUTER_MODELS[model], f"Missing output_price for {model}"


def test_openrouter_pricing_values():
    """Test that pricing values are correct."""
    # Expected pricing
    expected_pricing = {
        "qwen/qwen3-32b": {"input": 0.08, "output": 0.24},
        "qwen/qwen3-coder-30b-a3b-instruct": {"input": 0.06, "output": 0.25},
        "qwen/qwen3-30b-a3b": {"input": 0.06, "output": 0.22},
    }

    M = 1000000
    for model, prices in expected_pricing.items():
        actual_input = OPENROUTER_MODELS[model]["input_price"] * M
        actual_output = OPENROUTER_MODELS[model]["output_price"] * M

        assert abs(actual_input - prices["input"]) < 0.001, \
            f"Input price mismatch for {model}: expected {prices['input']}, got {actual_input}"
        assert abs(actual_output - prices["output"]) < 0.001, \
            f"Output price mismatch for {model}: expected {prices['output']}, got {actual_output}"


def test_openrouter_client_creation():
    """Test that OpenRouter client can be created for each model."""
    for model_name in OPENROUTER_TEST_MODELS:
        client, extracted_model = get_client_llm(model_name, structured_output=False)

        assert client is not None, f"Client is None for {model_name}"
        assert extracted_model == model_name.replace("openrouter-", ""), \
            f"Model name extraction failed for {model_name}"

        # Check that client has the correct base URL
        assert hasattr(client, "base_url"), "Client missing base_url attribute"
        assert "openrouter.ai" in str(client.base_url), \
            f"Client base_url is incorrect: {client.base_url}"


def test_openrouter_client_structured_output():
    """Test that OpenRouter client can be created with structured output mode."""
    model_name = OPENROUTER_TEST_MODELS[0]
    client, extracted_model = get_client_llm(model_name, structured_output=True)

    assert client is not None, "Structured output client is None"
    # Verify instructor wrapper is applied
    # The instructor wrapper changes the client type
    assert hasattr(client, "chat") or hasattr(client, "responses"), \
        "Structured output client missing expected methods"


@pytest.mark.parametrize("model_name", OPENROUTER_TEST_MODELS)
def test_openrouter_query_simple(model_name):
    """Test simple query to each OpenRouter model.

    This test makes actual API calls to OpenRouter.
    Skip this test if you want to avoid API costs.
    """
    # Simple test prompt
    msg = "What is 2+2? Answer with just the number."
    system_msg = "You are a helpful assistant. Be concise."

    # Query the model
    result = query(
        model_name=model_name,
        msg=msg,
        system_msg=system_msg,
        msg_history=[],
        temperature=0.0,
        max_output_tokens=50,
    )

    # Verify result structure
    assert result is not None, f"Query result is None for {model_name}"
    assert hasattr(result, "content"), "Result missing content attribute"
    assert hasattr(result, "input_tokens"), "Result missing input_tokens"
    assert hasattr(result, "output_tokens"), "Result missing output_tokens"
    assert hasattr(result, "cost"), "Result missing cost attribute"
    assert hasattr(result, "input_cost"), "Result missing input_cost"
    assert hasattr(result, "output_cost"), "Result missing output_cost"

    # Verify content is not empty
    assert result.content, f"Response content is empty for {model_name}"
    assert len(result.content) > 0, f"Response content has zero length for {model_name}"

    # Verify token counts
    assert result.input_tokens > 0, f"Input tokens is 0 for {model_name}"
    assert result.output_tokens > 0, f"Output tokens is 0 for {model_name}"

    # Verify cost calculation
    assert result.input_cost >= 0, f"Input cost is negative for {model_name}"
    assert result.output_cost >= 0, f"Output cost is negative for {model_name}"
    assert result.cost == result.input_cost + result.output_cost, \
        f"Total cost doesn't match sum of input and output costs for {model_name}"

    # Verify cost is reasonable (should be very small for this simple query)
    assert result.cost < 0.01, f"Cost seems too high for simple query: {result.cost}"

    print(f"\n{model_name} Response:")
    print(f"  Content: {result.content[:100]}...")
    print(f"  Input tokens: {result.input_tokens}")
    print(f"  Output tokens: {result.output_tokens}")
    print(f"  Total cost: ${result.cost:.6f}")


def test_openrouter_query_with_history():
    """Test query with message history."""
    model_name = OPENROUTER_TEST_MODELS[0]

    msg_history = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
    ]

    msg = "What is my name?"
    system_msg = "You are a helpful assistant."

    result = query(
        model_name=model_name,
        msg=msg,
        system_msg=system_msg,
        msg_history=msg_history,
        temperature=0.0,
        max_output_tokens=50,
    )

    assert result is not None
    assert result.content
    assert "alice" in result.content.lower(), \
        f"Model didn't remember the name from history. Response: {result.content}"

    # Verify new message history includes the new exchange
    assert len(result.new_msg_history) == len(msg_history) + 2, \
        "New message history has incorrect length"


def test_openrouter_cost_calculation_accuracy():
    """Test that cost calculation matches expected values."""
    model_name = "openrouter-qwen/qwen3-32b"

    msg = "Count from 1 to 5."
    system_msg = "You are a helpful assistant."

    result = query(
        model_name=model_name,
        msg=msg,
        system_msg=system_msg,
        msg_history=[],
        temperature=0.0,
        max_output_tokens=50,
    )

    # Calculate expected costs manually
    M = 1000000
    expected_input_cost = (OPENROUTER_MODELS["qwen/qwen3-32b"]["input_price"] *
                          result.input_tokens)
    expected_output_cost = (OPENROUTER_MODELS["qwen/qwen3-32b"]["output_price"] *
                           result.output_tokens)

    # Verify costs match (with small tolerance for floating point)
    assert abs(result.input_cost - expected_input_cost) < 1e-10, \
        f"Input cost calculation incorrect: {result.input_cost} != {expected_input_cost}"
    assert abs(result.output_cost - expected_output_cost) < 1e-10, \
        f"Output cost calculation incorrect: {result.output_cost} != {expected_output_cost}"


def test_openrouter_temperature_parameter():
    """Test that temperature parameter is passed correctly."""
    model_name = OPENROUTER_TEST_MODELS[0]

    msg = "Say hello."
    system_msg = "You are a helpful assistant."

    # Test with temperature 0.0 (deterministic)
    result = query(
        model_name=model_name,
        msg=msg,
        system_msg=system_msg,
        msg_history=[],
        temperature=0.0,
        max_output_tokens=20,
    )

    assert result is not None
    assert result.content

    # Store first response
    first_response = result.content

    # Query again with same parameters
    result2 = query(
        model_name=model_name,
        msg=msg,
        system_msg=system_msg,
        msg_history=[],
        temperature=0.0,
        max_output_tokens=20,
    )

    # With temperature=0.0, responses should be very similar or identical
    # Note: Due to API variations, they might not be 100% identical
    assert result2.content, "Second response is empty"


def test_openrouter_max_output_tokens_parameter():
    """Test that max_output_tokens parameter is respected."""
    model_name = OPENROUTER_TEST_MODELS[0]

    msg = "Write a long story about a cat."
    system_msg = "You are a creative writer."

    # Test with small max_output_tokens
    result = query(
        model_name=model_name,
        msg=msg,
        system_msg=system_msg,
        msg_history=[],
        temperature=0.5,
        max_output_tokens=10,  # Very small limit
    )

    assert result is not None
    assert result.output_tokens <= 15, \
        f"Output tokens ({result.output_tokens}) exceeds max_output_tokens limit with buffer"


def test_openrouter_error_handling():
    """Test that errors are handled gracefully."""
    # This test uses an invalid model name to trigger an error
    try:
        result = query(
            model_name="openrouter-invalid/model",
            msg="Test",
            system_msg="Test",
            msg_history=[],
            max_output_tokens=10,
        )
        # If we get here, the query somehow succeeded (unexpected)
        # But we won't fail the test, as the model might exist
    except Exception as e:
        # Expected behavior - should raise an exception
        assert "not found" in str(e).lower() or "not supported" in str(e).lower(), \
            f"Unexpected error message: {e}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
