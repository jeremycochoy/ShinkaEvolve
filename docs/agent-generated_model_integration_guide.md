# Model Integration Guide

## Overview

ShinkaEvolve uses a modular architecture for integrating LLM providers. This guide explains how to add new model providers by examining the local models and OpenRouter integrations.

## Integration Pattern (4 Steps)

All model integrations follow this consistent pattern:

### 1. Client Setup (`shinka/llm/client.py`)

Add client detection and configuration in the `get_client_llm()` function:

```python
elif model_name.startswith("your-prefix-"):
    # Parse model name/URL from the model string
    # Create OpenAI-compatible client
    client = openai.OpenAI(
        api_key="your-api-key",
        base_url="your-base-url"
    )

    # Wrap with instructor for structured output if needed
    if structured_output:
        client = instructor.from_openai(
            client,
            mode=instructor.Mode.JSON,
        )
```

### 2. Query Function (`shinka/llm/models/your_provider.py`)

Create a query function with:
- Backoff retry logic for error handling
- Support for both regular and structured outputs
- Thinking token extraction (for reasoning models)
- Cost calculation using pricing dictionary
- Return a `QueryResult` object

```python
@backoff.on_exception(
    backoff.expo,
    (openai.APIConnectionError, openai.APIStatusError, ...),
    max_tries=20,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_your_provider(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    # Implementation here
    pass
```

### 3. Export (`shinka/llm/models/__init__.py`)

Import and export your query function:

```python
from .your_provider import query_your_provider

__all__ = [
    "query_anthropic",
    "query_openai",
    # ... other providers
    "query_your_provider",  # Add here
    "QueryResult",
]
```

### 4. Registration (`shinka/llm/query.py`)

Import and register the query function:

```python
from .models import (
    query_anthropic,
    query_openai,
    # ... other providers
    query_your_provider,  # Add import
    QueryResult,
)

# In the query() function, add routing logic:
def query(...):
    # ...
    elif original_model_name.startswith('your-prefix-'):
        query_fn = query_your_provider
    # ...
```

### 5. Pricing (Optional) (`shinka/llm/models/pricing.py`)

Add pricing information for your models:

```python
YOUR_PROVIDER_MODELS = {
    "model-name": {
        "input_price": 0.28 / M,
        "output_price": 0.42 / M,
    },
}
```

## Example: Local Models Integration

**Model name format**: `local-ModelName-http://localhost:8000`

**Key features**:
- Pattern: `local-(.+?)-(https?://.+)` extracts model name and URL
- Connects to local OpenAI-compatible API (e.g., unsloth, ollama)
- Uses filler API key
- Fixed pricing from `LOCAL_MODELS` dict

**Client setup** (client.py:83-103):
```python
elif model_name.startswith("local-"):
    match = re.match(r"local-(.+?)-(https?://.+)", model_name)
    if match:
        model_name = match.group(1)
        url = match.group(2)
    else:
        raise ValueError(f"Invalid local model format: {model_name}")

    client = openai.OpenAI(
        api_key="filler",
        base_url=url
    )
```

**Query function** (shinka/llm/models/local.py):
- Handles `<think>...</think>` tags for reasoning models
- Uses `LOCAL_MODELS["local"]` for pricing

## Example: OpenRouter Integration

**Model name format**: `openrouter-qwen/qwen3-32b`

**Key features**:
- Strips `openrouter-` prefix to get actual model name
- Fixed base_url: `https://openrouter.ai/api/v1`
- Requires `OPENROUTER_API_KEY` environment variable
- Per-model pricing from `OPENROUTER_MODELS` dict

**Client setup** (client.py:104-119):
```python
elif model_name.startswith("openrouter-"):
    model_name = model_name.replace("openrouter-", "")

    client = openai.OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1"
    )
```

**Query function** (shinka/llm/models/openrouter.py):
- Handles both regular models and reasoning models
- Extracts thinking from `<think>` tags
- Per-model pricing lookup with fallback

## Comparison: Local vs OpenRouter

| Aspect | Local | OpenRouter |
|--------|-------|------------|
| **URL** | Embedded in model name | Fixed API endpoint |
| **Auth** | Filler key | Requires API key from env |
| **Pricing** | Single fixed rate | Per-model pricing lookup |
| **Model Name** | Extracted from string pattern | Strip prefix only |
| **Use Case** | Self-hosted models | Third-party API aggregator |

## Common Features

Both implementations support:
- Backoff retry logic for API errors
- Structured output via instructor
- Thinking token extraction for reasoning models
- Message history management
- Cost tracking

## Adding Embedding Models

For embedding models, modify `shinka/llm/embedding.py`:

```python
elif model_name.startswith("your-prefix-"):
    # Parse model name and URL
    client = openai.OpenAI(
        base_url=your_url,
        api_key="your-key"
    )

# Add cost calculation:
elif self.model_name.startswith("your-prefix-"):
    cost = 0.0  # or calculate based on usage
```

## Testing

After integration, test with:

```yaml
# In your config file
llm_models:
  - your-prefix-model-name-url
```

Run integration tests to verify the new provider works correctly.
