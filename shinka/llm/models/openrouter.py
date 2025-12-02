import backoff
import openai
import re
from .pricing import OPENROUTER_MODELS, REASONING_OPENROUTER_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"OpenRouter - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=20,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_openrouter(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query OpenRouter model."""
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    if output_model is None:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            **kwargs,
        )

        # Check if this is a reasoning model
        if model in REASONING_OPENROUTER_MODELS:
            # For reasoning models: output[0] = thinking, output[1] = answer
            try:
                thought = response.output[0].content[0].text
            except (IndexError, AttributeError):
                thought = ""

            try:
                raw_content = response.output[1].content[0].text
            except (IndexError, AttributeError):
                # Fallback to output[0] if output[1] doesn't exist
                raw_content = response.output[0].content[0].text
                thought = ""
        else:
            # For regular models: output[0] = answer
            raw_content = response.output[0].content[0].text
            thought = ""

        # Also extract thinking from <think>...</think> tags if present
        think_match = re.search(r"<think>(.*?)</think>", raw_content, re.DOTALL)

        if think_match:
            # If we found <think> tags, use that as thought (overriding any previous thought)
            thought = think_match.group(1).strip()
            # Remove the <think> tag from content
            content = (
                raw_content[: think_match.start()] + raw_content[think_match.end() :]
            ).strip()
        else:
            content = raw_content

        new_msg_history.append({"role": "assistant", "content": content})
    else:
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            text_format=output_model,
            **kwargs,
        )
        content = response.output_parsed
        new_content = ""
        for i in content:
            new_content += i[0] + ":" + i[1] + "\n"
        new_msg_history.append({"role": "assistant", "content": new_content})
        thought = ""

    # Get pricing for the specific model
    if model in OPENROUTER_MODELS:
        input_cost = OPENROUTER_MODELS[model]["input_price"] * response.usage.input_tokens
        output_cost = OPENROUTER_MODELS[model]["output_price"] * response.usage.output_tokens
    else:
        # Fallback to default pricing if model not found
        logger.warning(f"Model {model} not found in OPENROUTER_MODELS, using default pricing")
        input_cost = 0
        output_cost = 0

    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought=thought,
        model_posteriors=model_posteriors,
    )
    return result
