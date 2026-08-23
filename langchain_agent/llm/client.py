import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()


class LLMClientError(RuntimeError):
    """Raised when the LLM client cannot complete a request."""


_client = None


def _get_positive_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        value = float(raw_value)

    except ValueError as error:
        raise LLMClientError(f"{name} must be a number, got: {raw_value}") from error

    if value <= 0:
        raise LLMClientError(f"{name} must be greater than zero.")

    return value


def _get_non_negative_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as error:
        raise LLMClientError(f"{name} must be an integer, got: {raw_value}") from error

    if value < 0:
        raise LLMClientError(f"{name} must be zero or greater.")

    return value


def get_client():
    global _client

    if _client is not None:
        return _client

    try:
        from openai import OpenAI
    except ImportError as error:
        raise LLMClientError("LLM dependencies are missing; install openai") from error

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")

    if not api_key:
        raise LLMClientError("DEEPSEEK_API_KEY is not configured")
    if not base_url:
        raise LLMClientError("DEEPSEEK_BASE_URL is not configured")

    timeout_seconds = _get_positive_float_env("LLM_TIMEOUT_SECONDS", 120.0)

    max_retries = _get_non_negative_int_env("LLM_MAX_RETRIES", 2)

    _client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )

    return _client


def chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
    model = os.getenv("MODEL_NAME")
    if not model:
        raise LLMClientError("MODEL_NAME is not configured")

    client = get_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )

    except Exception as error:
        error_type = type(error).__name__
        request_id = getattr(error, "request_id", None)

        details = f"{error_type}: {error}"

        if request_id:
            details += f"request_id = {request_id}"

        raise LLMClientError(f"LLM request failed: {details}") from error

    if not response.choices:
        raise LLMClientError("LLM returned an empty choices list.")

    message = response.choices[0].message

    if message is None:
        raise LLMClientError("LLM returned no assistant message.")

    return message
