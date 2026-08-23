import json
from typing import Any, TypeAlias

ChatMessage: TypeAlias = dict[str, Any]


def system_message(
    content: str,
) -> ChatMessage:
    return {
        "role": "system",
        "content": content,
    }


def user_message(
    content: str,
) -> ChatMessage:
    return {
        "role": "user",
        "content": content,
    }


def assistant_message(
    content: str,
) -> ChatMessage:
    return {
        "role": "assistant",
        "content": content,
    }


def assistant_tool_call_message(
    *,
    content: str | None,
    tool_calls: list[dict[str, Any]],
) -> ChatMessage:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }


def tool_result_message(
    *,
    tool_call_id: str,
    tool_name: str,
    result: dict[str, Any],
) -> ChatMessage:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": json.dumps(
            result,
            ensure_ascii=False,
        ),
    }


def serialize_tool_call(
    tool_call: Any,
) -> dict[str, Any]:
    if hasattr(tool_call, "model_dump"):
        result = tool_call.model_dump(exclude_none=True)

        if not isinstance(result, dict):
            raise TypeError("Serialized tool call must be a dictionary.")

        return result

    if isinstance(tool_call, dict):
        return dict(tool_call)

    raise TypeError("Tool call must be a dictionary or support model_dump().")
