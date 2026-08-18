# -*- coding: utf-8 -*-
"""A lightweight async agent wrapper around the OpenAI Python SDK.

This module provides:
- An `Agent` class that can run chat requests and handle tool calling.
- Pydantic-based tool schema generation for OpenAI "tools".

Note: This implementation supports only a single `base_url`.
"""

from __future__ import annotations

import inspect
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, get_type_hints

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, create_model

# Backend-specific compatibility handlers are imported as needed.
from deepseek_v32 import is_deepseek_v32, deepseek_v32_chat_completions_create


@dataclass
class AgentConfig:
    """Configuration for the `Agent`.

    Attributes:
        model: Model name. If not provided, `OPENAI_MODEL` is used.
        api_key: API key. If not provided, `OPENAI_API_KEY` is used.
        base_url: Base URL of the OpenAI-compatible endpoint. If not provided,
            `OPENAI_BASE_URL` is used. Only a single URL is supported.
        system_prompt: Default system prompt inserted before user prompts.
    """

    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    system_prompt: Optional[str] = "You are a helpful assistant."


@dataclass
class ChatConfig:
    """Per-call chat configuration.

    Attributes:
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        max_tokens: Maximum tokens to generate.
        enable_thinking: Whether to enable "thinking" mode (passed via `extra_body`).
    """

    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 4096
    enable_thinking: bool = False


class BadToolCalling(Exception):
    """Raised when tools are provided but the model does not return tool calls."""


class Agent:
    """Async agent that wraps OpenAI-compatible chat/completions endpoints.

    This class supports standard OpenAI Chat Completions API via
    `client.chat.completions.create(...)` with transparent compatibility
    for specialized model backends.

    The agent can also execute tool calls returned by the model by mapping them
    to registered Python callables.
    """

    def __init__(self, config: AgentConfig):
        """Initialize the agent.

        Args:
            config: Agent configuration.
        """
        self._config = config
        self._client: AsyncOpenAI = self._setup_client()

    def _setup_client(self) -> AsyncOpenAI:
        """Create and return an `AsyncOpenAI` client.

        This version supports only a single `base_url`. If a comma is found in
        `base_url`, a `ValueError` is raised.

        Returns:
            An initialized `AsyncOpenAI` client.

        Raises:
            ValueError: If neither `api_key` nor `OPENAI_API_KEY` is provided.
            ValueError: If `base_url` contains multiple URLs separated by commas.
        """
        api_key = self._config.api_key or os.environ.get("OPENAI_API_KEY")
        base_url = self._config.base_url or os.environ.get("OPENAI_BASE_URL")

        if not api_key or not str(api_key).strip():
            raise ValueError("Missing API key. Set AgentConfig.api_key or OPENAI_API_KEY.")

        client_kwargs = {"api_key": api_key, "base_url": base_url}

        return AsyncOpenAI(**client_kwargs)

    async def chat(
            self,
            prompt: str | List,
            *,
            tools: Optional[Iterable[Callable]] = None,
            config: Optional[ChatConfig] = None,
            system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run a single-turn chat call and return the updated message list.

        Args:
            prompt: User prompt text.
            tools: Optional iterable of tool functions to expose to the model.
            config: Optional per-call chat configuration.
            system_prompt: Optional system prompt override.

        Returns:
            The full message list including the assistant reply (and any tool
            messages if tool calling occurred).
        """
        cfg = config or ChatConfig()

        request_kwargs: Dict[str, Any] = {
            "top_p": cfg.top_p,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "extra_body": (
                {"thinking": {"type": "disabled"}}
                if os.getenv("OPENAI_DISABLE_THINKING", "true").lower() == "true"
                else {"enable_thinking": cfg.enable_thinking}
            ),
        }

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        system_prompt = system_prompt or self._config.system_prompt
        if messages[0]["role"] != "system" and (system_prompt):
            messages = [{"role": "system", "content": system_prompt}] + messages

        return await self.create(messages, tools, **request_kwargs)

    async def create(
            self,
            messages: List[Dict[str, Any]],
            tools: Optional[Iterable[Callable]],
            *args: Any,
            **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run a chat turn, optionally execute tool calls, and return messages.

        Args:
            messages: Current chat messages in OpenAI format.
            tools: Optional tool callables.
            *args: Extra positional args forwarded to the SDK create method.
            **kwargs: Extra keyword args forwarded to the SDK create method.

        Returns:
            Updated messages list.

        Raises:
            BadToolCalling: If tools are provided but the model returns no tool calls
                (standard chat branch only, preserved from the original behavior).
            DeepseekV32DecodingError: If response parsing fails.
            Exception: Any SDK/network exception will bubble up.
        """
        model = self._config.model or os.environ.get("OPENAI_MODEL")
        extra_body = kwargs.pop("extra_body", {"enable_thinking": False})
        oai_tools = to_oai_tools(list(tools or []))

        if is_deepseek_v32(model):
            # Backend-specific compatibility: transparently handle model variations.
            message, response = await deepseek_v32_chat_completions_create(
                client=self._client,
                model=model,
                messages=messages,
                oai_tools=oai_tools,
                extra_body=extra_body,
                args=args,
                kwargs=kwargs,
            )
        else:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                tools=oai_tools,
                tool_choice="auto" if oai_tools else "none",
                extra_body=extra_body,
                *args,
                **kwargs,
            )

            message = {
                "role": "assistant",
                "content": response.choices[0].message.content,
                "tool_calls": _normalize_tool_calls(response.choices[0].message.tool_calls),
            }

            # Preserve original behavior: only standard chat branch retries on missing tool calls.
            if tools and ("tool_calls" not in message or len(message["tool_calls"]) == 0):
                raise BadToolCalling(response.choices[0].message.content)

        # Attach usage info (works for both branches).
        message["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

        messages.append(message)

        # Execute tool calling if present.
        if message.get("tool_calls"):
            tool_messages = call_oai_function(list(tools or []), message["tool_calls"])
            messages.extend(tool_messages)

        return messages


def _doc_first_paragraph(fn: Callable) -> str:
    """Return the first paragraph of a callable's docstring.

    Args:
        fn: A callable.

    Returns:
        The first paragraph of the docstring, or a fallback description.
    """
    doc = (inspect.getdoc(fn) or "").strip()
    if not doc:
        return f"Tool: {fn.__name__}"
    return doc.split("\n\n", 1)[0].strip()


def _build_args_model(fn: Callable) -> type[BaseModel]:
    """Build a Pydantic model from a function signature.

    This model is used to generate JSON Schema parameters for OpenAI tools, and
    to validate tool-call arguments at runtime.

    Args:
        fn: Tool function.

    Returns:
        A dynamically created Pydantic model class.
    """
    sig = inspect.signature(fn)
    # include_extras=True is important for supporting Annotated[..., Field(...)].
    hints = get_type_hints(fn, include_extras=True)

    fields: Dict[str, Tuple[Any, Any]] = {}
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            # *args/**kwargs are not included in the schema.
            continue

        ann = hints.get(name, Any)
        if p.default is inspect._empty:
            fields[name] = (ann, Field(...))
        else:
            fields[name] = (ann, Field(p.default))

    model_name = f"{fn.__name__.title().replace('_', '')}Args"
    return create_model(model_name, **fields)  # type: ignore[arg-type]


def to_oai_tools(fns: Iterable[Callable]) -> List[Dict[str, Any]]:
    """Convert Python callables into OpenAI tools schema.

    Args:
        fns: Iterable of tool callables.

    Returns:
        A list of tool definitions compatible with OpenAI Chat Completions API.
    """
    outputs: List[Dict[str, Any]] = []
    for fn in fns:
        args_model = _build_args_model(fn)
        parameters = args_model.model_json_schema()

        outputs.append(
            {
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": _doc_first_paragraph(fn),
                    "parameters": parameters,
                },
            }
        )

    return outputs


def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """Normalize `tool_calls` into a standard list-of-dicts format.

    The SDK may return:
    - list[dict]
    - list[OpenAI SDK objects] (with attributes: .id .type .function.name .function.arguments)

    This function normalizes them into:
        {"id": "...", "type": "function", "function": {"name": "...", "arguments": ...}}

    Args:
        tool_calls: Raw tool calls from the SDK response.

    Returns:
        Normalized list of tool call dictionaries.
    """
    if tool_calls is None:
        return []

    if isinstance(tool_calls, list) and (len(tool_calls) == 0 or isinstance(tool_calls[0], dict)):
        return tool_calls

    normalized: List[Dict[str, Any]] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            normalized.append(tc)
            continue

        tc_id = getattr(tc, "id", None)
        tc_type = getattr(tc, "type", None)

        fn_obj = getattr(tc, "function", None)
        if isinstance(fn_obj, dict):
            fn_name = fn_obj.get("name")
            fn_args = fn_obj.get("arguments")
        else:
            fn_name = getattr(fn_obj, "name", None)
            fn_args = getattr(fn_obj, "arguments", None)

        normalized.append({"id": tc_id, "type": tc_type, "function": {"name": fn_name, "arguments": fn_args}})

    return normalized


def _parse_arguments(arguments: Any) -> Dict[str, Any]:
    """Parse tool-call arguments into a Python dictionary.

    In OpenAI Chat Completions, `tool_call.function.arguments` is typically a JSON
    string, but may already be a dict.

    Args:
        arguments: Raw arguments from the tool call.

    Returns:
        Parsed arguments dict.

    Raises:
        ValueError: If the JSON is invalid or does not decode to an object.
        TypeError: If the arguments type is unsupported.
    """
    if arguments is None:
        return {}

    if isinstance(arguments, dict):
        return arguments

    if isinstance(arguments, str):
        s = arguments.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tool arguments: {e}. Raw={arguments!r}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"Tool arguments must be a JSON object (dict). Got {type(obj)}")
        return obj

    raise TypeError(f"Unsupported arguments type: {type(arguments)}")


def call_oai_function(
        fns: Iterable[Callable],
        tool_calls: Any,
        *,
        strict_validate: bool = True,
        on_error: str = "raise",  # "raise" | "return_error"
) -> List[Dict[str, Any]]:
    """Execute tool functions according to `message['tool_calls']`.

    Args:
        fns: The registered Python tool callables.
        tool_calls: Tool calls returned by the model (raw or normalized).
        strict_validate: If True, validate and coerce arguments using Pydantic.
            If False, call tools with raw kwargs.
        on_error: Error handling policy.
            - "raise": Raise immediately on any tool error.
            - "return_error": Return an error tool message and continue.

    Returns:
        A list of tool messages:
            [{"role": "tool", "tool_call_id": "...", "content": "..."}]

    Raises:
        ValueError/TypeError/ValidationError: If validation or parsing fails and
            `on_error="raise"`.
        KeyError: If the tool name is unknown and `on_error="raise"`.
        RuntimeError: If a tool returns an awaitable (use an async variant instead).
    """
    tc_list = _normalize_tool_calls(tool_calls)
    if not tc_list:
        return []

    fn_map: Dict[str, Callable] = {fn.__name__: fn for fn in fns}
    outputs: List[Dict[str, Any]] = []

    for tc in tc_list:
        tc_id = tc.get("id", uuid.uuid4().hex)
        fn_info = tc.get("function") or {}
        fn_name = fn_info.get("name")
        raw_args = fn_info.get("arguments")

        if not fn_name:
            err = ValueError(f"tool_call missing function.name: {tc!r}")
            if on_error == "return_error":
                outputs.append(
                    {"role": "tool", "tool_call_id": tc_id,
                     "content": json.dumps({"error": str(err)}, ensure_ascii=False)}
                )
                continue
            raise err

        fn = fn_map.get(fn_name)
        if fn is None:
            err = KeyError(f"Unknown tool/function name: {fn_name}")
            if on_error == "return_error":
                outputs.append(
                    {"role": "tool", "tool_call_id": tc_id,
                     "content": json.dumps({"error": str(err)}, ensure_ascii=False)}
                )
                continue
            raise err

        try:
            kwargs = _parse_arguments(raw_args)

            if strict_validate:
                model = _build_args_model(fn)
                parsed = model(**kwargs)
                kwargs = parsed.model_dump() if hasattr(parsed,
                                                        "model_dump") else parsed.dict()  # type: ignore[no-any-return]

            result = fn(**kwargs)

            if inspect.isawaitable(result):
                raise RuntimeError(
                    f"Tool {fn_name} returned an awaitable. "
                    f"Use an async tool runner (e.g., call_oai_function_async) for async tools."
                )

            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)

            outputs.append({"role": "tool", "tool_call_id": tc_id, "content": content})

        except (ValidationError, ValueError, TypeError, Exception) as e:
            if on_error == "return_error":
                outputs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(
                            {"error": str(e), "tool": fn_name, "arguments": raw_args},
                            ensure_ascii=False,
                            default=str,
                        ),
                    }
                )
                continue
            raise

    return outputs


__all__ = ["AgentConfig", "Agent", "ChatConfig"]
