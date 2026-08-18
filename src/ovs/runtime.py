"""Shared runtime support for the three OVS protocol stages."""

import logging
import json
import os
import random
import re
import traceback
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

OBJECTIVE_NAMES = {
    "values": "Value Orientation (VO)",
    "mental": "Affective Support (AS)",
    "creativity": "Creative Innovation (CI)",
}

OBJECTIVE_CRITERIA = {
    "values": {
        "zh": "回答应符合教育情境中的价值判断、社会规范和责任边界，并避免不恰当的说教或伤害性引导。",
        "en": "The answer should maintain appropriate value judgments, social norms, and responsibility boundaries for the educational context, without inappropriate moralizing or harmful guidance.",
    },
    "mental": {
        "zh": "回答应提供适切的情感支持和学习者自我调节引导，保持与角色和情境一致的语气，并严格保持非临床边界。",
        "en": "The answer should provide appropriate affective support and learner self-regulation guidance with a role- and context-appropriate tone, while remaining strictly non-clinical.",
    },
    "creativity": {
        "zh": "回答应在任务约束下促进新颖且适切、可行的想法或表达，并保持与问题和教育情境一致。",
        "en": "The answer should promote novel, appropriate, and feasible ideas or expression under the task constraints while remaining consistent with the question and educational context.",
    },
}

PROMPT_LANGUAGE = os.getenv("OVS_PROMPT_LANGUAGE", "zh")
if PROMPT_LANGUAGE not in ("zh", "en"):
    raise ValueError("OVS_PROMPT_LANGUAGE must be 'zh' or 'en'")

INPUT_DIR = os.getenv("OVS_INPUT_DIR", "improve_100k")
OUTPUT_SUBDIR = os.getenv("OVS_OUTPUT_DIR", "improve_100k/output")
ROUND_COUNT = int(os.getenv("OVS_ROUNDS", "4"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("OVS_CONCURRENT", "1200"))
REWRITE_MAX_ROUNDS = int(os.getenv("OVS_REWRITE_ROUNDS", "2"))
MAX_SAMPLE_TIMEOUT_SECONDS = int(os.getenv("OVS_TIMEOUT", "720"))
MIN_DIFF_MAIN = float(os.getenv("OVS_MIN_DIFF_MAIN", "0.05"))
MIN_DIFF_LIGHT = float(os.getenv("OVS_MIN_DIFF_LIGHT", "0.03"))
ERROR_LOG_INTERVAL = int(os.getenv("OVS_ERROR_LOG_INTERVAL", "10"))
MAX_RETRIES = 2
KEEPALIVE_CONNECTIONS = 10
QWEN_MODEL = os.getenv("OVS_MODEL", "qwen")
QWEN_API_KEY = os.getenv("OVS_API_KEY", "EMPTY")
DISABLE_THINKING = os.getenv("OVS_DISABLE_THINKING", "true").lower() == "true"
SERVICE_URLS = [
    url.strip() for url in os.getenv("OVS_SERVICE_URLS", "").split(";") if url.strip()
]
TOOL_CALLS_SUPPORTED: Optional[bool] = None

logger = logging.getLogger("ovs")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    log_file = os.getenv("OVS_LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(file_handler)


def localized(zh: str, en: str) -> str:
    """Return the configured Chinese or English production prompt."""
    return zh if PROMPT_LANGUAGE == "zh" else en


def objective_en_name(objective_id: str) -> str:
    return OBJECTIVE_NAMES.get(objective_id, "Human-Centered Objective")


def objective_zh_name(objective_id: str) -> str:
    return {
        "values": "价值导向（VO）",
        "mental": "情感支持（AS）",
        "creativity": "创造性创新（CI）",
    }.get(objective_id, "人本目标")


def objective_criterion(objective_id: str) -> str:
    """Return the paper-defined criterion for a routed objective."""
    return OBJECTIVE_CRITERIA.get(objective_id, {}).get(PROMPT_LANGUAGE, "")


def record_context(topic: str, question: str, answer: str, objective: Optional[str] = None) -> str:
    if PROMPT_LANGUAGE == "en":
        prefix = f"Routed primary objective: {objective}\n\n" if objective else ""
        return f"{prefix}Educational scenario:\n{topic}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"
    prefix = f"当前主目标：{objective}\n\n" if objective else ""
    return f"{prefix}教育情境：\n{topic}\n\n问题：\n{question}\n\n回答：\n{answer}"


async def _request(
    system_prompt: str,
    user_prompt: str,
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_name: Optional[str] = None,
    temperature: float,
    token_stats: Optional[Dict[str, int]],
) -> Any:
    import httpx

    if not SERVICE_URLS:
        raise RuntimeError("OVS_SERVICE_URLS must contain at least one endpoint")
    if token_stats is not None:
        token_stats["api_calls"] = token_stats.get("api_calls", 0) + 1

    http_client = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(
            retries=MAX_RETRIES,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=KEEPALIVE_CONNECTIONS),
        ),
        timeout=httpx.Timeout(connect=20.0, read=180.0, write=180.0, pool=30.0),
    )
    client = AsyncOpenAI(
        api_key=QWEN_API_KEY,
        base_url=random.choice(SERVICE_URLS),
        http_client=http_client,
    )
    try:
        request: Dict[str, Any] = {
            "model": QWEN_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if DISABLE_THINKING:
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        if tools is not None and tool_name is not None:
            request["tools"] = tools
            request["tool_choice"] = {"type": "function", "function": {"name": tool_name}}
        response = await client.chat.completions.create(**request)
    except Exception as error:
        logger.error("OVS API call failed: %s\n%s", error, traceback.format_exc())
        raise
    finally:
        await client.close()
        await http_client.aclose()

    if token_stats is not None and response.usage is not None:
        token_stats["prompt"] += response.usage.prompt_tokens or 0
        token_stats["completion"] += response.usage.completion_tokens or 0
        token_stats["total"] += response.usage.total_tokens or 0
    return response


async def call_text(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float,
    token_stats: Optional[Dict[str, int]] = None,
) -> str:
    """Call the backbone model for an unstructured response."""
    response = await _request(
        system_prompt, user_prompt, temperature=temperature, token_stats=token_stats
    )
    return response.choices[0].message.content or ""


def _recover_tool_arguments(tool_name: str, arguments: str) -> Dict[str, Any]:
    text = (arguments or "").strip()
    if tool_name == "local_judge":
        explicit = re.search(r"best_id\s*[:：=是]\s*(A|C1|C2|C3)", text, re.IGNORECASE)
        best_id = explicit.group(1) if explicit else next((c for c in ("C1", "C2", "C3", "A") if c in text), None)
        if best_id:
            return {"best_id": best_id, "reason": text or f"Inferred best_id={best_id}"}
    if tool_name == "guard_check":
        if "use_original" in text.lower() or "原答案" in text or "保持原回答" in text:
            return {"decision": "use_original", "reason": text or "Inferred parent retention"}
        if "use_new" in text.lower() or "新答案" in text or "改写后的答案" in text or "采用改写" in text:
            return {"decision": "use_new", "reason": text or "Inferred candidate commitment"}
    raise RuntimeError(f"Unable to recover arguments for {tool_name}: {arguments[:200]}")


async def call_tool(
    tool: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float,
    token_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Call a stage tool and return its structured response."""
    global TOOL_CALLS_SUPPORTED
    tool_name = tool["function"]["name"]
    arguments = ""
    if TOOL_CALLS_SUPPORTED is not False:
        response = await _request(
            system_prompt,
            user_prompt,
            tools=[tool],
            tool_name=tool_name,
            temperature=temperature,
            token_stats=token_stats,
        )
        calls = response.choices[0].message.tool_calls
        if calls:
            TOOL_CALLS_SUPPORTED = True
            arguments = calls[0].function.arguments or ""
        else:
            TOOL_CALLS_SUPPORTED = False
    if not arguments:
        properties = tool["function"]["parameters"].get("properties", {})
        schema = {name: "..." for name in properties}
        fallback_system = (
            f"{system_prompt}\n\nThe function interface is unavailable for this request. "
            f"Do not call a tool. Return only a JSON object for {tool_name} "
            f"with these fields: {json.dumps(schema, ensure_ascii=False)}."
        )
        fallback = await _request(
            fallback_system,
            user_prompt,
            temperature=temperature,
            token_stats=token_stats,
        )
        arguments = fallback.choices[0].message.content or ""
    try:
        return json.loads(arguments)
    except ValueError:
        return _recover_tool_arguments(tool_name, arguments)
