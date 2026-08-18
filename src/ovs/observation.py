"""Observation stage: ROUTE samples and collect CRITIQUE evidence."""

import re
from typing import Any, Dict, Optional

from runtime import (
    PROMPT_LANGUAGE,
    call_tool,
    localized,
    objective_criterion,
    objective_en_name,
    objective_zh_name,
    record_context,
)


def _tool(name: str, description: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties}},
    }


ROUTE_TOOL = _tool(
    "route_objective",
    localized("为该样本选择唯一的主目标。", "Select the sample's single primary objective."),
    {"objective": {"type": "string", "enum": ["Value Orientation", "Affective Support", "Creative Innovation"]}},
)
CONTENT_CRITIC_TOOL = _tool(
    "content_critic",
    localized("从内容充分性角度评估辅导回答质量。", "Evaluate tutoring-answer quality under content adequacy."),
    {key: {"type": "string"} for key in ("score_content", "strengths", "missing_points", "logic_issues", "suggestions")},
)

ROUTE_SYSTEM = localized(
    """你是 Observation-Variation-Selection 协议中的 ROUTE 智能体。请为给定的教育情境、问题和回答指定唯一的主目标。有效目标仅为 Value Orientation、Affective Support 和 Creative Innovation。选择应主导后续评估的目标。必须调用 route_objective 工具返回结果。""",
    """You are a ROUTE agent in an Observation-Variation-Selection protocol. Assign the given educational scenario, question, and answer to exactly one primary objective. The only valid objectives are Value Orientation, Affective Support, and Creative Innovation. Choose the objective that should dominate later evaluation. Return the result through the route_objective tool.""",
)
STYLE_CRITIC_TOOL = _tool(
    "style_critic",
    localized("从表达清晰度与结构角度评估辅导回答质量。", "Evaluate tutoring-answer quality under writing clarity and structure."),
    {key: {"type": "string"} for key in ("score_style", "clarity_issues", "structure_suggestions", "expression_suggestions")},
)
DIM_CRITIC_TOOL = _tool(
    "dim_critic",
    localized("从指定人本目标评估辅导回答质量。", "Evaluate tutoring-answer quality under the routed human-centered objective."),
    {key: {"type": "string"} for key in ("score_dim", "strengths_dim", "suggestions_dim")},
)

CONTENT_CRITIC_SYSTEM = localized(
    """你是一名 K-12 辅导回答的督导评审。你将收到教育情境、问题、回答和内容充分性标准。仅依据该标准一致地评估回答：分别用一句话说明优点、缺失或弱点，以及可执行的改进建议，并给出 1 至 5 的整数分数。不要复述标准。必须调用 content_critic 工具返回结果。""",
    """You are a school-inspection reviewer for K-12 tutoring responses. You receive an educational scenario, question, answer, and a content-adequacy criterion. Apply that criterion consistently: give one sentence each for strengths, weaknesses or missing points, and actionable suggestions, plus an integer score from 1 to 5. Do not restate the criterion. Return the result through the content_critic tool.""",
)
STYLE_CRITIC_SYSTEM = localized(
    """你是一名 K-12 辅导回答的督导评审。你将从表达清晰度、结构和适切语气三个方面评估回答，并提供与 1 至 5 整数分数一致的优势、问题和可执行建议。不要复述评估标准。必须调用 style_critic 工具返回结果。""",
    """You are a school-inspection reviewer for K-12 tutoring responses. Evaluate the answer's clarity, structure, and role-appropriate tone. Provide strengths, weaknesses, and actionable suggestions consistent with an integer score from 1 to 5. Do not restate the evaluation criteria. Return the result through the style_critic tool.""",
)
DIM_CRITIC_SYSTEM = localized(
    """你是一名 K-12 辅导回答的督导评审。仅依据指定人本目标及其评价准则评估回答。分别用一句话说明优点、弱点和可执行建议，并给出与描述一致的 1 至 5 整数分数。不要复述目标或准则。必须调用 dim_critic 工具返回结果。""",
    """You are a school-inspection reviewer for K-12 tutoring responses. Evaluate the answer only under the routed human-centered objective and its criterion. Give one sentence each for strengths, weaknesses, and actionable suggestions, plus an integer score from 1 to 5. Do not restate the objective or criterion. Return the result through the dim_critic tool.""",
)


async def route(
    topic: str,
    question: str,
    answer: str,
    token_stats: Optional[Dict[str, int]] = None,
) -> str:
    """Use ROUTE to assign one paper-defined primary objective."""
    result = await call_tool(
        ROUTE_TOOL,
        ROUTE_SYSTEM,
        record_context(topic, question, answer),
        temperature=0.1,
        token_stats=token_stats,
    )
    return {
        "Value Orientation": "values",
        "Affective Support": "mental",
        "Creative Innovation": "creativity",
    }.get((result.get("objective") or "").strip(), "unknown")


def parse_score(value: Any, field_name: str) -> int:
    """Strictly parse a 1--5 critic score."""
    if isinstance(value, bool):
        score = int(value)
    elif isinstance(value, (int, float)):
        score = int(value)
    elif isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        score = int(value.strip())
    else:
        raise ValueError(f"{field_name} must be a numeric value: {value!r}")
    if not 1 <= score <= 5:
        raise ValueError(f"{field_name} is outside the range 1--5: {score}")
    return score


async def critique(
    objective_id: str,
    topic: str,
    question: str,
    answer: str,
    token_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Collect factorized content, style, and objective-specific critiques."""
    base_context = record_context(topic, question, answer)
    objective = objective_zh_name(objective_id) if PROMPT_LANGUAGE == "zh" else objective_en_name(objective_id)
    criterion = objective_criterion(objective_id)
    return {
        "content": await call_tool(CONTENT_CRITIC_TOOL, CONTENT_CRITIC_SYSTEM, base_context, temperature=0.1, token_stats=token_stats),
        "style": await call_tool(STYLE_CRITIC_TOOL, STYLE_CRITIC_SYSTEM, base_context, temperature=0.1, token_stats=token_stats),
        "objective": await call_tool(
            DIM_CRITIC_TOOL,
            DIM_CRITIC_SYSTEM,
            record_context(topic, question, answer, objective)
            + "\n\n"
            + localized(f"评价准则：{criterion}", f"Criterion: {criterion}"),
            temperature=0.1,
            token_stats=token_stats,
        ),
    }
