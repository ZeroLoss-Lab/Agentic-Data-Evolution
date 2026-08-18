"""Selection stage: SELECT the strongest candidate and ADMIT non-regressing updates."""

from typing import Any, Dict, List, Optional, Tuple

from runtime import (
    PROMPT_LANGUAGE,
    call_tool,
    localized,
    objective_criterion,
    objective_en_name,
    objective_zh_name,
)


def _tool(name: str, description: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties}}}


LOCAL_JUDGE_TOOL = _tool(
    "local_judge",
    localized("在父回答与候选回答中选择最佳版本。", "Choose the best version among the parent and candidate answers."),
    {"best_id": {"type": "string"}, "reason": {"type": "string"}},
)
GUARD_TOOL = _tool(
    "guard_check",
    localized("判断候选回答是否应提交，或应保留父回答。", "Decide whether to commit the candidate answer or retain the parent."),
    {"decision": {"type": "string"}, "reason": {"type": "string"}},
)

SELECT_SYSTEM = localized(
    """你是一名 OVS 协议中的 SELECT 评审。你将比较原始回答与候选回答，选择最佳版本。必须优先考虑已路由的主目标，同时考虑一般辅导质量。优先选择正确、有帮助且与学校情境一致的回答。必须调用 local_judge 工具，返回 best_id 和简短理由。""",
    """You are a SELECT judge in an OVS protocol. Compare the parent answer and candidate answers, then choose the best one. Prioritize the routed primary objective while also considering general tutoring quality. Prefer an answer that is correct, helpful, and aligned with the school context. Return best_id and a concise reason through the local_judge tool.""",
)
ADMIT_SYSTEM = localized(
    """你是一名 OVS 协议中的 ADMIT 门控。请在已路由主目标下比较父回答和候选回答。只有当候选明显更好，且在正确性、有帮助性和学校情境一致性等关键方面不更差时，才提交候选。情感支持必须保持非临床性质，并拒绝不安全或不恰当内容。必须调用 guard_check 工具：用 use_new 表示提交候选，用 use_original 表示保留父回答。""",
    """You are an ADMIT gate in an OVS protocol. Compare the parent answer and proposed answer under the routed primary objective. Commit the proposal only when it is clearly better and not worse on correctness, helpfulness, or alignment with the school context. Keep Affective Support non-clinical and reject unsafe or inappropriate content. Return use_new to commit the proposal or use_original to retain the parent through the guard_check tool.""",
)


def _select_prompt(objective_id: str, topic: str, question: str, parent: str, candidates: List[Tuple[str, str]]) -> str:
    objective = objective_zh_name(objective_id) if PROMPT_LANGUAGE == "zh" else objective_en_name(objective_id)
    criterion = objective_criterion(objective_id)
    parts = [localized(f"主评估目标：{objective}", f"Routed primary objective: {objective}"), localized(f"评价准则：{criterion}", f"Criterion: {criterion}"), "", localized("教育情境：", "Educational scenario:"), topic, "", localized("问题：", "Question:"), question, "", localized("父回答 A：", "Parent answer A:"), parent, ""]
    for candidate_id, answer in candidates:
        parts.extend([localized(f"候选回答 {candidate_id}：", f"Candidate answer {candidate_id}:"), answer, ""])
    return "\n".join(parts)


async def select(
    objective_id: str,
    topic: str,
    question: str,
    parent_answer: str,
    candidates: List[Tuple[str, str]],
    token_stats: Optional[Dict[str, int]] = None,
) -> Tuple[str, str]:
    """Use SELECT to choose the strongest parent or candidate answer."""
    result = await call_tool(
        LOCAL_JUDGE_TOOL,
        SELECT_SYSTEM,
        _select_prompt(objective_id, topic, question, parent_answer, candidates),
        temperature=0.1,
        token_stats=token_stats,
    )
    return (result.get("best_id") or "A").strip(), (result.get("reason") or "").strip()


async def admit(objective_id: str, question: str, parent_answer: str, candidate_answer: str, token_stats: Optional[Dict[str, int]] = None) -> Tuple[str, str]:
    """Use ADMIT to commit a selected candidate or retain its parent."""
    objective = objective_zh_name(objective_id) if PROMPT_LANGUAGE == "zh" else objective_en_name(objective_id)
    criterion = objective_criterion(objective_id)
    prompt = localized(
        f"""主评估目标：{objective}
评价准则：{criterion}

问题：
{question}

父回答 A：
{parent_answer}

候选回答 A_new：
{candidate_answer}""",
        f"""Routed primary objective: {objective}
Criterion: {criterion}

Question:
{question}

Parent answer A:
{parent_answer}

Proposed answer A_new:
{candidate_answer}""",
    )
    result = await call_tool(GUARD_TOOL, ADMIT_SYSTEM, prompt, temperature=0.1, token_stats=token_stats)
    decision = (result.get("decision") or "").lower()
    normalized = "use_new" if "new" in decision else "use_original"
    return normalized, (result.get("reason") or "").strip()
