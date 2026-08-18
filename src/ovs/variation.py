"""Variation stage: MUTATE the parent answer into diverse candidates."""

import json
from typing import Any, Dict, Optional

from runtime import PROMPT_LANGUAGE, call_text, localized, objective_en_name, objective_zh_name

EDITOR_MAIN_SYSTEM = localized(
    """你是一名 OVS 协议中的 MUTATE 智能体，负责对 K-12 辅导回答进行较大幅度的修订。你将收到问题、当前回答、批评证据和可选的上一轮拒绝信号。可重组回答，并在有助于学习者完成任务时补充缺失步骤；但必须保持任务意图，不得改变问题含义，不得编造情境未支持的事实。尊重学校情境，情感支持不得包含临床诊断、治疗建议或危机干预。仅输出修订后的完整回答。""",
    """You are a MUTATE agent in an OVS protocol performing aggressive revision of K-12 tutoring answers. You receive a question, current answer, critique evidence, and an optional prior rejection signal. You may reorganize the response and add missing steps when they help the learner achieve the task. Preserve task intent, do not change the question's meaning, and do not invent facts unsupported by the scenario. Respect the school context; Affective Support must remain non-clinical. Output only the complete revised answer.""",
)
EDITOR_LIGHT_SYSTEM = localized(
    """你是一名 OVS 协议中的 MUTATE 智能体，负责对 K-12 辅导回答进行保守修订。你将收到问题、当前回答、批评证据和可选的上一轮拒绝信号。尽可能保留原意与主要结构，只做必要修改，避免加入无关内容。尊重学校情境，情感支持不得包含临床诊断、治疗建议或危机干预。仅输出修订后的完整回答。""",
    """You are a MUTATE agent in an OVS protocol performing conservative revision of K-12 tutoring answers. You receive a question, current answer, critique evidence, and an optional prior rejection signal. Preserve the intent and, where possible, the main structure; make only necessary edits and avoid unrelated content. Respect the school context; Affective Support must remain non-clinical. Output only the complete revised answer.""",
)
def difference_ratio(parent: str, candidate: str) -> float:
    """Approximate edit magnitude with a character-level difference ratio."""
    parent, candidate = parent or "", candidate or ""
    if not parent and not candidate:
        return 0.0
    length = max(len(parent), len(candidate))
    same = sum(parent[i] == candidate[i] for i in range(min(len(parent), len(candidate))))
    return 1.0 - same / length if length else 0.0


def _prompt(
    topic: str,
    question: str,
    answer: str,
    objective_id: str,
    critiques: Dict[str, Dict[str, Any]],
    iteration_index: int,
    variant: str,
    previous_candidate: Optional[str],
    previous_select_reason: Optional[str],
    previous_admit_reason: Optional[str],
) -> str:
    objective = objective_zh_name(objective_id) if PROMPT_LANGUAGE == "zh" else objective_en_name(objective_id)
    parts = [localized("教育情境：", "Educational scenario:"), topic, "", localized("问题：", "Question:"), question, "", localized("当前回答：", "Current answer:"), answer, ""]
    for label, key in (("[内容评审]", "content"), ("[表达评审]", "style"), (f"[{objective} 目标评审]", "objective")):
        translated = {"[内容评审]": "[Content critique]", "[表达评审]": "[Style critique]"}.get(label, f"[{objective} objective critique]")
        parts.extend([localized(label, translated), json.dumps(critiques.get(key, {}), ensure_ascii=False, indent=2), ""])
    if iteration_index == 0:
        parts.extend([localized(f"这是第 1 次改写，当前编辑模式为：{variant}。", f"This is revision 1. Local editor mode: {variant}."), localized("请按照系统提示修订当前回答。", "Revise the current answer under the system instructions.")])
    else:
        parts.append(localized(f"这是第 {iteration_index + 1} 次改写，当前编辑模式为：{variant}。", f"This is revision {iteration_index + 1}. Local editor mode: {variant}."))
        if previous_candidate:
            parts.extend(["", localized("以下为上一轮候选及反馈，请针对反馈修订。", "Below are the prior candidate and feedback. Revise in response to the feedback."), localized("【上一轮候选】", "[Prior candidate]"), previous_candidate])
        if previous_select_reason or previous_admit_reason:
            parts.extend(["", localized("【比较与门控反馈】", "[Selection and admission feedback]")])
            if previous_select_reason:
                parts.append(localized(f"- SELECT：{previous_select_reason}", f"- SELECT: {previous_select_reason}"))
            if previous_admit_reason:
                parts.append(localized(f"- ADMIT：{previous_admit_reason}", f"- ADMIT: {previous_admit_reason}"))
        parts.extend(["", localized("保留已有的有效内容，避免无关扩写或空洞表述。", "Retain what already works and avoid unrelated expansion or empty statements.")])
    parts.extend(["", localized("仅输出修订后的完整回答，不要解释。", "Output only the complete revised answer, without explanation.")])
    return "\n".join(parts)


async def mutate(
    variant: str,
    topic: str,
    question: str,
    parent_answer: str,
    objective_id: str,
    critiques: Dict[str, Dict[str, Any]],
    iteration_index: int,
    previous_candidate: Optional[str],
    previous_select_reason: Optional[str],
    previous_admit_reason: Optional[str],
    token_stats: Optional[Dict[str, int]] = None,
) -> str:
    """Generate one aggressive or conservative MUTATE candidate."""
    system_prompt = {"main": EDITOR_MAIN_SYSTEM, "light": EDITOR_LIGHT_SYSTEM}.get(variant)
    if system_prompt is None:
        raise ValueError(f"Unknown MUTATE variant: {variant}")
    return (await call_text(
        system_prompt,
        _prompt(topic, question, parent_answer, objective_id, critiques, iteration_index, variant, previous_candidate, previous_select_reason, previous_admit_reason),
        temperature=0.4 if variant == "main" else 0.2,
        token_stats=token_stats,
    )).strip()

