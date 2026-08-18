# -*- coding: utf-8 -*-
import asyncio
import json
import os
import re
from argparse import ArgumentParser
from typing import Any, Dict, Iterable, List, Optional
from tqdm import tqdm

from agent import Agent, AgentConfig, ChatConfig


async def _maybe_await(x):
    """Await a value if it's awaitable, otherwise return it directly."""
    if asyncio.iscoroutine(x) or isinstance(x, asyncio.Future):
        return await x
    return x

# =========================
# Interface aligned with the example eval.py.
# =========================

SYSTEM_PROMPTS = {
    "zh": """你是一位专业的教育评估专家。你将根据给定的评估维度与评分标准，对模型回答进行打分并给出理由。请只输出 JSON，不要输出多余文字。""",
    "en": """You are a professional educational evaluator. Score the model answer using the supplied metric and rubric, then provide a concise rationale. Output JSON only, with no additional text.""",
}


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _strip_code_fence(s: str) -> str:
    return re.sub(r"```(json)?\s*|\s*```", "", s or "").strip()


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    s = _strip_code_fence(s)
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


# =========================
# Evaluation metrics and prompts.
# =========================
METRICS = {
    "IFTC": "指令遵循与任务完成",
    "RTC": "角色与语气一致性",
    "CRSC": "内容相关性与范围控制",
    "SEI": "场景元素整合",
    "BFA": "基础事实准确性",
    "DKA": "领域知识准确性",
    "RPR": "推理过程严谨性",
    "EICP": "错误识别与纠正精度",
    "CSI": "清晰简洁与启发性",
    "MGP": "激励引导与正向反馈",
    "PAS": "个性化适应与学习支持",
    "HOTS": "高阶思维与技能发展"
}

# Domain groups: average each domain, then average across domains.
# Stable output keys follow the EduBench artifact schema. Their paper-level
# interpretations are Value Orientation (VO), Affective Support (AS), and
# Creative Innovation (CI), respectively.
OBJECTIVE_LABELS = {
    "values": "Value Orientation (VO)",
    "mental_health": "Affective Support (AS)",
    "creativity": "Creative Innovation (CI)",
}
DOMAIN_GROUPS = {
    "creativity": ["CSI", "HOTS", "PAS"],
    "mental_health": ["MGP", "RTC", "SEI"],
    "values": ["IFTC", "CRSC", "BFA", "DKA", "RPR", "EICP"],
}

# Detailed five-level scoring rubrics.
SCORING_RUBRICS = {
    "IFTC": {
        "name": "指令遵循与任务完成",
        "description": "评估模型是否准确理解并执行了用户指令，完成了指定任务",
        "rubrics": {
            "9-10": "完全理解并精确执行所有指令要求，任务完成度100%，输出格式和内容完全符合预期",
            "7-8": "理解并执行了大部分指令，任务基本完成，可能存在少量格式或细节偏差",
            "5-6": "部分理解指令，完成了主要任务但遗漏了一些要求，输出有明显不足",
            "3-4": "对指令理解有限，任务完成度较低，存在多处偏离或遗漏",
            "1-2": "未能理解指令或完全偏离任务要求，输出与预期严重不符"
        }
    },
    "RTC": {
        "name": "角色与语气一致性",
        "description": "评估模型是否保持了适当的角色定位和语气风格",
        "rubrics": {
            "9-10": "完美保持角色定位（如教师、辅导员），语气始终符合教育场景，专业且得体",
            "7-8": "角色定位清晰，语气基本得体，偶有轻微偏离",
            "5-6": "角色定位不够稳定，语气有时不够专业或与场景不符",
            "3-4": "角色定位模糊，语气经常不当或与教育场景脱节",
            "1-2": "完全缺乏角色意识，语气不适合教育场景或存在严重问题"
        }
    },
    "CRSC": {
        "name": "内容相关性与范围控制",
        "description": "评估回答内容是否紧扣主题，范围控制是否恰当",
        "rubrics": {
            "9-10": "内容高度相关，范围控制精准，无冗余信息，重点突出",
            "7-8": "内容相关性强，范围基本合理，可能有少量偏离或冗余",
            "5-6": "内容基本相关但存在偏题或范围过宽/过窄的问题",
            "3-4": "内容相关性较弱，存在明显偏题或大量无关信息",
            "1-2": "内容严重偏离主题，范围控制失当，大部分内容不相关"
        }
    },
    "SEI": {
        "name": "场景元素整合",
        "description": "评估是否有效整合了场景中的关键元素",
        "rubrics": {
            "9-10": "完美整合所有场景元素，体现在回答的各个方面，元素运用自然贴切",
            "7-8": "有效整合了主要场景元素，大部分元素得到体现和运用",
            "5-6": "整合了部分场景元素，但有重要元素被忽视或运用不当",
            "3-4": "场景元素整合较少，大部分关键元素未被考虑或体现",
            "1-2": "完全忽视场景元素，回答缺乏针对性"
        }
    },
    "BFA": {
        "name": "基础事实准确性",
        "description": "评估基本事实、概念、定义等是否准确无误",
        "rubrics": {
            "9-10": "所有基础事实完全准确，无任何错误或误导性陈述",
            "7-8": "绝大部分事实准确，可能存在极少量次要错误",
            "5-6": "主要事实基本准确，但存在一些明显错误",
            "3-4": "事实准确性较差，存在多处错误或不准确陈述",
            "1-2": "基础事实严重错误，存在大量错误或误导性信息"
        }
    },
    "DKA": {
        "name": "领域知识准确性",
        "description": "评估学科专业知识的准确性和深度",
        "rubrics": {
            "9-10": "领域知识完全准确，深度适当，体现专业水准",
            "7-8": "领域知识基本准确，深度合理，可能有少量小瑕疵",
            "5-6": "领域知识大致正确但深度不足或存在部分错误",
            "3-4": "领域知识准确性较差，存在明显错误或深度严重不足",
            "1-2": "领域知识严重错误或完全缺乏专业性"
        }
    },
    "RPR": {
        "name": "推理过程严谨性",
        "description": "评估推理逻辑的严密性和论证的充分性",
        "rubrics": {
            "9-10": "推理逻辑严密，论证充分，步骤清晰，结论可靠",
            "7-8": "推理基本严谨，论证较为充分，可能有少量逻辑跳跃",
            "5-6": "推理大致合理但存在逻辑漏洞或论证不够充分",
            "3-4": "推理存在明显逻辑问题，论证薄弱",
            "1-2": "推理混乱，逻辑错误严重，缺乏论证"
        }
    },
    "EICP": {
        "name": "错误识别与纠正精度",
        "description": "评估识别错误的准确性和纠正方法的有效性",
        "rubrics": {
            "9-10": "准确识别所有错误，纠正方法精准有效，解释清晰透彻",
            "7-8": "识别了主要错误，纠正方法基本正确，解释较为清晰",
            "5-6": "识别了部分错误，但有遗漏或误判，纠正方法基本可行",
            "3-4": "错误识别不准确，纠正方法存在问题或效果有限",
            "1-2": "未能识别错误或识别错误，纠正方法无效或错误"
        }
    },
    "CSI": {
        "name": "清晰简洁与启发性",
        "description": "评估表达的清晰度、简洁性以及对学生的启发作用",
        "rubrics": {
            "9-10": "表达清晰简洁，易于理解，具有很强的启发性，能引导学生深入思考",
            "7-8": "表达清晰，基本简洁，有一定启发性",
            "5-6": "表达基本清楚但不够简洁，启发性较弱",
            "3-4": "表达不够清晰或过于冗长，缺乏启发性",
            "1-2": "表达混乱，难以理解，完全缺乏启发性"
        }
    },
    "MGP": {
        "name": "激励引导与正向反馈",
        "description": "评估是否提供了积极的激励和建设性的引导",
        "rubrics": {
            "9-10": "激励恰当有力，引导明确有效，充满正能量，能显著提升学生信心和动力",
            "7-8": "提供了较好的激励和引导，基本正向，效果良好",
            "5-6": "有一定激励和引导，但力度不足或方式欠佳",
            "3-4": "激励和引导较弱，效果有限",
            "1-2": "缺乏激励和引导，或存在负面消极内容"
        }
    },
    "PAS": {
        "name": "个性化适应与学习支持",
        "description": "评估是否针对学生个体差异提供了个性化的学习支持",
        "rubrics": {
            "9-10": "高度个性化，完全契合学生特点和需求，支持措施具体有效",
            "7-8": "较好地体现个性化，支持措施基本符合学生需求",
            "5-6": "有一定个性化考虑，但针对性不强或支持措施不够具体",
            "3-4": "个性化程度较低，支持措施缺乏针对性",
            "1-2": "完全缺乏个性化，未考虑学生个体差异"
        }
    },
    "HOTS": {
        "name": "高阶思维与技能发展",
        "description": "评估是否促进了批判性思维、创造性思维等高阶认知能力的发展",
        "rubrics": {
            "9-10": "显著促进高阶思维发展，引导学生进行深度分析、综合、评价和创造",
            "7-8": "较好地促进高阶思维，有引导学生深入思考的设计",
            "5-6": "有一定的高阶思维引导，但深度和广度不足",
            "3-4": "较少涉及高阶思维，主要停留在记忆和理解层面",
            "1-2": "完全缺乏高阶思维引导，仅涉及低层次认知"
        }
    }
}

# Metrics evaluated for each task type.
TASK_METRICS = {
    "QA": ["IFTC", "CRSC", "BFA", "DKA", "CSI", "MGP", "HOTS"],
    "EC": ["IFTC", "RTC", "CRSC", "BFA", "EICP", "CSI", "MGP"],
    "IP": ["IFTC", "CRSC", "RPR", "CSI", "MGP", "HOTS"],
    "PLS": ["IFTC", "SEI", "PAS", "CSI", "MGP"],
    "ES": ["RTC", "SEI", "CSI", "MGP"],
    "QG": ["IFTC", "CRSC", "DKA", "CSI"],
    "AG": ["IFTC", "CRSC", "BFA", "EICP", "CSI", "MGP", "HOTS"],
    "TMG": ["IFTC", "CRSC", "DKA", "CSI", "HOTS"],
    "PCC": ["IFTC", "SEI", "PAS", "CSI", "MGP"]
}


def build_task_context(task_type: str, task_data: Dict[str, Any]) -> str:
    """Build task-specific evaluation context."""
    parts = []

    if task_type == "QA":
        parts.append(f"问题：{task_data.get('Question', '')}")
        parts.append(f"标准答案：{task_data.get('Answer', '')}")
        parts.append(f"学科：{task_data.get('Subject', '')}")
        parts.append(f"教育阶段：{task_data.get('Education Level', '')}")

    elif task_type == "EC":
        parts.append(f"原始问题：{task_data.get('Question', '')}")
        parts.append(f"学生的错误答案：{task_data.get('Student Answer', '')}")
        parts.append(f"正确答案：{task_data.get('Correct Answer', '')}")

    elif task_type == "IP":
        parts.append(f"问题：{task_data.get('Question', '')}")
        parts.append(f"学科：{task_data.get('Subject', '')}")

    elif task_type == "PLS":
        parts.append(f"学生画像：{task_data.get('Student Profile', '')}")
        parts.append(f"学科：{task_data.get('Subject', '')}")

    elif task_type == "ES":
        parts.append(f"对话场景：{task_data.get('Scenario Description', '')}")
        parts.append(f"焦虑级别：{task_data.get('Anxiety Level', '')}")

    elif task_type == "QG":
        parts.append(f"知识点：{task_data.get('Knowledge Point', '')}")
        parts.append(f"学科：{task_data.get('Subject', '')}")
        parts.append(f"难度：{task_data.get('Education Level', '')}")

    elif task_type == "AG":
        parts.append(f"题目：{task_data.get('Question', '')}")
        parts.append(f"学生答案：{task_data.get('Student Answer', '')}")
        parts.append(f"标准答案：{task_data.get('Standard Answer', '')}")

    elif task_type == "TMG":
        parts.append(f"知识点：{task_data.get('Knowledge Point', '')}")
        parts.append(f"学科：{task_data.get('Subject', '')}")

    elif task_type == "PCC":
        parts.append(f"学生画像：{task_data.get('Student Profile', '')}")
        parts.append(f"内容类型：{task_data.get('Content Type', '')}")

    return "\n".join(parts)


def build_eval_prompt(
        task_type: str,
        task_data: Dict[str, Any],
        model_answer: str,
        metric: str,
        prompt_language: str,
) -> str:
    """Build the local EduBench metric-evaluation prompt."""
    metric_info = SCORING_RUBRICS.get(metric, {})
    metric_name = metric_info.get("name", metric)
    metric_desc = metric_info.get("description", "")
    rubrics = metric_info.get("rubrics", {})

    rubrics_text = "\n".join([f"【{score}分】{desc}" for score, desc in rubrics.items()])
    context = build_task_context(task_type, task_data)

    if prompt_language == "en":
        return f"""You are an educational evaluator assessing an AI educational assistant.

Evaluation task: score the answer from 1 to 10 using the metric and rubric below.

Metric: {metric_name} ({metric})
Description: {metric_desc}

Rubric:
{rubrics_text}

Task type: {task_type}
{context}

AI assistant answer:
{model_answer}

Return JSON only:
{{
    "strengths": "<strengths>",
    "weaknesses": "<weaknesses>",
    "suggestions": "<actionable suggestions>",
    "reasoning": "<scoring rationale>",
    "score": <integer from 1 to 10>
}}"""

    return f"""你是一位教育评估专家，负责评估 AI 教育助手的回答质量。

【评估任务】
请根据以下评估维度对AI助手的回答进行打分（1-10分制）：

评估维度：{metric_name} ({metric})
维度说明：{metric_desc}

【评分标准】
{rubrics_text}

【任务背景】
任务类型：{task_type}
{context}

【AI助手的回答】
{model_answer}

【评估要求】
1. 严格按照上述5级评分标准进行评估
2. 给出1-10之间的整数分数
3. 提供详细的评分理由

请以JSON格式返回评估结果：
{{
    "优点": "<列出回答的优点>",
    "不足": "<指出回答的不足之处>",
    "改进建议": "<提供具体的改进建议>",
    "评分理由": "<详细说明评分依据>",
    "分数": <1-10之间的整数>
}}"""


async def evaluate_single_metric(
        task_type: str,
        task_data: Dict[str, Any],
        model_answer: str,
        metric: str,
        agent: Agent,
        args: Any,
) -> Optional[Dict[str, Any]]:
    """Evaluate one metric through the API interface used by example eval.py."""
    prompt = build_eval_prompt(task_type, task_data, model_answer, metric, args.prompt_language)

    chat_config = ChatConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        enable_thinking=(not args.no_thinking),
    )

    msgs = await agent.chat(
        prompt,
        config=chat_config
    )

    content = (msgs[-1] or {}).get("content", "")
    result = _safe_json_loads(content)
    if not result:
        return None

    return {
        "metric": metric,
        "metric_name": METRICS.get(metric, metric),
        "score": int(result.get("分数", result.get("score", 0)) or 0),
        "reasoning": result.get("评分理由", result.get("reasoning", "")) or "",
        "strengths": result.get("优点", result.get("strengths", "")) or "",
        "weaknesses": result.get("不足", result.get("weaknesses", "")) or "",
        "suggestions": result.get("改进建议", result.get("suggestions", "")) or "",
        "judge_raw": content,
    }


async def evaluate_task(
        task: Dict[str, Any],
        agent: Agent,
        args: Any,
        metric_semaphore: asyncio.Semaphore,
        previous_evaluations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Evaluate one sample while preserving the original evaluate.py schema.

    Args:
        task: Task to evaluate
        agent: Agent for judge API
        args: Arguments
        metric_semaphore: Semaphore for concurrent metric evaluation
        previous_evaluations: Previous successful evaluations (for resume)
    """
    task_type = task.get("task_type")
    model_name = task.get("model_name", "unknown")
    model_answer = task.get("model_answer", "")

    if not task_type or not isinstance(model_answer, str):
        return {
            "ok": False,
            "id": task.get("id"),
            "task_id": task.get("task_id"),
            "error": "missing task_type or model_answer",
        }

    metrics = TASK_METRICS.get(task_type, list(METRICS.keys()))
    evaluations: List[Dict[str, Any]] = list(previous_evaluations) if previous_evaluations else []

    # Get set of already evaluated metrics
    evaluated_metrics = {e["metric"] for e in evaluations}

    # Only run metrics that haven't been successfully evaluated yet
    metrics_to_run = [m for m in metrics if m not in evaluated_metrics]

    if metrics_to_run:
        # Concurrent evaluation of metrics with semaphore control
        async def evaluate_with_semaphore(metric: str) -> Optional[Dict[str, Any]]:
            async with metric_semaphore:
                try:
                    # Timeout per metric: 120 seconds
                    metric_timeout = args.task_timeout
                    return await asyncio.wait_for(
                        evaluate_single_metric(task_type, task, model_answer, metric, agent, args),
                        timeout=metric_timeout
                    )
                except asyncio.TimeoutError:
                    print(f"[WARN] Metric {metric} timeout for task {task.get('id')}")
                    return None
                except Exception as e:
                    print(f"[ERROR] Metric {metric} failed for task {task.get('id')}: {e}")
                    return None

        # Create tasks for metrics to run
        metric_tasks = [evaluate_with_semaphore(metric) for metric in metrics_to_run]

        # Wait for all metric evaluations to complete
        results = await asyncio.gather(*metric_tasks, return_exceptions=True)

        # Collect successful results
        for result in results:
            if isinstance(result, Exception):
                # Log error but continue with other metrics
                continue
            if result:
                evaluations.append(result)

    # Average scores within each domain.
    metric_scores = {e["metric"]: float(e.get("score", 0) or 0) for e in evaluations}
    domain_scores: Dict[str, float] = {}
    for domain, metric_list in DOMAIN_GROUPS.items():
        vals = [metric_scores[m] for m in metric_list if m in metric_scores]
        if vals:
            domain_scores[domain] = sum(vals) / len(vals)

    avg_score = (sum(domain_scores.values()) / len(domain_scores)) if domain_scores else 0.0

    # Only set ok=True if all metrics succeeded
    all_metrics_ok = len(evaluations) == len(metrics)

    # Track failed metrics for partial resume
    evaluated_metric_set = {e["metric"] for e in evaluations}
    failed_metrics = [m for m in metrics if m not in evaluated_metric_set]

    return {
        "ok": all_metrics_ok,
        "id": task.get("id"),
        "task_id": task.get("task_id"),
        "task_type": task_type,
        "model": model_name,
        "response": model_answer,
        "evaluations": evaluations,
        "domain_scores": {k: round(v, 2) for k, v in domain_scores.items()},
        "average_score": round(avg_score, 2),
        "total_metrics": len(metrics),
        "evaluated_metrics": len(evaluations),
        "failed_metrics": failed_metrics if failed_metrics else [],
    }


def _compute_stats(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    ok = 0
    fail = 0
    scores: List[float] = []

    for r in records:
        if r.get("ok") is True:
            ok += 1
        else:
            fail += 1
        # Include scores from both ok=True and ok=False tasks (if they have partial results)
        if r.get("domain_scores"):
            try:
                scores.append(float(r.get("average_score", 0) or 0))
            except Exception:
                pass

    total = ok + fail
    avg = (sum(scores) / len(scores)) if scores else 0.0
    return {"ok": ok, "fail": fail, "total": total, "avg_average_score": round(avg, 4)}


def _get_category(r: Dict[str, Any]) -> str:
    # Use task_type as the category concept used by the example script.
    return r.get("task_type") or "unknown"


def _compute_stats_by_category(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    cat2scores: Dict[str, List[float]] = {}
    cat2total: Dict[str, int] = {}

    for r in records:
        cat = _get_category(r)
        # Count all tasks (including ok=False)
        cat2total[cat] = cat2total.get(cat, 0) + 1
        # Include scores from tasks with domain_scores (including ok=False)
        if r.get("domain_scores"):
            try:
                cat2scores.setdefault(cat, []).append(float(r.get("average_score", 0) or 0))
            except Exception:
                pass

    out: Dict[str, Any] = {}
    for cat, vals in cat2scores.items():
        out[cat] = {
            "count": cat2total.get(cat, 0),
            "count_with_scores": len(vals),
            "avg_average_score": round(sum(vals) / len(vals), 4) if vals else 0.0,
        }
    return out



def _compute_stats_by_dim(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize human-centered objectives by DOMAIN_GROUPS.

    The stable keys ``values``, ``mental_health``, and ``creativity`` map to
    VO, AS, and CI, respectively. The summary includes partial results from
    tasks with ``ok=False``.
    """
    dim2scores: Dict[str, List[float]] = {k: [] for k in DOMAIN_GROUPS.keys()}
    total_ok = 0
    total_with_scores = 0

    for r in records:
        if r.get("ok") is True:
            total_ok += 1
        ds = r.get("domain_scores") or {}
        if not isinstance(ds, dict):
            continue
        # Include domain scores from both ok=True and ok=False tasks
        if ds:
            total_with_scores += 1
            for dim in dim2scores.keys():
                v = ds.get(dim)
                try:
                    if v is not None:
                        dim2scores[dim].append(float(v))
                except Exception:
                    continue

    out: Dict[str, Any] = {"count": total_ok, "count_with_scores": total_with_scores, "dims": {}}
    for dim, vals in dim2scores.items():
        out["dims"][dim] = {
            "count": len(vals),
            "avg_score": round(sum(vals) / len(vals), 4) if vals else 0.0,
        }
    return out


def _compute_stats_by_metric(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-metric evaluation success statistics.

    ``total`` is the number of attempted evaluations, ``success`` is the
    number present in completed evaluations, ``fail`` is the difference, and
    ``success_rate`` is the corresponding ratio.
    """
    metric_total: Dict[str, int] = {}
    metric_success: Dict[str, int] = {}

    for r in records:
        task_type = r.get("task_type", "")
        metrics = TASK_METRICS.get(task_type, list(METRICS.keys()))
        evaluations = r.get("evaluations", [])

        # Count total attempts for this task's metrics
        for m in metrics:
            metric_total[m] = metric_total.get(m, 0) + 1

        # Count successful evaluations
        for e in evaluations:
            metric = e.get("metric")
            if metric:
                metric_success[metric] = metric_success.get(metric, 0) + 1

    # Calculate stats for each metric
    out: Dict[str, Any] = {}
    for metric in sorted(metric_total.keys()):
        total = metric_total.get(metric, 0)
        success = metric_success.get(metric, 0)
        fail = total - success
        success_rate = round(success / total, 4) if total > 0 else 0.0

        out[metric] = {
            "total": total,
            "success": success,
            "fail": fail,
            "success_rate": success_rate,
        }

    return out


async def main(args: Any) -> None:
    # 1) Read predictions from inp_file and task metadata from ref_file, then
    #    align records by ID before evaluation.
    inp_file = args.inp_file
    ref_file = args.ref_file
    out_file = args.out_file

    # Ensure that the output directory exists.
    out_dir = os.path.dirname(os.path.abspath(out_file)) if out_file else ""
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Read inp_file first and collect IDs to evaluate.
    inp_records: List[Dict[str, Any]] = []
    id_set = set()
    for idx, rec in enumerate(_read_jsonl(inp_file)):
        rid = rec.get("id")
        if rid is None:
            continue
        rid = str(rid)
        item = dict(rec)
        item["task_id"] = idx
        item["id"] = rid
        inp_records.append(item)
        id_set.add(rid)

    # Read only the reference benchmark records needed by inp_file.
    ref_map: Dict[str, Dict[str, Any]] = {}
    for rec in _read_jsonl(ref_file):
        rid = rec.get("id")
        if rid is None:
            continue
        rid = str(rid)
        if rid in id_set:
            ref_map[rid] = dict(rec)

    # Merge benchmark and prediction fields, giving precedence to predictions.
    tasks: List[Dict[str, Any]] = []
    miss_ref = 0
    for item in inp_records:
        rid = item["id"]
        ref = ref_map.get(rid)
        merged: Dict[str, Any] = {}
        if ref:
            merged.update(ref)
        else:
            miss_ref += 1

        merged.update(item)
        merged["id"] = rid
        merged["task_id"] = item["task_id"]

        # Normalize task_type, preferring the benchmark field.
        if not merged.get("task_type"):
            for k in ("Task Type", "taskType", "type", "category"):
                if merged.get(k):
                    merged["task_type"] = merged.get(k)
                    break

        # Use the CLI model name when inp_file omits it.
        if not merged.get("model_name"):
            merged["model_name"] = getattr(args, "model_name", None) or "unknown"

        # Fall back across common model-answer field names.
        if not isinstance(merged.get("model_answer"), str):
            for k in ("model_answer", "model_response", "response", "answer", "model_output"):
                v = merged.get(k)
                if isinstance(v, str):
                    merged["model_answer"] = v
                    break

        # Fall back to prediction question fields when the benchmark lacks Question.
        if not merged.get("Question"):
            vq = merged.get("model_question") or merged.get("question") or merged.get("prompt")
            if isinstance(vq, str):
                merged["Question"] = vq

        tasks.append(merged)

    if miss_ref:
        print(f"[WARNING] ref_file missing {miss_ref}/{len(tasks)} ids; these samples will likely be skipped or evaluated with incomplete context.")
    # 2) Construct the judge agent.
    agent = Agent(
        AgentConfig(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            system_prompt=SYSTEM_PROMPTS[args.prompt_language],
        )
    )

    # 3) Create semaphore for controlling concurrent API calls to judge model
    # This limits the total number of simultaneous metric evaluations across all tasks
    metric_semaphore = asyncio.Semaphore(args.num_workers)

    # 4) Send evaluation requests to the agent.
    async def _inp_func(task: Dict[str, Any], agent: Agent, prev_evals: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return await evaluate_task(task, agent, args, metric_semaphore, prev_evals)

    # 5) Write JSONL records serially.
    def _out_func(task: Dict[str, Any], completion: Dict[str, Any]) -> None:
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(completion, ensure_ascii=False) + "\n")

    def _done_func() -> Dict[str, Optional[List[Dict[str, Any]]]]:
        """
        Load partial results for checkpoint resumption as ``id -> evaluations``.

        ``None`` marks a completed task and is skipped. ``[]`` marks a failed
        task with no progress. A non-empty list preserves partial progress so
        only failed metrics are rerun.
        """
        if not os.path.exists(out_file):
            return {}

        partial_results: Dict[str, Optional[List[Dict[str, Any]]]] = {}
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if isinstance(r, dict) and "id" in r:
                    rid = r["id"]
                    if r.get("ok") is True:
                        # Fully complete: mark with None (will be skipped)
                        partial_results[rid] = None
                    else:
                        # Failed task: save partial evaluations if any
                        evaluations = r.get("evaluations", [])
                        if evaluations:
                            partial_results[rid] = evaluations  # Has partial progress
                        # else: don't add to partial_results (will be treated as new task)

        return partial_results

    # 6) Run the evaluation.
    # Load partial results for checkpointing
    partial_results = _done_func() if _done_func else {}

    async def process_single_task(task):
        """Process a single task with retry logic and partial resume support."""
        task_id = task.get("id")
        if task_id in partial_results:
            prev_evals = partial_results[task_id]
            if prev_evals is None:
                # None means task was fully completed (ok=True)
                return None
            # Non-None value: either [] (no progress) or [...] (partial progress)
            if prev_evals:
                print(f"[INFO] Resuming task {task_id} with {len(prev_evals)}/{len(TASK_METRICS.get(task.get('task_type', []), []))} metrics already evaluated")
        else:
            prev_evals = None

        for attempt in range(args.max_retries + 1):
            try:
                completion = await _inp_func(task, agent, prev_evals)
                await _maybe_await(_out_func(task, completion))
                return completion
            except asyncio.TimeoutError:
                if attempt < args.max_retries:
                    await asyncio.sleep(args.retry_delay_s)
                else:
                    # Write failure record, but preserve partial evaluations for resume
                    fail_completion = {
                        "ok": False,
                        "error_type": "TimeoutError",
                        "error": f"Task timeout after {args.task_timeout}s",
                        "attempts": attempt + 1,
                        # Preserve partial evaluations from prev_evals for resume
                        "evaluations": prev_evals if prev_evals else [],
                    }
                    await _maybe_await(_out_func(task, fail_completion))
                    return fail_completion
            except Exception as e:
                if attempt < args.max_retries:
                    await asyncio.sleep(args.retry_delay_s)
                else:
                    # Write failure record, but preserve partial evaluations for resume
                    fail_completion = {
                        "ok": False,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "attempts": attempt + 1,
                        # Preserve partial evaluations from prev_evals for resume
                        "evaluations": prev_evals if prev_evals else [],
                    }
                    await _maybe_await(_out_func(task, fail_completion))
                    return fail_completion
        return None

    # Process all tasks concurrently with limited parallelism
    # Increase num_workers significantly since we now control concurrency at metric level
    # Use a larger value here to allow all 90 tasks to start concurrently
    task_semaphore = asyncio.Semaphore(args.num_workers * 4)

    async def process_with_semaphore(task):
        async with task_semaphore:
            try:
                return await process_single_task(task)
            except asyncio.TimeoutError:
                # Write failure record for timeout, preserve partial evaluations if any
                task_id = task.get("id")
                prev_evals = partial_results.get(task_id)
                # prev_evals can be: None (completed task), list (partial progress), or missing (new task)
                if prev_evals is None:
                    prev_evals = []
                fail_completion = {
                    "ok": False,
                    "error_type": "TimeoutError",
                    "error": f"Task timeout after {args.task_timeout}s",
                    "id": task_id,
                    # Preserve partial evaluations from previous run for resume
                    "evaluations": prev_evals if prev_evals else [],
                }
                await _maybe_await(_out_func(task, fail_completion))
                return fail_completion
            except Exception as e:
                # Write failure record for other exceptions, preserve partial evaluations if any
                task_id = task.get("id")
                prev_evals = partial_results.get(task_id)
                # prev_evals can be: None (completed task), list (partial progress), or missing (new task)
                if prev_evals is None:
                    prev_evals = []
                fail_completion = {
                    "ok": False,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "id": task_id,
                    # Preserve partial evaluations from previous run for resume
                    "evaluations": prev_evals if prev_evals else [],
                }
                await _maybe_await(_out_func(task, fail_completion))
                return fail_completion

    # Collect all tasks and process them concurrently
    task_list = list(tasks)
    # Tasks not in partial_results haven't been processed at all -> run
    # Tasks in partial_results with value None are fully complete (ok=True) -> skip
    # Tasks in partial_results with non-None value have failed/partial progress -> run
    pending_tasks = [t for t in task_list if t.get("id") not in partial_results or partial_results.get(t.get("id")) is not None]

    # Create async tasks for concurrent execution
    async def run_with_progress(task):
        result = await process_with_semaphore(task)
        return result

    # Spawn all tasks concurrently
    tasks_to_run = [run_with_progress(t) for t in pending_tasks]

    # Use as_completed to track progress
    with tqdm(total=len(pending_tasks), desc="Evaluating") as pbar:
        for coro in asyncio.as_completed(tasks_to_run):
            await coro
            pbar.update(1)

    # 7) Compute the summary, retaining only the final record for each ID.
    def _iter_records():
        if not os.path.exists(out_file):
            return
        # Collect all records and retain the final record for each ID.
        records_by_id: Dict[str, Dict[str, Any]] = {}
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if isinstance(r, dict) and "id" in r:
                    rid = r["id"]
                    records_by_id[rid] = r  # Later writes supersede earlier ones.
        # Yield records in insertion order.
        for r in records_by_id.values():
            yield r

    stats = _compute_stats(_iter_records())
    cat2stats = _compute_stats_by_category(_iter_records())
    dim2stats = _compute_stats_by_dim(_iter_records())
    metric2stats = _compute_stats_by_metric(_iter_records())

    print("\n===== by task_type =====")
    print(json.dumps(cat2stats, ensure_ascii=False, indent=2))
    print("\n===== by_metric (success rates) =====")
    print(json.dumps(metric2stats, ensure_ascii=False, indent=2))
    print("\n===== summary =====")
    print(f"ok: {stats['ok']}, fail: {stats['fail']}, total: {stats['total']}")
    print(f"avg_average_score: {stats['avg_average_score']}")
    print("\n===== by_dim =====")
    print(json.dumps(dim2stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = ArgumentParser()

    # Input/output names compatible with evaluate.py and example eval.py.
    p.add_argument("--inp_file", type=str, required=True)
    p.add_argument("--ref_file", type=str, required=True)
    p.add_argument("--out_file", type=str, required=True)
    # Evaluated model name, stored in the output ``model`` field.
    p.add_argument("--model_name", type=str, default="unknown")

    # judge model endpoint
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--api_key", type=str, required=True)
    p.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    p.add_argument("--prompt-language", choices=("zh", "en"), default=os.getenv("EVAL_PROMPT_LANGUAGE", "zh"))

    # generation params
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--no_thinking", action="store_true")

    # retry controls
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--retry_delay_s", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=32,
                   help="Max concurrent metric evaluations (API calls to judge model)")
    p.add_argument("--task_timeout", type=int, default=300,
                   help="Timeout in seconds for each metric evaluation (default: 600)")

    args = p.parse_args()

    asyncio.run(main(args))
