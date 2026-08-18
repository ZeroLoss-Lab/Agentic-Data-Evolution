#!/usr/bin/env python3
"""Construct D(0) and DEV300 without production-only dependencies.

The script implements the paper's top-down pipeline:
structured controls -> topic scenario -> question -> answer. DEV300 uses the
same construction process and rejects questions whose BLEU-2 similarity to any
D(0) question exceeds the paper's 0.7 threshold.
"""

import argparse
import asyncio
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from nltk.translate.bleu_score import sentence_bleu
from openai import AsyncOpenAI
from tqdm import tqdm

try:
    import jieba
except ImportError:
    jieba = None


OBJECTIVE_IDS = ("values", "mental", "creativity")
OBJECTIVE_LABELS = {
    "values": "Value Orientation (VO)",
    "mental": "Affective Support (AS)",
    "creativity": "Creative Innovation (CI)",
}
VALUE_ORIENTATION_CUES = {
    "moral character": ["honesty", "responsibility", "fairness", "mutual help"],
    "rule of law": ["online speech boundaries", "privacy protection", "copyright", "evidence awareness"],
    "cultural literacy": ["respect for diversity", "public etiquette", "cultural expression", "cross-cultural communication"],
    "family values": ["family responsibilities", "intergenerational communication", "conflict negotiation", "care and support"],
    "civic identity": ["public participation", "social responsibility", "collective interests", "rational expression"],
}
AFFECTIVE_SUPPORT_CUES = [
    "exam anxiety", "physiological regulation", "breathing and mindfulness",
    "attention management", "phone distraction", "procrastination", "self-efficacy",
    "emotion identification", "resilience", "stress coping", "sleep routines",
    "social anxiety", "interpersonal boundaries", "peer pressure", "self-talk",
]
CREATIVE_INNOVATION_ACTION_CUES = ["break through", "reframe", "re-examine", "challenge", "integrate", "reason through"]
CREATIVE_INNOVATION_DOMAIN_CUES = [
    "social rule design", "nonlinear narrative", "AI ethics", "resource scarcity",
    "historical counterfactuals", "interdisciplinary project design",
]
CREATIVE_INNOVATION_GOAL_CUES = [
    "interdisciplinary transfer", "counterintuitive problem solving",
    "creative problem framing", "sensitivity to technology ethics",
]
GRADE_BANDS = ["elementary school", "middle school", "high school"]
ROLE_VIEWS = ["student", "teacher", "parent", "peer", "no_asker"]
ROLE_WEIGHTS = [0.35, 0.35, 0.10, 0.10, 0.10]
ACTIVITY_DOMAINS = [
    "classroom learning", "homework and self-study", "exams and assessment",
    "class meetings and character education", "peer interaction",
    "home-school communication", "online spaces", "clubs and projects",
]
TASK_TYPES = [
    "guided discussion and reflection", "case-based analysis", "action planning",
    "role play", "project design", "writing and expression", "peer collaboration",
    "classroom facilitation and guidance",
]
ARTIFACTS = [
    "class discussion question", "reflection log", "study strategy card",
    "class meeting plan", "communication script", "project brief",
    "peer agreement", "writing outline",
]
QUESTION_TYPES = ["decision making", "process focused", "evidence seeking", "reflective", "perspective taking"]
CONSTRAINTS = [
    "Avoid a preachy or moralizing tone.",
    "Do not name individuals and protect privacy.",
    "Keep the scenario concrete and avoid empty slogans.",
    "Ask a specific and answerable question.",
    "Use standard, natural, professional Chinese.",
]

PROMPTS = {
    "en": {
        "spec_judge": """You are an experienced curriculum and instruction expert. Review whether the supplied concept instantiation is self-consistent, usable, and implementable for a realistic K-12 topic scenario and tutoring question. The category and concept keywords must be consistent; role, activity domain, task type, and artifact must fit a school setting; and constraints must be clear and non-contradictory. Reply only KEEP or REJECT.""",
        "topic": """You are an experienced education expert. Generate one educational topic scenario from the input controls for subsequent tutoring-question generation. Ground it in the specified school context. State the situation, involved roles, central tension, and educational goal. Make it age appropriate, consistent with the requested objective and concept cues, and aligned with the intended pedagogical form and role. Respect all constraints. Do not provide solutions, advice, step-by-step procedures, or evaluative conclusions. Use standard, professional, natural Chinese by default. Output the topic scenario only.""",
        "question": """You are a senior teacher and curriculum researcher. Rewrite the input topic scenario into one instructional question for teaching or tutoring. Focus on its core tension and educational goal. Align the question with the requested role and question type. Decision Making asks for a choice and justification. Process Focused asks for key considerations and reasoning steps without providing a final answer. Evidence Seeking asks for evidence, criteria, or supporting observations. Reflective asks for reflection on goals, feelings, or learning process. Perspective Taking asks the respondent to consider another role or viewpoint. Use standard, professional, natural Chinese by default. Output the question only. Do not provide the answer, advice, or step-by-step procedures.""",
        "answer": """You are a helpful educational tutor. Answer the given question directly, accurately, and constructively. Use standard, professional, natural Chinese by default. Output only the answer.""",
    },
    "zh": {
        "spec_judge": """你是一名课程与教学专家。请审查输入的概念实例是否自洽、可用，并适合用于生成真实的 K-12 主题情境和辅导问题。category 与 concept_keywords 必须一致；角色、活动领域、任务类型和预期产出必须符合学校场景；constraints 必须清晰且不矛盾。仅回复 KEEP 或 REJECT。""",
        "topic": """你是一名资深教育专家。请根据输入控制项生成一个教育主题情境，用于后续生成辅导问题。情境应立足于指定的学校场景，清楚说明情境、涉及角色、核心张力和教育目标；应符合相应年龄特点，与目标和概念线索一致，并符合预期的教学形式和角色。请遵守所有约束。默认使用规范、专业、自然的中文。不要提供解决方案、建议、分步骤程序或评价性结论。仅输出主题情境。""",
        "question": """你是一名资深教师和课程研究者。请将输入主题情境改写为一个可直接用于教学或辅导的教学问题。聚焦主题情境中的核心张力和教育目标，并与指定角色和问题类型对齐。Decision Making 要求做出选择并说明理由。Process Focused 要求给出关键考虑因素和推理步骤，但不提供最终答案。Evidence Seeking 要求给出证据、标准或支持性观察。Reflective 要求反思目标、感受或学习过程。Perspective Taking 要求考虑另一角色或观点。默认使用规范、专业、自然的中文。仅输出问题。不要提供答案、建议或分步骤程序。""",
        "answer": """你是一名乐于助人的教育辅导者。请直接、准确且具有建设性地回答给定问题。默认使用规范、专业、自然的中文。仅输出回答。""",
    },
}


def balanced_counts(total: int) -> dict[str, int]:
    if total < len(OBJECTIVE_IDS):
        raise ValueError("size must be at least three")
    base, remainder = divmod(total, len(OBJECTIVE_IDS))
    return {
        objective_id: base + (position < remainder)
        for position, objective_id in enumerate(OBJECTIVE_IDS)
    }


def sample_spec(objective_id: str, ordinal: int, seed: int) -> dict[str, Any]:
    rng = random.Random("{}:{}:{}".format(seed, objective_id, ordinal))
    spec = {
        "category": OBJECTIVE_LABELS[objective_id],
        "objective_id": objective_id,
        "grade_band": rng.choice(GRADE_BANDS),
        "role_view": rng.choices(ROLE_VIEWS, weights=ROLE_WEIGHTS, k=1)[0],
        "activity_domain": rng.choice(ACTIVITY_DOMAINS),
        "task_type": rng.choice(TASK_TYPES),
        "artifact": rng.choice(ARTIFACTS),
        "question_type": rng.choice(QUESTION_TYPES),
        "constraints": rng.sample(CONSTRAINTS, k=rng.randint(2, 4)),
    }
    if objective_id == "values":
        dimension = rng.choice(list(VALUE_ORIENTATION_CUES))
        spec["concept_keywords"] = [dimension] + rng.sample(VALUE_ORIENTATION_CUES[dimension], k=2)
    elif objective_id == "mental":
        spec["concept_keywords"] = rng.sample(AFFECTIVE_SUPPORT_CUES, k=3)
    else:
        action = rng.choice(CREATIVE_INNOVATION_ACTION_CUES)
        domain = rng.choice(CREATIVE_INNOVATION_DOMAIN_CUES)
        goal = rng.choice(CREATIVE_INNOVATION_GOAL_CUES)
        spec.update(
            {
                "concept_keywords": [action, domain, goal],
                "creative_innovation_action_cue": action,
                "creative_innovation_domain_cue": domain,
                "creative_innovation_goal_cue": goal,
            }
        )
    return spec


def strip_fences(content: str) -> str:
    fence = chr(96) * 3
    return content.strip().replace(fence + "json", "").replace(fence, "").strip()


def parse_field(content: str, field: str) -> str:
    cleaned = strip_fences(content)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            value = json.loads(match.group(0)).get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        except json.JSONDecodeError:
            pass
    tagged = re.search(r"<{}>([\s\S]*?)</{}>".format(field, field), cleaned, re.I)
    if tagged and tagged.group(1).strip():
        return tagged.group(1).strip()
    labels = {
        "topic": ("topic", "主题", "情境", "主题情境"),
        "question": ("question", "问题", "教学问题"),
        "answer": ("answer", "回答", "答案"),
    }
    prefix = re.match(r"^\s*(?:{})\s*[:：]\s*".format("|".join(labels[field])), cleaned, re.I)
    value = cleaned[prefix.end():] if prefix else cleaned
    if not value.strip():
        raise ValueError("response lacks a non-empty '{}' field".format(field))
    return value.strip()


async def call_field(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    system: str,
    payload: Any,
    field: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    request: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": args.temperature if temperature is None else temperature,
        "top_p": 1.0,
        "max_tokens": args.max_tokens if max_tokens is None else max_tokens,
    }
    if args.disable_thinking:
        request["extra_body"] = {"thinking": {"type": "disabled"}}
    response = await client.chat.completions.create(**request)
    return parse_field(response.choices[0].message.content or "", field)


async def passes_plausibility_filter(
    client: AsyncOpenAI, args: argparse.Namespace, spec: dict[str, Any]
) -> bool:
    try:
        request: dict[str, Any] = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": PROMPTS[args.prompt_language]["spec_judge"]},
                {"role": "user", "content": json.dumps(spec, ensure_ascii=False)},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 128,
        }
        if args.disable_thinking:
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        response = await client.chat.completions.create(**request)
        cleaned = strip_fences(response.choices[0].message.content or "")
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return bool(json.loads(match.group(0)).get("keep"))
            except json.JSONDecodeError:
                pass
        normalized = cleaned.strip().upper()
        return "KEEP" in normalized and "REJECT" not in normalized
    except Exception as error:
        raise RuntimeError(
            "plausibility filter failed: {}: {}".format(type(error).__name__, error)
        ) from error


def tokenize(text: str) -> list[str]:
    if jieba is not None:
        tokens = [token.strip() for token in jieba.lcut(text) if token.strip()]
        if tokens:
            return tokens
    return [character for character in text if not character.isspace()]


class Bleu2Index:
    """Compute exact BLEU-2 only for references sharing a candidate bigram."""

    def __init__(self, questions: Iterable[str] = ()) -> None:
        self.references: list[list[str]] = []
        self.bigram_index: dict[tuple[str, str], set[int]] = defaultdict(set)
        for question in questions:
            self.add(question)

    def add(self, question: str) -> None:
        tokens = tokenize(question)
        index = len(self.references)
        self.references.append(tokens)
        for bigram in set(zip(tokens, tokens[1:])):
            self.bigram_index[bigram].add(index)

    def max_similarity(self, question: str) -> float:
        candidate = tokenize(question)
        candidate_bigrams = set(zip(candidate, candidate[1:]))
        if len(candidate) < 2 or not candidate_bigrams:
            return 0.0

        candidate_indices: set[int] = set()
        for bigram in candidate_bigrams:
            candidate_indices.update(self.bigram_index.get(bigram, set()))

        return max(
            (
                sentence_bleu([self.references[index]], candidate, weights=(0.5, 0.5))
                for index in candidate_indices
            ),
            default=0.0,
        )


async def generate_record(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    objective_id: str,
    ordinal: int,
    split: str,
    novelty_index: Bleu2Index | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    for spec_attempt in range(args.max_retries):
        spec = sample_spec(objective_id, ordinal + spec_attempt * 1000000, args.seed)
        try:
            plausible = await passes_plausibility_filter(client, args, spec)
        except Exception as error:
            errors.append("plausibility filter: {}: {}".format(type(error).__name__, error))
            continue
        if not plausible:
            errors.append("plausibility filter rejected sampled controls")
            continue

        try:
            topic = await call_field(client, args, PROMPTS[args.prompt_language]["topic"], spec, "topic")
        except Exception as error:
            errors.append("topic generation: {}: {}".format(type(error).__name__, error))
            continue

        for _ in range(args.max_retries):
            try:
                question = await call_field(
                    client,
                    args,
                    PROMPTS[args.prompt_language]["question"],
                    {
                        "topic": topic,
                        "role_view": spec["role_view"],
                        "question_type": spec["question_type"],
                    },
                    "question",
                )
            except Exception as error:
                errors.append("question generation: {}: {}".format(type(error).__name__, error))
                continue

            if novelty_index is not None and novelty_index.max_similarity(question) > args.bleu2_threshold:
                errors.append("question rejected by BLEU-2 threshold")
                continue

            try:
                answer = await call_field(
                    client,
                    args,
                    PROMPTS[args.prompt_language]["answer"],
                    question,
                    "answer",
                )
            except Exception as error:
                errors.append("answer generation: {}: {}".format(type(error).__name__, error))
                continue

            return {
                "id": "{}-{}-{:05d}".format(objective_id, split, ordinal),
                "topic": topic,
                "question": question,
                "answer": answer,
                "category": spec["category"],
                "objective_id": objective_id,
                "structured_controls": spec,
            }

    last_error = errors[-1] if errors else "no diagnostic was recorded"
    raise RuntimeError(
        "failed to generate {}-{}-{:05d} after retries; last error: {}".format(
            objective_id, split, ordinal, last_error
        )
    )


async def generate_split(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    quotas: dict[str, int],
    split: str,
    novelty_index: Bleu2Index | None = None,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(args.concurrency)

    async def worker(objective_id: str, ordinal: int) -> dict[str, Any]:
        async with semaphore:
            return await generate_record(client, args, objective_id, ordinal, split, novelty_index)

    tasks = [
        asyncio.create_task(worker(objective_id, ordinal))
        for objective_id, count in quotas.items()
        for ordinal in range(1, count + 1)
    ]
    records: list[dict[str, Any]] = []
    with tqdm(total=len(tasks), desc=split, unit="pair") as progress:
        for task in asyncio.as_completed(tasks):
            records.append(await task)
            progress.update(1)
    return sorted(records, key=lambda record: record["id"])


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def main(args: argparse.Namespace) -> None:
    train_quotas = balanced_counts(args.train_size)
    dev_quotas = balanced_counts(args.dev_size)
    if len(set(dev_quotas.values())) != 1:
        raise ValueError("--dev-size must be divisible by three")

    output_dir = Path(args.output_dir)
    d0_path = output_dir / "D0.jsonl"
    dev_path = output_dir / "DEV300.jsonl"
    manifest_path = output_dir / "construction_manifest.json"
    existing = [path for path in (d0_path, dev_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("output exists; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    try:
        d0_records = await generate_split(client, args, train_quotas, "d0")
        d0_index = Bleu2Index(record["question"] for record in d0_records)
        dev_records = await generate_split(client, args, dev_quotas, "dev300", d0_index)
    finally:
        await client.close()

    write_jsonl(d0_path, d0_records)
    write_jsonl(dev_path, dev_records)
    manifest = {
        "model": args.model,
        "seed": args.seed,
        "train_size": len(d0_records),
        "train_quotas": train_quotas,
        "dev_size": len(dev_records),
        "dev_quotas": dev_quotas,
        "objective_labels": OBJECTIVE_LABELS,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "prompt_language": args.prompt_language,
        "thinking_disabled": args.disable_thinking,
        "spec_filtering_enabled": True,
        "bleu2_threshold": args.bleu2_threshold,
        "d0_file": d0_path.name,
        "dev300_file": dev_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Wrote {} D(0) pairs to {}".format(len(d0_records), d0_path))
    print("Wrote {} DEV300 pairs to {}".format(len(dev_records), dev_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--dev-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--prompt-language", choices=("zh", "en"), default="zh")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--bleu2-threshold", type=float, default=0.7)
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("D0_DISABLE_THINKING", "true").lower() == "true",
    )
    parser.add_argument("--overwrite", action="store_true")
    asyncio.run(main(parser.parse_args()))
