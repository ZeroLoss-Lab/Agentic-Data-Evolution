"""Coordinate OVS observations, variations, and selections across snapshots."""

import asyncio
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
from tqdm import tqdm

from observation import critique, parse_score, route
from runtime import (
    ERROR_LOG_INTERVAL,
    INPUT_DIR,
    MAX_CONCURRENT_REQUESTS,
    MAX_SAMPLE_TIMEOUT_SECONDS,
    MIN_DIFF_LIGHT,
    MIN_DIFF_MAIN,
    OBJECTIVE_NAMES,
    OUTPUT_SUBDIR,
    REWRITE_MAX_ROUNDS,
    ROUND_COUNT,
)
from selection import admit, select
from variation import difference_ratio, mutate


@dataclass
class OvsResult:
    final_answer: str
    dim_type: str
    content_score_base: int
    style_score_base: int
    dim_score_base: int
    critic_ok: bool
    critic_error: str
    rewrite_rounds: int
    success: bool
    force_accepted: bool
    local_judge_reasons: List[str]
    guard_reasons: List[str]
    final_decision: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    api_calls: int
    output_tokens_est: int


def _observation_failure(
    parent_answer: str,
    objective_id: str,
    error: Exception,
    token_stats: Dict[str, int],
) -> OvsResult:
    """Preserve the parent when ROUTE or CRITIQUE evidence is unavailable."""
    return OvsResult(
        final_answer=parent_answer,
        dim_type=objective_id,
        content_score_base=0,
        style_score_base=0,
        dim_score_base=0,
        critic_ok=False,
        critic_error=str(error),
        rewrite_rounds=0,
        success=False,
        force_accepted=False,
        local_judge_reasons=[],
        guard_reasons=[],
        final_decision="observation_failed",
        prompt_tokens=token_stats["prompt"],
        completion_tokens=token_stats["completion"],
        total_tokens=token_stats["total"],
        api_calls=token_stats.get("api_calls", 0),
        output_tokens_est=len(parent_answer),
    )


async def evolve_sample(record: Dict[str, Any]) -> OvsResult:
    """Apply one OVS outer-round update to a single dataset record."""
    topic = record.get("topic", "") or ""
    question = record.get("question", "") or ""
    parent_answer = record.get("answer", "") or ""
    token_stats = {"prompt": 0, "completion": 0, "total": 0, "api_calls": 0}
    try:
        objective_id = await route(topic, question, parent_answer, token_stats)
        if objective_id == "unknown":
            raise ValueError(f"ROUTE returned no valid objective for id={record.get('id', '')}")
    except Exception as error:
        return _observation_failure(parent_answer, "unknown", error, token_stats)

    critic_ok, critic_error = True, ""
    scores = {"content": 3, "style": 3, "objective": 3}
    try:
        critiques = await critique(objective_id, topic, question, parent_answer, token_stats)
        scores["content"] = parse_score(critiques["content"].get("score_content"), "score_content")
        scores["style"] = parse_score(critiques["style"].get("score_style"), "score_style")
        scores["objective"] = parse_score(critiques["objective"].get("score_dim"), "score_dim")
    except Exception as error:
        return _observation_failure(parent_answer, objective_id, error, token_stats)

    select_reasons: List[str] = []
    admit_reasons: List[str] = []
    previous_candidate = previous_select_reason = previous_admit_reason = None
    final_answer = parent_answer
    success = False
    rounds_used = 0

    for iteration_index in range(REWRITE_MAX_ROUNDS):
        rounds_used = iteration_index + 1
        main, light = await asyncio.gather(
            mutate("main", topic, question, parent_answer, objective_id, critiques, iteration_index, previous_candidate, previous_select_reason, previous_admit_reason, token_stats),
            mutate("light", topic, question, parent_answer, objective_id, critiques, iteration_index, previous_candidate, previous_select_reason, previous_admit_reason, token_stats),
        )
        candidates: List[Tuple[str, str]] = []
        raw_candidates = [("C1", main.strip()), ("C2", light.strip())]
        if main and difference_ratio(parent_answer, main) > MIN_DIFF_MAIN:
            candidates.append(("C1", main.strip()))
        if light and difference_ratio(parent_answer, light) > MIN_DIFF_LIGHT:
            candidates.append(("C2", light.strip()))
        if not candidates:
            candidates = [(candidate_id, answer) for candidate_id, answer in raw_candidates if answer][:1]
        if not candidates:
            break
        selected_id, previous_select_reason = await select(
            objective_id, topic, question, parent_answer, candidates, token_stats
        )
        select_reasons.append(previous_select_reason)
        if selected_id == "A":
            previous_candidate, previous_admit_reason = candidates[0][1], None
            continue
        selected_answer = next((answer for candidate_id, answer in candidates if candidate_id == selected_id), candidates[0][1])
        decision, previous_admit_reason = await admit(objective_id, question, parent_answer, selected_answer, token_stats)
        admit_reasons.append(previous_admit_reason)
        if decision == "use_new":
            final_answer, success = selected_answer, True
            break
        previous_candidate = selected_answer

    return OvsResult(
        final_answer=final_answer,
        dim_type=objective_id,
        content_score_base=scores["content"],
        style_score_base=scores["style"],
        dim_score_base=scores["objective"],
        critic_ok=critic_ok,
        critic_error=critic_error,
        rewrite_rounds=rounds_used,
        success=success,
        force_accepted=False,
        local_judge_reasons=select_reasons,
        guard_reasons=admit_reasons,
        final_decision="accepted" if success else "retained",
        prompt_tokens=token_stats["prompt"],
        completion_tokens=token_stats["completion"],
        total_tokens=token_stats["total"],
        api_calls=token_stats["api_calls"],
        output_tokens_est=len(final_answer),
    )


async def evolve_file(output_file: str, input_file: str) -> None:
    """Evolve an input JSONL file once, resuming from existing output records."""
    processed_ids = set()
    if os.path.exists(output_file):
        async with aiofiles.open(output_file, "r", encoding="utf-8", errors="ignore") as handle:
            async for line in handle:
                try:
                    processed_ids.add(json.loads(line).get("id"))
                except (ValueError, TypeError):
                    continue
    records: List[Dict[str, Any]] = []
    async with aiofiles.open(input_file, "r", encoding="utf-8", errors="ignore") as handle:
        async for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("id") not in processed_ids:
                    records.append(record)
    if not records:
        print("No new samples to process.")
        return

    lock, semaphore = asyncio.Lock(), asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    with tqdm(total=len(records), desc="OVS iterative revision") as progress:
        async def handle(record: Dict[str, Any]) -> None:
            async with semaphore:
                try:
                    result = await asyncio.wait_for(evolve_sample(record), MAX_SAMPLE_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    print(f"[TIMEOUT] id={record.get('id')} exceeded {MAX_SAMPLE_TIMEOUT_SECONDS}s")
                except Exception as error:
                    print(f"[ERROR] id={record.get('id')} failed: {type(error).__name__}: {error}")
                else:
                    output = dict(record)
                    output["old_answer"] = record.get("answer", "")
                    output["answer"] = result.final_answer
                    output["ovs_meta"] = {
                        "objective_id": result.dim_type,
                        "primary_objective": OBJECTIVE_NAMES.get(result.dim_type, "Unknown"),
                        "dim_type": result.dim_type,
                        "content_score_base": result.content_score_base,
                        "style_score_base": result.style_score_base,
                        "dim_score_base": result.dim_score_base,
                        "critic_ok": result.critic_ok,
                        "critic_error": result.critic_error,
                        "rewrite_rounds": result.rewrite_rounds,
                        "success": result.success,
                        "force_accepted": result.force_accepted,
                        "local_judge_reasons": result.local_judge_reasons,
                        "guard_reasons": result.guard_reasons,
                        "final_decision": result.final_decision,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "total_tokens": result.total_tokens,
                        "api_calls": result.api_calls,
                        "output_tokens_est": result.output_tokens_est,
                    }
                    async with lock:
                        async with aiofiles.open(output_file, "a", encoding="utf-8") as handle:
                            await handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                progress.update(1)
        await asyncio.gather(*(handle(record) for record in records))


def _line_count(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def run() -> None:
    """Run OVS snapshot evolution for all input JSONL files."""
    input_files = [
        path for path in glob.glob(os.path.join(INPUT_DIR, "**", "*.jsonl"), recursive=True)
        if "output" not in os.path.relpath(path, INPUT_DIR)
    ]
    if not input_files:
        raise SystemExit(f"No JSONL files found under {INPUT_DIR}")
    for input_file in input_files:
        expected_rows = _line_count(input_file)
        current_input = input_file
        stem = os.path.splitext(os.path.basename(input_file))[0]
        for round_index in range(1, ROUND_COUNT + 1):
            output_dir = os.path.join(OUTPUT_SUBDIR, f"round{round_index}", "data")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{stem}.jsonl")
            if os.path.exists(output_file) and _line_count(output_file) >= expected_rows:
                current_input = output_file
                continue
            print(f"Starting OVS iteration {round_index}/{ROUND_COUNT}: {input_file}")
            asyncio.run(evolve_file(output_file, current_input))
            current_input = output_file
