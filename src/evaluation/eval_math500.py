"""Score MATH-500 predictions by exact normalized final-answer matching."""

import json
import re
from argparse import ArgumentParser
from pathlib import Path

PROMPT_TEMPLATES = {
    "zh": "你是一名乐于助人的数学助手。请逐步解决给定数学问题。将最终答案放在 \\boxed{} 中。",
    "en": "You are a helpful mathematical assistant. Solve the given mathematical problem step by step. Put your final answer in \\boxed{}.",
}


def extract_answer(text):
    matches = []
    for match in re.finditer(r"\\boxed\{", text or ""):
        start, depth, index = match.end(), 1, match.end()
        while index < len(text) and depth:
            depth += (text[index] == "{") - (text[index] == "}")
            index += 1
        if not depth:
            matches.append(text[start:index - 1].strip())
    if matches:
        return matches[-1]
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    match = re.search(r"(?:answer[:\s]+|the answer is[:\s]+)(.+)$", lines[-1], re.IGNORECASE)
    return match.group(1).strip() if match else lines[-1]


def normalize(answer):
    answer = (answer or "").strip().lower()
    answer = re.sub(r"^(answer[:\s]+|the answer is[:\s]+)", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\\frac\{(\d+)\}\{(\d+)\}", r"\1/\2", answer)
    answer = answer.replace("$", "")
    return re.sub(r"\s*([+\-*/=])\s*", r"\1", answer).strip()


def is_correct(prediction, reference):
    predicted, expected = normalize(prediction), normalize(reference)
    if not predicted or not expected:
        return False
    if predicted == expected:
        return True
    try:
        if "/" in predicted and "/" in expected:
            p_num, p_den = map(float, predicted.split("/"))
            e_num, e_den = map(float, expected.split("/"))
            return p_den != 0 and e_den != 0 and abs(p_num / p_den - e_num / e_den) < 1e-6
        return abs(float(predicted.replace(",", "")) - float(expected.replace(",", ""))) < 1e-6
    except (ValueError, ZeroDivisionError):
        return False


def main(args):
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    total = correct = 0
    with open(args.pred_file, encoding="utf-8") as source, open(args.output_file, "w", encoding="utf-8") as output:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            prediction = extract_answer(record.get("response", ""))
            reference = record.get("answer") or record.get("solution", "")
            record["predicted_answer"] = prediction
            record["gold_answer"] = reference
            record["correct"] = is_correct(prediction, reference)
            total += 1
            correct += record["correct"]
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"total": total, "correct": correct, "accuracy": correct / total if total else 0.0}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--output-file", required=True)
    main(parser.parse_args())
