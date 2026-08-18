# -*- coding: utf-8 -*-
"""
Generate predictions for evaluation using a user-provided model API.

Usage:
    python llm_gen.py \
        --model qwen \
        --api_key empty \
        --base_url http://127.0.0.1:8088/v1 \
        --inp_file input.jsonl \
        --out_file output.jsonl \
        --inp_tag question \
        --out_tag answer
"""

import argparse
import asyncio
import json
import os

from agent import Agent, AgentConfig, ChatConfig
from eval_math500 import PROMPT_TEMPLATES as MATH500_PROMPT_TEMPLATES
from eval_toxicn import PROMPT_TEMPLATES as TOXICN_PROMPT_TEMPLATES
from tqdm import tqdm


TASK_CONFIGS = {
    "math500": {
        "prompt_templates": MATH500_PROMPT_TEMPLATES,
        "inp_tag": "problem",
        "out_tag": "response",
        "temperature": 0.0,
    },
    "toxicn": {
        "prompt_templates": TOXICN_PROMPT_TEMPLATES,
        "inp_tag": "text",
        "out_tag": "response",
        "temperature": 0.0,
    },
}


async def call_api(agent, prompt, temperature, max_tokens):
    """Call the configured model API to generate a response."""
    chat_config = ChatConfig(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    msgs = await agent.chat(prompt, config=chat_config)
    return (msgs[-1] or {}).get("content", "") or ""


async def process_record(record, agent, semaphore, pbar, args):
    """Process single record."""
    async with semaphore:
        prompt = record.get(args.inp_tag, "")
        rid = record.get("id", "unknown")

        # Write error record if prompt is empty
        if not prompt:
            output = dict(record)
            output["ok"] = False
            output["error_type"] = "MissingInput"
            output["error"] = f"Field '{args.inp_tag}' is empty"
            pbar.update(1)
            return output

        try:
            answer = await call_api(
                agent,
                prompt,
                args.temperature,
                args.max_tokens,
            )

            output = dict(record)
            output[args.out_tag] = answer
            output["ok"] = True

            pbar.update(1)
            return output

        except asyncio.TimeoutError:
            print(f"[ERROR] Timeout id={rid} after {args.task_timeout}s")
            output = dict(record)
            output["ok"] = False
            output["error_type"] = "TimeoutError"
            output["error"] = f"Timeout after {args.task_timeout}s"
            pbar.update(1)
            return output
        except Exception as e:
            print(f"[ERROR] Failed id={rid}: {e}")
            output = dict(record)
            output["ok"] = False
            output["error_type"] = type(e).__name__
            output["error"] = str(e)
            pbar.update(1)
            return output


async def main(args):
    """Main function."""
    # Check input file exists
    if not os.path.exists(args.inp_file):
        raise FileNotFoundError(f"Input file not found: {args.inp_file}")

    # Read input file
    print(f"Reading input file: {args.inp_file}")
    records = []
    with open(args.inp_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping invalid line: {e}")
                continue

    total = len(records)
    print(f"Total records to process: {total}")

    # Create output directory
    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Create agent
    agent = Agent(
        AgentConfig(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            system_prompt=args.system_prompt,
        )
    )

    # Concurrency control
    semaphore = asyncio.Semaphore(args.num_workers)
    pbar = tqdm(total=total, desc="Generating predictions")

    async def handle_record(rec):
        try:
            return await asyncio.wait_for(
                process_record(rec, agent, semaphore, pbar, args),
                timeout=args.task_timeout
            )
        except asyncio.TimeoutError:
            # Write error record for timeout
            output = dict(rec)
            output["ok"] = False
            output["error_type"] = "TimeoutError"
            output["error"] = f"Timeout after {args.task_timeout}s"
            pbar.update(1)
            return output

    # Create all tasks
    tasks = [asyncio.create_task(handle_record(r)) for r in records]

    # Collect results and write to file
    success_count = 0
    fail_count = 0

    with open(args.out_file, "w", encoding="utf-8") as f:
        for task in asyncio.as_completed(tasks):
            result = await task
            if result:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                if result.get("ok") is True:
                    success_count += 1
                else:
                    fail_count += 1

    pbar.close()

    print(f"\nProcessing complete!")
    print(f"Success: {success_count}, Failed: {fail_count}")
    print(f"Output file: {args.out_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate predictions for evaluation using a configured model API",
    )

    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--api_key", type=str, default="empty", help="API key")
    parser.add_argument("--base_url", type=str, required=True, help="Base URL of the model API")
    parser.add_argument("--inp_file", type=str, required=True, help="Input file path (jsonl)")
    parser.add_argument("--out_file", type=str, required=True, help="Output file path (jsonl)")
    parser.add_argument("--task", choices=TASK_CONFIGS, help="Paper benchmark prompt and field configuration")
    parser.add_argument("--prompt-language", choices=("zh", "en"), default=os.getenv("EVAL_PROMPT_LANGUAGE", "zh"))
    parser.add_argument("--inp_tag", type=str, help="Input field name in record")
    parser.add_argument("--out_tag", type=str, help="Output field name in record")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of concurrent workers")
    parser.add_argument("--system_prompt", type=str, help="System prompt")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, help="Temperature")
    parser.add_argument("--task_timeout", type=int, default=120,
                       help="Timeout in seconds for each record (default: 120)")

    args = parser.parse_args()
    if args.task:
        config = TASK_CONFIGS[args.task]
        args.inp_tag = args.inp_tag or config["inp_tag"]
        args.out_tag = args.out_tag or config["out_tag"]
        args.system_prompt = args.system_prompt or config["prompt_templates"][args.prompt_language]
        if args.temperature is None:
            args.temperature = config["temperature"]
    else:
        args.inp_tag = args.inp_tag or "question"
        args.out_tag = args.out_tag or "answer"
        args.system_prompt = args.system_prompt or "You are a helpful assistant."
        args.temperature = 0.7 if args.temperature is None else args.temperature
    return args


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
