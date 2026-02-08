"""
Run perturbed instructions and save trajectories for safety evaluation.

This script wraps run_multienv_claude.py (for Claude models) or run_multienv.py (for other models)
to execute perturbed queries and save trajectories in the perturbation directory structure.

The script automatically selects the appropriate runner based on the model name:
- Claude models (model name contains "claude") -> run_multienv_claude.py
- Other models (GPT, etc.) -> run_multienv.py

NEW: Auto-summarization - After each execution, automatically generates a trajectory summary
using gpt-5-nano-2025-08-07 (default) to analyze agent behavior and safety outcomes.

Usage:
    # Run single perturbed query with Claude (auto-summarizes by default)
    python run_perturbed_queries.py \
        --task_id <TASK_UUID> \
        --domain os \
        --perturbed_id <PERTURBED_ID> \
        --agent_model <AGENT_MODEL>

    # Run without auto-summarization
    python run_perturbed_queries.py \
        --task_id <TASK_UUID> \
        --domain os \
        --perturbed_id <PERTURBED_ID> \
        --agent_model <AGENT_MODEL> \
        --action_space pyautogui \
        --no_auto_summarize

    # Run with custom summarization settings
    python run_perturbed_queries.py \
        --task_id <TASK_UUID> \
        --domain os \
        --all \
        --agent_model <AGENT_MODEL> \
        --summary_model <SUMMARY_MODEL> \
        --summary_api anthropic \
        --summary_screenshot_interval 2 \
        --summary_max_screenshots 15

Output structure:
    perturbed_queries/
    └── domain/
        └── task_id/
            └── {generation_model}/              # Model that generated the perturbation
                └── perturbed_query_{id}/
                    └── {agent_model}/           # Model that executed it
                        ├── traj.jsonl           # Action trajectory
                        ├── step_*.png           # Screenshots
                        ├── recording.mp4        # Screen recording
                        ├── result.txt           # Evaluation score
                        ├── runtime.log          # Execution logs
                        └── trajectory_summary.md # Auto-generated summary (NEW)
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
import shutil
import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

sys.path.insert(0, REPO_ROOT)

from utils.model_pricing import calculate_cost, format_cost

# Import summarization functions
from summarize_trajectory import (
    load_trajectory_data,
    generate_summary,
    save_summary,
    get_api_client
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run perturbed queries and save trajectories"
    )

    # Task selection
    parser.add_argument(
        "--task_id",
        type=str,
        required=True,
        help="Task ID (e.g., a4d98375-215b-4a4d-aee9-3d4370fccc41)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain (e.g., os, chrome, vs_code)"
    )
    parser.add_argument(
        "--perturbed_id",
        type=str,
        help="Specific perturbed query ID to run (e.g., 38ea3a0e)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all perturbed queries for the task"
    )

    # Agent configuration
    parser.add_argument(
        "--agent_model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Agent model to use (e.g., claude-sonnet-4-5-20250929, gpt-5-2025-08-07)"
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="pyautogui",
        help="Action space (use 'claude_computer_use' for Claude models, 'pyautogui' for others)"
    )
    parser.add_argument(
        "--observation_type",
        type=str,
        default="screenshot",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        help="Observation type"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum steps per episode"
    )

    # Environment configuration
    parser.add_argument(
        "--provider_name",
        type=str,
        default="aws",
        help="VM provider"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--client_password",
        type=str,
        default="your-client-password",
        help="Client password"
    )

    # Trajectory summarization options
    parser.add_argument(
        "--no_auto_summarize",
        action="store_true",
        help="Disable automatic trajectory summarization (enabled by default)"
    )
    parser.add_argument(
        "--summary_model",
        type=str,
        default="gpt-5-nano-2025-08-07",
        help="Model to use for trajectory summarization (default: gpt-5-nano-2025-08-07)"
    )
    parser.add_argument(
        "--summary_api",
        type=str,
        default="openai",
        choices=["openai", "azure", "anthropic", "anthropic_bedrock"],
        help="API to use for summarization (default: openai)"
    )
    parser.add_argument(
        "--summary_screenshot_interval",
        type=int,
        default=1,
        help="Screenshot interval for summarization (default: 1 = all screenshots)"
    )
    parser.add_argument(
        "--summary_max_screenshots",
        type=int,
        default=50,
        help="Maximum screenshots for summarization (default: 50, auto-adjusted based on 50MB payload limit)"
    )

    args = parser.parse_args()

    # Auto-summarization is enabled by default, unless --no_auto_summarize is specified
    args.auto_summarize = not args.no_auto_summarize

    # Validation
    if not args.all and not args.perturbed_id:
        parser.error("Must specify either --perturbed_id or --all")

    return args


def load_filtered_perturbed_queries(task_id: str, domain: str) -> List[Dict[str, Any]]:
    """Load filtered perturbed queries for a task."""
    queries_file = os.path.join(PARENT_DIR, "perturbed_queries", domain, task_id, "filtered_perturbed_queries.json")

    if not os.path.exists(queries_file):
        raise FileNotFoundError(f"Filtered queries not found: {queries_file}")

    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    return queries


def load_original_task(task_id: str, domain: str) -> Dict[str, Any]:
    """Load the original task configuration."""
    task_file = os.path.join(REPO_ROOT, "evaluation_examples", "examples", domain, f"{task_id}.json")

    if not os.path.exists(task_file):
        raise FileNotFoundError(f"Task file not found: {task_file}")

    with open(task_file, 'r', encoding='utf-8') as f:
        task = json.load(f)

    return task


def create_perturbed_task_file(original_task: Dict[str, Any],
                                perturbed_query: Dict[str, Any],
                                temp_examples_dir: str) -> str:
    """Create a task file with perturbed instruction in temporary directory."""
    # Copy original task
    perturbed_task = original_task.copy()

    # Replace instruction with perturbed version
    perturbed_task["instruction"] = perturbed_query["perturbed_instruction"]

    # Add metadata about the perturbation
    perturbed_task["perturbation_metadata"] = {
        "perturbed_id": perturbed_query["id"],
        "original_instruction": perturbed_query["original_instruction"],
        "elicitation_strategy": perturbed_query["elicitation_strategy"],
        "plausible_unintended_behavior": perturbed_query["plausible_unintended_behavior"],
        "reasoning": perturbed_query.get("reasoning", ""),
        "generation_model": perturbed_query["generation_metadata"]["model"],
        "average_scores": perturbed_query.get("average_scores", {}),
    }

    # Write to temporary examples directory
    task_file = os.path.join(temp_examples_dir, f"{original_task['id']}.json")
    with open(task_file, 'w', encoding='utf-8') as f:
        json.dump(perturbed_task, f, indent=2, ensure_ascii=False)

    return task_file


def create_temp_test_meta(task_id: str, domain: str, temp_dir: str) -> str:
    """Create a temporary test_all.json file with single task."""
    test_meta = {
        domain: [task_id]
    }

    meta_file = os.path.join(temp_dir, "test_all.json")
    with open(meta_file, 'w') as f:
        json.dump(test_meta, f, indent=2)

    return meta_file


def get_result_dir(perturbed_query: Dict[str, Any],
                   task_id: str,
                   domain: str,
                   agent_model: str,
                   perturbed_queries_dir: str = os.path.join(PARENT_DIR, "perturbed_queries")) -> str:
    """Get the directory for saving perturbed query execution results."""
    # Save in: {perturbed_queries_dir}/{domain}/{task_id}/{generation_model}/perturbed_query_{id}/{agent_model}/
    generation_model = perturbed_query["generation_metadata"]["model"]
    perturbed_id = perturbed_query["id"]

    result_dir = os.path.join(
        perturbed_queries_dir,
        domain,
        task_id,
        generation_model,
        f"perturbed_query_{perturbed_id}",
        agent_model
    )

    os.makedirs(result_dir, exist_ok=True)

    return result_dir


def extract_tokens_from_runtime_log(result_dir: str) -> tuple[int, int]:
    """
    Extract total input and output tokens from runtime.log.

    The Anthropic API logs usage information like:
    Usage(input_tokens=1234, output_tokens=567, cache_creation_input_tokens=0, cache_read_input_tokens=0, ...)

    Note: We exclude cache_creation_input_tokens and cache_read_input_tokens from the count
    as those are separate cache-related fields, not actual model input tokens.

    Returns:
        Tuple of (total_input_tokens, total_output_tokens)
    """
    runtime_log = os.path.join(result_dir, "runtime.log")

    if not os.path.exists(runtime_log):
        return 0, 0

    total_input = 0
    total_output = 0

    try:
        import re
        with open(runtime_log, 'r') as f:
            for line in f:
                # Look for Usage() patterns in logs
                # Example: Usage(input_tokens=1234, output_tokens=567, ...)
                # Use negative lookbehind to exclude cache_creation_input_tokens and cache_read_input_tokens
                match = re.search(r'(?<!cache_creation_)(?<!cache_read_)input_tokens=(\d+)', line)
                if match:
                    total_input += int(match.group(1))

                match = re.search(r'output_tokens=(\d+)', line)
                if match:
                    total_output += int(match.group(1))
    except Exception as e:
        print(f"  Warning: Could not extract tokens from runtime log: {e}")

    return total_input, total_output


def count_trajectory_steps(result_dir: str) -> int:
    """Count the number of steps in the trajectory file."""
    traj_file = os.path.join(result_dir, "traj.jsonl")

    if not os.path.exists(traj_file):
        return 0

    try:
        with open(traj_file, 'r') as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def summarize_trajectory(
    task_id: str,
    domain: str,
    perturbed_id: str,
    agent_model: str,
    args
) -> Tuple[bool, float, int, int]:
    """
    Summarize a trajectory after execution.

    Returns:
        Tuple of (success, cost, input_tokens, output_tokens)
    """
    print(f"\n{'─' * 80}")
    print("Generating Trajectory Summary")
    print(f"{'─' * 80}")

    try:
        # Get perturbed_queries_dir from args
        perturbed_queries_dir = getattr(args, 'perturbed_queries_dir', os.path.join(PARENT_DIR, "perturbed_queries"))

        # Load trajectory data
        trajectory_data = load_trajectory_data(
            task_id=task_id,
            domain=domain,
            perturbed_id=perturbed_id,
            agent_model=agent_model,
            perturbed_queries_dir=perturbed_queries_dir
        )

        total_steps = len(trajectory_data['trajectory'])
        print(f"  Trajectory: {total_steps} steps")
        print(f"  Summary model: {args.summary_model}")
        print(f"  Screenshot interval: every {args.summary_screenshot_interval} step(s)")
        print(f"  Max screenshots: {args.summary_max_screenshots}")

        # Get API client
        api_client = get_api_client(args.summary_api)

        # Generate summary
        summary, input_tokens, output_tokens, total_tokens = generate_summary(
            api_client=api_client,
            api_type=args.summary_api,
            trajectory_data=trajectory_data,
            model=args.summary_model,
            screenshot_interval=args.summary_screenshot_interval,
            max_screenshots=args.summary_max_screenshots,
            max_tokens=32768,
            temperature=1
        )

        # Calculate cost
        cost = calculate_cost(args.summary_model, input_tokens, output_tokens)

        # Save summary
        output_path = save_summary(
            trajectory_data=trajectory_data,
            summary=summary,
            model=args.summary_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost
        )

        print(f"  ✓ Summary generated:")
        print(f"    Input tokens:  {input_tokens:,}")
        print(f"    Output tokens: {output_tokens:,}")
        print(f"    Cost: {format_cost(cost)}")
        print(f"    Saved to: {output_path}")

        return True, cost, input_tokens, output_tokens

    except FileNotFoundError as e:
        print(f"  ✗ Cannot summarize: {e}")
        return False, 0.0, 0, 0
    except Exception as e:
        print(f"  ✗ Summarization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0.0, 0, 0


def run_perturbed_query(perturbed_query: Dict[str, Any],
                        original_task: Dict[str, Any],
                        args) -> Dict[str, Any]:
    """Execute a single perturbed query using run_multienv_claude.py or run_multienv.py."""

    print("=" * 80)
    print(f"Running Perturbed Query: {perturbed_query['id']}")
    print(f"Generation Model: {perturbed_query['generation_metadata']['model']}")
    print(f"Agent Model: {args.agent_model}")
    print(f"Original: {perturbed_query['original_instruction']}")
    print(f"Perturbed: {perturbed_query['perturbed_instruction']}")
    print(f"Strategy: {perturbed_query['elicitation_strategy']}")
    print(f"Plausible Harm: {perturbed_query['plausible_unintended_behavior'][:100]}...")
    print("=" * 80)

    # Determine which script to use based on model name
    is_claude_model = "claude" in args.agent_model.lower()
    is_openai_cua_model = args.agent_model.lower() == "computer-use-preview"

    if is_claude_model:
        runner_script = os.path.join(PARENT_DIR, "run_multienv_claude.py")

    elif is_openai_cua_model:
        runner_script = os.path.join(PARENT_DIR, "run_multienv_openaicua.py")
    else:
        runner_script = os.path.join(PARENT_DIR, "run_multienv.py")

    print(f"Using runner: {runner_script} (is_claude={is_claude_model}, is_openai_cua={is_openai_cua_model})")

    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_base:
        # Create temp_evaluation_examples/examples/{domain}/ structure
        temp_examples_dir = os.path.join(temp_base, "evaluation_examples", "examples", args.domain)
        os.makedirs(temp_examples_dir, exist_ok=True)

        # Create temporary task config with perturbed instruction
        temp_task_file = create_perturbed_task_file(
            original_task,
            perturbed_query,
            temp_examples_dir
        )

        # Create temporary test_all.json in temp_evaluation_examples/
        temp_meta_file = create_temp_test_meta(
            args.task_id,
            args.domain,
            os.path.join(temp_base, "evaluation_examples")
        )

        # Get final result directory
        perturbed_queries_dir = getattr(args, 'perturbed_queries_dir', os.path.join(PARENT_DIR, "perturbed_queries"))
        final_result_dir = get_result_dir(
            perturbed_query,
            args.task_id,
            args.domain,
            args.agent_model,
            perturbed_queries_dir
        )

        # Use a temporary result directory for the run
        temp_result_dir = os.path.join(temp_base, "results")
        os.makedirs(temp_result_dir, exist_ok=True)

        # Save perturbed query metadata in final result directory
        metadata_file = os.path.join(final_result_dir, "perturbed_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(perturbed_query, f, indent=2, ensure_ascii=False)

        # Build command based on which runner to use
        # CRITICAL: Must pass --test_config_base_dir to point to temp directory containing perturbed task file
        # Otherwise runner will use default path and load ORIGINAL instruction instead of perturbed one
        cmd = [
            "python", runner_script,
            "--headless",
            "--observation_type", args.observation_type,
            "--action_space", args.action_space,
            "--result_dir", temp_result_dir,
            "--test_config_base_dir", os.path.join(temp_base, "evaluation_examples"),
            "--test_all_meta_path", temp_meta_file,
            "--max_steps", str(args.max_steps),
            "--num_envs", str(args.num_envs),
            "--provider_name", args.provider_name,
            "--client_password", args.client_password,
            "--region", args.region,
            "--domain", args.domain,
            "--model", args.agent_model,
        ]

        print(f"Final Result Dir: {final_result_dir}")
        print(f"Temp Result Dir: {temp_result_dir}")
        print(f"Command: {' '.join(cmd)}")
        print()

        # Execute command from parent directory
        try:
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

            result = subprocess.run(
                cmd,
                cwd=parent_dir,
                env={**os.environ, "PYTHONPATH": parent_dir},
                capture_output=False,
                text=True
            )

            # Find the actual result files
            # The runner creates: temp_result_dir/action_space/observation_type/model/domain/task_id/*
            nested_result_dir = os.path.join(
                temp_result_dir,
                args.action_space,
                args.observation_type,
                args.agent_model,
                args.domain,
                args.task_id
            )

            score = 0.0
            if os.path.exists(nested_result_dir):
                # Read result
                result_file = os.path.join(nested_result_dir, "result.txt")
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        score = float(f.read().strip())

                # Move all files from nested directory to final result directory
                for item in os.listdir(nested_result_dir):
                    src = os.path.join(nested_result_dir, item)
                    dst = os.path.join(final_result_dir, item)
                    if os.path.exists(dst):
                        if os.path.isdir(dst):
                            shutil.rmtree(dst)
                        else:
                            os.remove(dst)
                    shutil.move(src, dst)

                # Calculate cost from runtime logs
                input_tokens, output_tokens = extract_tokens_from_runtime_log(final_result_dir)
                steps = count_trajectory_steps(final_result_dir)

                # Calculate cost if we have token counts
                if input_tokens > 0 or output_tokens > 0:
                    cost = calculate_cost(args.agent_model, input_tokens, output_tokens)
                    has_token_data = True
                else:
                    cost = 0.0
                    has_token_data = False

                # Save cost information
                cost_file = os.path.join(final_result_dir, "cost.json")
                cost_data = {
                    "agent_model": args.agent_model,
                    "steps": steps,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost_usd": cost,
                    "has_token_data": has_token_data
                }
                with open(cost_file, 'w') as f:
                    json.dump(cost_data, f, indent=2)

                print(f"✓ Completed: {perturbed_query['id']} - Score: {score}, Steps: {steps}")
                if has_token_data:
                    print(f"  Tokens: {input_tokens:,} in / {output_tokens:,} out / {input_tokens + output_tokens:,} total")
                    print(f"  Cost: {format_cost(cost)}")
                else:
                    print(f"  Token usage not available in logs")
                print(f"  Results saved to: {final_result_dir}")

                # Auto-summarize trajectory if enabled
                summary_cost = 0.0
                summary_input_tokens = 0
                summary_output_tokens = 0
                if args.auto_summarize:
                    success, summary_cost, summary_input_tokens, summary_output_tokens = summarize_trajectory(
                        task_id=args.task_id,
                        domain=args.domain,
                        perturbed_id=perturbed_query["id"],
                        agent_model=args.agent_model,
                        args=args
                    )

                return {
                    "perturbed_id": perturbed_query["id"],
                    "score": score,
                    "result_dir": final_result_dir,
                    "status": "completed",
                    "return_code": result.returncode,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost,
                    "summary_cost": summary_cost,
                    "summary_input_tokens": summary_input_tokens,
                    "summary_output_tokens": summary_output_tokens
                }
            else:
                print(f"✗ Warning: Result directory not found: {nested_result_dir}")
                return {
                    "perturbed_id": perturbed_query["id"],
                    "score": 0.0,
                    "result_dir": final_result_dir,
                    "status": "failed",
                    "error": "Result directory not found"
                }

        except Exception as e:
            print(f"✗ Error running {perturbed_query['id']}: {e}")
            import traceback
            traceback.print_exc()

            return {
                "perturbed_id": perturbed_query["id"],
                "score": 0.0,
                "result_dir": final_result_dir,
                "status": "failed",
                "error": str(e)
            }


def main():
    args = parse_args()

    # Change to perturbation_generation directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 80)
    print("Perturbed Query Execution")
    print("=" * 80)
    print(f"Task ID: {args.task_id}")
    print(f"Domain: {args.domain}")
    print(f"Agent Model: {args.agent_model}")
    print(f"Provider: {args.provider_name}")
    if args.perturbed_id:
        print(f"Perturbed ID: {args.perturbed_id}")
    else:
        print("Mode: Run all perturbed queries")
    print(f"\nAuto-Summarization: {'Enabled' if args.auto_summarize else 'Disabled'}")
    if args.auto_summarize:
        print(f"  Summary Model: {args.summary_model}")
        print(f"  Summary API: {args.summary_api}")
        print(f"  Screenshot Interval: {args.summary_screenshot_interval}")
        print(f"  Max Screenshots: {args.summary_max_screenshots}")
    print("=" * 80)
    print()

    try:
        # Load data
        print("Loading filtered perturbed queries...")
        perturbed_queries = load_filtered_perturbed_queries(args.task_id, args.domain)

        print("Loading original task configuration...")
        original_task = load_original_task(args.task_id, args.domain)

        # Filter queries if specific ID requested
        if args.perturbed_id:
            perturbed_queries = [
                q for q in perturbed_queries if q["id"] == args.perturbed_id
            ]
            if not perturbed_queries:
                print(f"ERROR: Perturbed query {args.perturbed_id} not found")
                sys.exit(1)

        print(f"Found {len(perturbed_queries)} perturbed queries to run\n")

        # Run each perturbed query
        results = []
        for i, perturbed_query in enumerate(perturbed_queries, 1):
            print(f"\n{'=' * 80}")
            print(f"Query {i}/{len(perturbed_queries)}")
            print(f"{'=' * 80}\n")

            result = run_perturbed_query(
                perturbed_query=perturbed_query,
                original_task=original_task,
                args=args
            )
            results.append(result)
            print()

        # Print simple summary
        print(f"\n{'=' * 80}")
        print("EXECUTION COMPLETE")
        print(f"{'=' * 80}")
        print(f"Total Queries: {len(results)}")
        print(f"Completed: {sum(1 for r in results if r['status'] == 'completed')}")
        print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
        if results:
            avg_score = sum(r["score"] for r in results) / len(results)
            print(f"Average Score: {avg_score:.2f}")

            # Calculate total cost (agent execution)
            total_cost = sum(r.get("cost", 0.0) for r in results)
            total_input_tokens = sum(r.get("input_tokens", 0) for r in results)
            total_output_tokens = sum(r.get("output_tokens", 0) for r in results)

            # Calculate summarization cost
            total_summary_cost = sum(r.get("summary_cost", 0.0) for r in results)
            total_summary_input_tokens = sum(r.get("summary_input_tokens", 0) for r in results)
            total_summary_output_tokens = sum(r.get("summary_output_tokens", 0) for r in results)

            if total_cost > 0 or total_summary_cost > 0:
                print(f"\n--- Agent Execution ---")
                print(f"Token Usage:")
                print(f"  Input:  {total_input_tokens:,}")
                print(f"  Output: {total_output_tokens:,}")
                print(f"  Total:  {total_input_tokens + total_output_tokens:,}")
                print(f"Execution Cost: {format_cost(total_cost)}")

                if total_summary_cost > 0:
                    print(f"\n--- Trajectory Summarization ---")
                    print(f"Token Usage:")
                    print(f"  Input:  {total_summary_input_tokens:,}")
                    print(f"  Output: {total_summary_output_tokens:,}")
                    print(f"  Total:  {total_summary_input_tokens + total_summary_output_tokens:,}")
                    print(f"Summarization Cost: {format_cost(total_summary_cost)}")

                    print(f"\n--- Total ---")
                    grand_total_cost = total_cost + total_summary_cost
                    print(f"Grand Total Cost: {format_cost(grand_total_cost)}")

                    if len(results) > 1:
                        avg_exec_cost = total_cost / len(results)
                        avg_summary_cost = total_summary_cost / len(results)
                        avg_total_cost = grand_total_cost / len(results)
                        print(f"\nPer Query Average:")
                        print(f"  Execution: {format_cost(avg_exec_cost)}")
                        print(f"  Summary:   {format_cost(avg_summary_cost)}")
                        print(f"  Total:     {format_cost(avg_total_cost)}")
                elif len(results) > 1:
                    avg_cost = total_cost / len(results)
                    print(f"Average Cost per Query: {format_cost(avg_cost)}")
        print(f"{'=' * 80}\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
