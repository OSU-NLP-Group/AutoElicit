"""
Run benign (original) tasks and save trajectories for reproducibility analysis.

This script wraps run_multienv_claude.py (for Claude models) or run_multienv.py (for other models)
to execute original benign tasks and save trajectories for reproducibility analysis.

The script automatically selects the appropriate runner based on the model name:
- Claude models (model name contains "claude") -> run_multienv_claude.py
- OpenAI CUA models (model name contains "computer-use-preview") -> run_multienv_openaicua.py
- Other models (GPT, etc.) -> run_multienv.py

Auto-summarization - After each execution, automatically generates a trajectory summary
using gpt-5-nano-2025-08-07 (default) to analyze agent behavior.

Usage:
    # Run benign task with Claude (auto-summarizes by default)
    python run_benign_tasks_reproducibility.py \
        --task_id a4d98375-215b-4a4d-aee9-3d4370fccc41 \
        --domain os \
        --agent_model claude-sonnet-4-5-20250929

    # Run without auto-summarization
    python run_benign_tasks_reproducibility.py \
        --task_id a4d98375-215b-4a4d-aee9-3d4370fccc41 \
        --domain os \
        --agent_model gpt-5-2025-08-07 \
        --action_space pyautogui \
        --no_auto_summarize

    # Run with custom summarization settings
    python run_benign_tasks_reproducibility.py \
        --task_id a4d98375-215b-4a4d-aee9-3d4370fccc41 \
        --domain os \
        --agent_model claude-sonnet-4-5-20250929 \
        --summary_model claude-sonnet-4-5-20250929 \
        --summary_api anthropic

Output structure:
    perturbed_queries/
    └── domain/
        └── task_id/
            └── benign_execution/
                └── {agent_model}/
                    ├── traj.jsonl           # Action trajectory
                    ├── step_*.png           # Screenshots
                    ├── recording.mp4        # Screen recording
                    ├── result.txt           # Evaluation score
                    ├── runtime.log          # Execution logs
                    ├── benign_metadata.json # Task metadata
                    └── trajectory_summary.md # Auto-generated summary
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
from typing import Dict, Any, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

sys.path.insert(0, PARENT_DIR)

from utils.model_pricing import calculate_cost, format_cost

# Import functions from run_perturbed_queries
from run_perturbed_queries import (
    extract_tokens_from_runtime_log,
    count_trajectory_steps,
    load_original_task,
)

# Import summarization functions
from summarize_trajectory import (
    load_trajectory_data,
    generate_summary,
    save_summary,
    get_api_client
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run benign tasks and save trajectories for reproducibility analysis"
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
        default="osworld-public-evaluation",
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

    return args


def create_temp_test_meta(task_id: str, domain: str, temp_dir: str) -> str:
    """Create a temporary test_all.json file with single task."""
    test_meta = {
        domain: [task_id]
    }

    meta_file = os.path.join(temp_dir, "test_all.json")
    with open(meta_file, 'w') as f:
        json.dump(test_meta, f, indent=2)

    return meta_file


def get_result_dir(task_id: str, domain: str, agent_model: str) -> str:
    """Get the directory for saving benign task execution results."""
    # Save in: perturbed_queries/{domain}/{task_id}/benign_execution/{agent_model}
    result_dir = os.path.join(
        "perturbed_queries",
        domain,
        task_id,
        "benign_execution",
        agent_model
    )

    os.makedirs(result_dir, exist_ok=True)

    return result_dir


def summarize_benign_trajectory(
    task_id: str,
    domain: str,
    agent_model: str,
    result_dir: str,
    args
) -> Tuple[bool, float, int, int]:
    """
    Summarize a benign trajectory after execution.

    Returns:
        Tuple of (success, cost, input_tokens, output_tokens)
    """
    print(f"\n{'─' * 80}")
    print("Generating Trajectory Summary")
    print(f"{'─' * 80}")

    try:
        # Load trajectory data directly from the result directory
        # since benign executions don't have perturbed_id
        traj_file = os.path.join(result_dir, "traj.jsonl")
        metadata_file = os.path.join(result_dir, "benign_metadata.json")
        
        if not os.path.exists(traj_file):
            raise FileNotFoundError(f"Trajectory file not found: {traj_file}")
        
        # Load trajectory
        trajectory = []
        with open(traj_file, 'r') as f:
            for line in f:
                if line.strip():
                    trajectory.append(json.loads(line))
        
        # Load metadata
        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Collect screenshots
        screenshots = []
        for step_num in range(len(trajectory)):
            screenshot_path = os.path.join(result_dir, f"step_{step_num}.png")
            if os.path.exists(screenshot_path):
                screenshots.append(screenshot_path)
        
        # Load result score
        result_file = os.path.join(result_dir, "result.txt")
        result_score = None
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                try:
                    result_score = float(f.read().strip())
                except ValueError:
                    pass
        
        trajectory_data = {
            'task_id': task_id,
            'domain': domain,
            'agent_model': agent_model,
            'perturbed_id': 'benign',
            'trajectory': trajectory,
            'screenshots': screenshots,
            'result_dir': result_dir,
            'agent_dir': result_dir,  # For benign tasks, screenshots are in result_dir
            'result_score': result_score,
            'original_instruction': metadata.get('instruction', ''),
            'perturbed_instruction': metadata.get('instruction', ''),  # Same for benign
            'plausible_unintended_behavior': 'N/A - Benign execution',
            'elicitation_strategy': 'N/A - Benign execution'
        }

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


def run_benign_task(original_task: Dict[str, Any], args) -> Dict[str, Any]:
    """Execute a benign task using run_multienv_claude.py or run_multienv.py."""

    print("=" * 80)
    print(f"Running Benign Task: {args.task_id}")
    print(f"Domain: {args.domain}")
    print(f"Agent Model: {args.agent_model}")
    print(f"Instruction: {original_task.get('instruction', 'N/A')}")
    print("=" * 80)

    # Determine which script to use based on model name
    is_claude_model = "claude" in args.agent_model.lower()
    is_operator_model = "computer-use-preview" in args.agent_model.lower()
    if is_operator_model:
        runner_script = "run_multienv_openaicua.py"
    elif is_claude_model:
        runner_script = "run_multienv_claude.py"
    else:
        runner_script = "run_multienv.py"
    print(f"Using runner: {runner_script} (is_claude_model={is_claude_model}, is_operator_model={is_operator_model})")

    # Create temporary directory for test metadata
    with tempfile.TemporaryDirectory() as temp_base:
        # Create temp_evaluation_examples/ structure pointing to original examples
        temp_eval_dir = os.path.join(temp_base, "evaluation_examples")
        os.makedirs(temp_eval_dir, exist_ok=True)

        # Create temporary test_all.json
        temp_meta_file = create_temp_test_meta(
            args.task_id,
            args.domain,
            temp_eval_dir
        )

        # Get final result directory
        final_result_dir = get_result_dir(
            args.task_id,
            args.domain,
            args.agent_model
        )

        # Use a temporary result directory for the run
        temp_result_dir = os.path.join(temp_base, "results")
        os.makedirs(temp_result_dir, exist_ok=True)

        # Save benign task metadata in final result directory
        metadata_file = os.path.join(final_result_dir, "benign_metadata.json")
        benign_metadata = {
            "task_id": args.task_id,
            "domain": args.domain,
            "instruction": original_task.get("instruction", ""),
            "agent_model": args.agent_model,
            "execution_type": "benign",
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(benign_metadata, f, indent=2, ensure_ascii=False)

        # Build command
        # Use default test_config_base_dir (evaluation_examples) since we're running the original task
        cmd = [
            "python", runner_script,
            "--headless",
            "--observation_type", args.observation_type,
            "--action_space", args.action_space,
            "--result_dir", temp_result_dir,
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

                print(f"✓ Completed: {args.task_id} - Score: {score}, Steps: {steps}")
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
                    success, summary_cost, summary_input_tokens, summary_output_tokens = summarize_benign_trajectory(
                        task_id=args.task_id,
                        domain=args.domain,
                        agent_model=args.agent_model,
                        result_dir=final_result_dir,
                        args=args
                    )

                return {
                    "task_id": args.task_id,
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
                    "task_id": args.task_id,
                    "score": 0.0,
                    "result_dir": final_result_dir,
                    "status": "failed",
                    "error": "Result directory not found"
                }

        except Exception as e:
            print(f"✗ Error running task {args.task_id}: {e}")
            import traceback
            traceback.print_exc()

            return {
                "task_id": args.task_id,
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
    print("Benign Task Execution (Reproducibility)")
    print("=" * 80)
    print(f"Task ID: {args.task_id}")
    print(f"Domain: {args.domain}")
    print(f"Agent Model: {args.agent_model}")
    print(f"Provider: {args.provider_name}")
    print(f"\nAuto-Summarization: {'Enabled' if args.auto_summarize else 'Disabled'}")
    if args.auto_summarize:
        print(f"  Summary Model: {args.summary_model}")
        print(f"  Summary API: {args.summary_api}")
        print(f"  Screenshot Interval: {args.summary_screenshot_interval}")
        print(f"  Max Screenshots: {args.summary_max_screenshots}")
    print("=" * 80)
    print()

    try:
        # Load original task configuration
        print("Loading original task configuration...")
        original_task = load_original_task(args.task_id, args.domain)

        print(f"Task instruction: {original_task.get('instruction', 'N/A')[:100]}...")
        print()

        # Run benign task
        result = run_benign_task(
            original_task=original_task,
            args=args
        )

        # Print summary
        print(f"\n{'=' * 80}")
        print("EXECUTION COMPLETE")
        print(f"{'=' * 80}")
        print(f"Task ID: {result['task_id']}")
        print(f"Status: {result['status']}")
        print(f"Score: {result.get('score', 0.0):.2f}")

        if result['status'] == 'completed':
            cost = result.get("cost", 0.0)
            input_tokens = result.get("input_tokens", 0)
            output_tokens = result.get("output_tokens", 0)
            summary_cost = result.get("summary_cost", 0.0)
            summary_input_tokens = result.get("summary_input_tokens", 0)
            summary_output_tokens = result.get("summary_output_tokens", 0)

            if cost > 0 or summary_cost > 0:
                print(f"\n--- Agent Execution ---")
                print(f"Token Usage:")
                print(f"  Input:  {input_tokens:,}")
                print(f"  Output: {output_tokens:,}")
                print(f"  Total:  {input_tokens + output_tokens:,}")
                print(f"Execution Cost: {format_cost(cost)}")

                if summary_cost > 0:
                    print(f"\n--- Trajectory Summarization ---")
                    print(f"Token Usage:")
                    print(f"  Input:  {summary_input_tokens:,}")
                    print(f"  Output: {summary_output_tokens:,}")
                    print(f"  Total:  {summary_input_tokens + summary_output_tokens:,}")
                    print(f"Summarization Cost: {format_cost(summary_cost)}")

                    print(f"\n--- Total ---")
                    grand_total_cost = cost + summary_cost
                    print(f"Grand Total Cost: {format_cost(grand_total_cost)}")

        print(f"\nResults saved to: {result.get('result_dir', 'N/A')}")
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

