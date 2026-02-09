"""
Run AI agents on the robustness evaluation benchmark with multiple runs per example.

This script executes agents on the curated set of successful elicitation
instructions with multiple runs per example for reproducibility analysis.

Usage:
    # Run a single agent with 3 runs per example
    python run_benchmark_multiple_runs.py \
        --agent_model claude-sonnet-4-5-20250929 \
        --provider_name aws \
        --num_runs 3

    # Run specific domain with 5 runs each
    python run_benchmark_multiple_runs.py \
        --agent_model gpt-5-2025-08-07 \
        --domain os \
        --provider_name aws \
        --num_runs 5

    # Run with parallel environments
    python run_benchmark_multiple_runs.py \
        --agent_model claude-4-sonnet-20250514 \
        --provider_name aws \
        --num_envs 5 \
        --num_runs 3

Output structure:
    robustness_evaluation/results/
    └── {agent_model}/
        ├── execution_summary.json           # Overall statistics
        └── {example_id}/                    # Per-example results
            ├── run_0/                       # First run
            │   ├── execution_metadata.json
            │   ├── traj.jsonl
            │   ├── step_*.png
            │   ├── recording.mp4
            │   ├── result.txt
            │   ├── runtime.log
            │   └── cost.json
            ├── run_1/                       # Second run
            │   └── ...
            ├── run_2/                       # Third run
            │   └── ...
            └── aggregated_results.json      # Statistics across all runs
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
from collections import defaultdict

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)

from utils.model_pricing import calculate_cost, format_cost


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run agents on robustness evaluation benchmark with multiple runs"
    )

    # Agent configuration
    parser.add_argument(
        "--agent_model",
        type=str,
        required=True,
        help="Agent model to test (e.g., claude-sonnet-4-5-20250929, gpt-5-2025-08-07)"
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="pyautogui",
        help="Action space (default: pyautogui)"
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

    # Multiple runs configuration
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of runs per example (default: 3)"
    )
    parser.add_argument(
        "--start_run",
        type=int,
        default=0,
        help="Starting run index (default: 0)"
    )

    # Benchmark selection
    parser.add_argument(
        "--domain",
        type=str,
        choices=["os", "multi_apps", "multi_apps_test", "all"],
        default="all",
        help="Filter by domain (default: all). Use 'multi_apps' or 'multi_apps_test' for multi-app tasks."
    )
    parser.add_argument(
        "--severity",
        type=str,
        choices=["minimal", "low", "medium", "high", "critical", "all"],
        default="all",
        help="Filter by severity (default: all)"
    )
    parser.add_argument(
        "--example_id",
        type=str,
        help="Run specific example ID only"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of examples to run (for testing)"
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

    # Output configuration
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory (default: ./results/)"
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        default=True,
        help="Force re-execution of all runs even if results exist (default: skip existing)"
    )

    return parser.parse_args()


def load_benchmark_examples(
    benchmark_dir: str = "robustness_benchmark",
    domain_filter: str = "all",
    severity_filter: str = "all",
    example_id_filter: str = None
) -> List[Dict[str, Any]]:
    """Load benchmark examples from the robustness_benchmark directory."""
    examples = []

    # Normalize domain filter (both multi_apps and multi_apps_test refer to same benchmark domain)
    if domain_filter == "multi_apps":
        domain_filter = "multi_apps_test"

    # Read benchmark metadata
    metadata_file = os.path.join(benchmark_dir, "benchmark_metadata.json")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Benchmark metadata not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        benchmark_meta = json.load(f)

    # Iterate through each agent directory
    for agent_name in benchmark_meta["by_agent"].keys():
        agent_dir = os.path.join(benchmark_dir, agent_name)
        agent_meta_file = os.path.join(agent_dir, "agent_metadata.json")

        if not os.path.exists(agent_meta_file):
            continue

        with open(agent_meta_file, 'r') as f:
            agent_meta = json.load(f)

        # Load each example
        for example_meta in agent_meta["examples"]:
            example_id = example_meta["example_id"]
            domain = example_meta["domain"]
            severity = example_meta["severity"]

            # Apply filters
            if example_id_filter and example_id != example_id_filter:
                continue
            if domain_filter != "all" and domain != domain_filter:
                continue
            if severity_filter != "all" and severity != severity_filter:
                continue

            # Load the actual example data
            # Directory name uses underscores instead of colons
            example_dir_name = example_id.replace(':', '_')
            example_dir = os.path.join(agent_dir, example_dir_name)
            query_file = os.path.join(example_dir, "final_successful_query.json")

            if not os.path.exists(query_file):
                print(f"Warning: Query file not found for {example_id}")
                continue

            with open(query_file, 'r') as f:
                query_data = json.load(f)

            # Parse example_id to get task_id
            # Format: task_id:perturbed_id:generation_model:execution_agent
            parts = example_id.split(':')
            task_id = parts[0]

            # Combine all data
            example = {
                "example_id": example_id,
                "task_id": task_id,
                "domain": domain,
                "severity": severity,
                "perturbed_instruction": query_data["perturbed_instruction"],
                "elicitation_strategy": query_data["elicitation_strategy"],
                "behavior_elicitation_score": query_data["behavior_elicitation_score"],
            }

            examples.append(example)

    return examples


def load_original_task(task_id: str, domain: str) -> Dict[str, Any]:
    """Load the original task configuration."""
    # Map benchmark domain names to actual directory names
    domain_mapping = {
        "multi_apps_test": "multi_apps",
        "os": "os"
    }

    actual_domain = domain_mapping.get(domain, domain)
    task_file = os.path.join("../evaluation_examples/examples", actual_domain, f"{task_id}.json")

    if not os.path.exists(task_file):
        raise FileNotFoundError(f"Task file not found: {task_file}")

    with open(task_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_temp_task_file(
    original_task: Dict[str, Any],
    example: Dict[str, Any],
    temp_examples_dir: str
) -> str:
    """Create a task file with perturbed instruction."""
    perturbed_task = original_task.copy()
    perturbed_task["instruction"] = example["perturbed_instruction"]

    # Add metadata
    perturbed_task["robustness_benchmark_metadata"] = {
        "example_id": example["example_id"],
        "elicitation_strategy": example["elicitation_strategy"],
        "behavior_elicitation_score": example["behavior_elicitation_score"],
        "severity": example["severity"]
    }

    task_file = os.path.join(temp_examples_dir, f"{original_task['id']}.json")
    with open(task_file, 'w', encoding='utf-8') as f:
        json.dump(perturbed_task, f, indent=2, ensure_ascii=False)

    return task_file


def run_single_execution(
    example: Dict[str, Any],
    args,
    original_task: Dict[str, Any],
    run_index: int,
    result_dir: str
) -> Dict[str, Any]:
    """Execute a single run for a benchmark example."""

    run_dir = os.path.join(result_dir, f"run_{run_index}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"  Run {run_index}: Starting execution...")

    # Map domain names
    domain_mapping = {
        "multi_apps_test": "multi_apps",
        "os": "os"
    }
    actual_domain = domain_mapping.get(example["domain"], example["domain"])

    # Determine runner script
    is_claude = "claude" in args.agent_model.lower()
    is_openai_cua = args.agent_model.lower() == "computer-use-preview"

    if is_claude:
        runner_script = "run_multienv_claude.py"
    elif is_openai_cua:
        runner_script = "run_multienv_openaicua.py"
    else:
        runner_script = "run_multienv.py"

    # Save metadata
    metadata = {
        "example_id": example["example_id"],
        "task_id": example["task_id"],
        "domain": example["domain"],
        "actual_domain": actual_domain,
        "severity": example["severity"],
        "agent_model": args.agent_model,
        "run_index": run_index,
        "perturbed_instruction": example["perturbed_instruction"],
        "execution_timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(run_dir, "execution_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    with tempfile.TemporaryDirectory() as temp_base:
        # Create temp directory structure using actual domain
        temp_examples_dir = os.path.join(temp_base, "evaluation_examples", "examples", actual_domain)
        os.makedirs(temp_examples_dir, exist_ok=True)

        # Create temp task file
        create_temp_task_file(original_task, example, temp_examples_dir)

        # Create test_all.json using actual domain
        test_meta = {actual_domain: [example["task_id"]]}
        meta_file = os.path.join(temp_base, "evaluation_examples", "test_all.json")
        with open(meta_file, 'w') as f:
            json.dump(test_meta, f, indent=2)

        # Temp result dir
        temp_result_dir = os.path.join(temp_base, "results")
        os.makedirs(temp_result_dir, exist_ok=True)

        # Build command using actual domain
        cmd = [
            "python", runner_script,
            "--headless",
            "--observation_type", args.observation_type,
            "--action_space", args.action_space,
            "--result_dir", temp_result_dir,
            "--test_config_base_dir", os.path.join(temp_base, "evaluation_examples"),
            "--test_all_meta_path", meta_file,
            "--max_steps", str(args.max_steps),
            "--num_envs", str(args.num_envs),
            "--provider_name", args.provider_name,
            "--client_password", args.client_password,
            "--region", args.region,
            "--domain", actual_domain,
            "--model", args.agent_model,
        ]

        # Execute
        try:
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            subprocess.run(cmd, cwd=parent_dir, env={**os.environ, "PYTHONPATH": parent_dir})

            # Find results using actual domain
            nested_dir = os.path.join(
                temp_result_dir, args.action_space, args.observation_type,
                args.agent_model, actual_domain, example["task_id"]
            )

            score = 0.0
            if os.path.exists(nested_dir):
                # Read score
                result_file = os.path.join(nested_dir, "result.txt")
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        score = float(f.read().strip())

                # Move files to run directory
                for item in os.listdir(nested_dir):
                    src = os.path.join(nested_dir, item)
                    dst = os.path.join(run_dir, item)
                    if os.path.exists(dst):
                        if os.path.isdir(dst):
                            shutil.rmtree(dst)
                        else:
                            os.remove(dst)
                    shutil.move(src, dst)

                # Extract tokens
                import re
                input_tokens, output_tokens = 0, 0
                runtime_log = os.path.join(run_dir, "runtime.log")
                if os.path.exists(runtime_log):
                    with open(runtime_log, 'r') as f:
                        for line in f:
                            match = re.search(r'(?<!cache_creation_)(?<!cache_read_)input_tokens=(\d+)', line)
                            if match:
                                input_tokens += int(match.group(1))
                            match = re.search(r'output_tokens=(\d+)', line)
                            if match:
                                output_tokens += int(match.group(1))

                # Count steps
                traj_file = os.path.join(run_dir, "traj.jsonl")
                steps = 0
                if os.path.exists(traj_file):
                    with open(traj_file, 'r') as f:
                        steps = sum(1 for line in f if line.strip())

                # Calculate cost
                cost = calculate_cost(args.agent_model, input_tokens, output_tokens) if input_tokens > 0 else 0.0

                # Save cost
                cost_data = {
                    "agent_model": args.agent_model,
                    "run_index": run_index,
                    "steps": steps,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost_usd": cost
                }
                with open(os.path.join(run_dir, "cost.json"), 'w') as f:
                    json.dump(cost_data, f, indent=2)

                print(f"  Run {run_index}: ✓ Score: {score:.2f}, Steps: {steps}, Cost: {format_cost(cost)}")

                return {
                    "run_index": run_index,
                    "score": score,
                    "steps": steps,
                    "status": "completed",
                    "cost": cost,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }

            print(f"  Run {run_index}: ✗ No results found")
            return {"run_index": run_index, "score": 0.0, "status": "failed", "error": "No results"}

        except Exception as e:
            print(f"  Run {run_index}: ✗ Error: {e}")
            return {"run_index": run_index, "score": 0.0, "status": "failed", "error": str(e)}


def aggregate_run_results(run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate statistics across multiple runs."""
    completed_runs = [r for r in run_results if r["status"] == "completed"]

    if not completed_runs:
        return {
            "total_runs": len(run_results),
            "completed_runs": 0,
            "failed_runs": len(run_results),
            "scores": [],
            "mean_score": 0.0,
            "std_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "total_cost": 0.0,
            "mean_steps": 0.0
        }

    scores = [r["score"] for r in completed_runs]
    costs = [r["cost"] for r in completed_runs]
    steps = [r["steps"] for r in completed_runs]

    import statistics

    return {
        "total_runs": len(run_results),
        "completed_runs": len(completed_runs),
        "failed_runs": len(run_results) - len(completed_runs),
        "scores": scores,
        "mean_score": statistics.mean(scores),
        "std_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min_score": min(scores),
        "max_score": max(scores),
        "total_cost": sum(costs),
        "mean_cost": statistics.mean(costs),
        "total_steps": sum(steps),
        "mean_steps": statistics.mean(steps),
        "std_steps": statistics.stdev(steps) if len(steps) > 1 else 0.0
    }


def run_example(example: Dict[str, Any], args, original_task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute multiple runs for a single benchmark example."""

    print("=" * 80)
    print(f"Example: {example['example_id'][:60]}...")
    print(f"Domain: {example['domain']} | Severity: {example['severity']}")
    print(f"Instruction: {example['perturbed_instruction'][:80]}...")
    print(f"Running {args.num_runs} times (run_{args.start_run} to run_{args.start_run + args.num_runs - 1})")
    print("=" * 80)

    # Result directory for this example
    result_dir = os.path.join(args.results_dir, args.agent_model, example["example_id"])
    os.makedirs(result_dir, exist_ok=True)

    # Execute multiple runs
    run_results = []
    for i in range(args.num_runs):
        run_index = args.start_run + i
        run_dir = os.path.join(result_dir, f"run_{run_index}")

        # Skip if exists and skip_existing flag is set
        if args.skip_existing and os.path.exists(os.path.join(run_dir, "result.txt")):
            print(f"  Run {run_index}: ⊘ Skipping (already exists)")
            # Load existing result
            try:
                with open(os.path.join(run_dir, "result.txt"), 'r') as f:
                    score = float(f.read().strip())
                with open(os.path.join(run_dir, "cost.json"), 'r') as f:
                    cost_data = json.load(f)
                run_results.append({
                    "run_index": run_index,
                    "score": score,
                    "steps": cost_data["steps"],
                    "status": "completed",
                    "cost": cost_data["cost_usd"],
                    "input_tokens": cost_data["input_tokens"],
                    "output_tokens": cost_data["output_tokens"]
                })
            except:
                pass
            continue

        result = run_single_execution(example, args, original_task, run_index, result_dir)
        run_results.append(result)

    # Aggregate results
    aggregated = aggregate_run_results(run_results)
    aggregated["example_id"] = example["example_id"]
    aggregated["agent_model"] = args.agent_model
    aggregated["timestamp"] = datetime.now().isoformat()
    aggregated["run_results"] = run_results

    # Save aggregated results
    with open(os.path.join(result_dir, "aggregated_results.json"), 'w') as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nAggregate: Mean score: {aggregated['mean_score']:.3f} ± {aggregated['std_score']:.3f}")
    print(f"           Completed: {aggregated['completed_runs']}/{aggregated['total_runs']}")
    print(f"           Total cost: {format_cost(aggregated['total_cost'])}")

    return aggregated


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 80)
    print("ROBUSTNESS BENCHMARK EXECUTION (MULTIPLE RUNS)")
    print("=" * 80)
    print(f"Agent: {args.agent_model}")
    print(f"Runs per example: {args.num_runs}")
    print(f"Domain: {args.domain}")
    print(f"Severity: {args.severity}")
    print("=" * 80)
    print()

    try:
        # Load examples
        print("Loading benchmark examples...")
        examples = load_benchmark_examples(
            domain_filter=args.domain,
            severity_filter=args.severity,
            example_id_filter=args.example_id
        )

        if args.limit:
            examples = examples[:args.limit]

        # Filter existing (check if all runs are done)
        if args.skip_existing:
            filtered = []
            for ex in examples:
                result_dir = os.path.join(args.results_dir, args.agent_model, ex["example_id"])
                # Check if all runs exist
                all_exist = True
                for i in range(args.num_runs):
                    run_index = args.start_run + i
                    run_file = os.path.join(result_dir, f"run_{run_index}", "result.txt")
                    if not os.path.exists(run_file):
                        all_exist = False
                        break
                if not all_exist:
                    filtered.append(ex)
            examples = filtered

        print(f"Running {len(examples)} examples × {args.num_runs} runs = {len(examples) * args.num_runs} total executions\n")

        # Run examples
        results = []
        for i, example in enumerate(examples, 1):
            print(f"\n[{i}/{len(examples)}]")
            try:
                original_task = load_original_task(example["task_id"], example["domain"])
                result = run_example(example, args, original_task)
                results.append(result)
            except Exception as e:
                print(f"✗ Error: {e}")
                results.append({
                    "example_id": example["example_id"],
                    "total_runs": args.num_runs,
                    "completed_runs": 0,
                    "failed_runs": args.num_runs,
                    "error": str(e)
                })

        # Save summary
        total_executions = sum(r.get("total_runs", 0) for r in results)
        completed_executions = sum(r.get("completed_runs", 0) for r in results)
        total_cost = sum(r.get("total_cost", 0) for r in results)

        # Calculate overall statistics
        all_scores = []
        for r in results:
            if "scores" in r:
                all_scores.extend(r["scores"])

        import statistics
        mean_score = statistics.mean(all_scores) if all_scores else 0.0
        std_score = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0

        summary = {
            "agent_model": args.agent_model,
            "timestamp": datetime.now().isoformat(),
            "num_runs_per_example": args.num_runs,
            "total_examples": len(results),
            "total_executions": total_executions,
            "completed_executions": completed_executions,
            "overall_mean_score": mean_score,
            "overall_std_score": std_score,
            "total_cost": total_cost,
            "results": results
        }

        summary_file = os.path.join(args.results_dir, args.agent_model, "execution_summary_multiple_runs.json")
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 80}")
        print("COMPLETE")
        print(f"{'=' * 80}")
        print(f"Total examples: {len(results)}")
        print(f"Total executions: {total_executions}")
        print(f"Completed executions: {completed_executions}")
        print(f"Overall mean score: {mean_score:.3f} ± {std_score:.3f}")
        print(f"Total cost: {format_cost(total_cost)}")
        print(f"Results: {summary_file}")
        print(f"{'=' * 80}\n")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
