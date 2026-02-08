#!/usr/bin/env python3
"""
Generate a summary of iterative refinement experiment results.

This script reads task_id:perturbed_query_id pairs from the seed files,
looks up the corresponding refinement_summary.json files, and aggregates
statistics across all runs.

Usage:
    python generate_summary_public.py --domain multi_apps_test \
        --perturbation_model o4-mini-2025-04-16 \
        --refinement_model us_anthropic_claude-haiku-4-5-20251001-v1_0 \
        --agent claude-haiku-4-5-20251001
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a summary of iterative refinement experiment results."
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain name (e.g., 'multi_apps_test', 'os')"
    )
    parser.add_argument(
        "--perturbation_model",
        type=str,
        required=True,
        help="Perturbation model (e.g., 'o4-mini-2025-04-16')"
    )
    parser.add_argument(
        "--refinement_model",
        type=str,
        required=True,
        help="Refinement model identifier (e.g., 'us_anthropic_claude-haiku-4-5-20251001-v1_0')"
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Execution agent (e.g., 'claude-haiku-4-5-20251001')"
    )
    parser.add_argument(
        "--seed_prefix",
        type=str,
        default=None,
        help="Prefix for seed files (default: '{domain}_filtered_seeds_part')"
    )
    parser.add_argument(
        "--num_parts",
        type=int,
        default=5,
        help="Number of seed file parts (default: 5)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Configuration from arguments
    domain = args.domain
    perturbation_model = args.perturbation_model
    refinement_model = args.refinement_model
    agent = args.agent
    
    # Derived paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    perturbed_queries_dir = base_dir / "perturbed_queries" / domain
    
    # Folder names for path construction
    refinement_folder = f"iterative_refinement_{refinement_model}"
    agent_folder = f"agent_{agent}"
    
    # Seed files
    seed_prefix = args.seed_prefix or f"{domain}_filtered_seeds_part"
    seed_files = [
        script_dir / f"{seed_prefix}{i}.txt"
        for i in range(1, args.num_parts + 1)
    ]
    print(seed_files)
    
    # Collect all task pairs
    all_pairs = []
    for seed_file in seed_files:
        if seed_file.exists():
            with open(seed_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        task_id, perturbed_id = line.split(':')
                        all_pairs.append((task_id, perturbed_id))
    
    total_expected = len(all_pairs)
    
    if total_expected == 0:
        print(f"Error: No task pairs found in seed files with prefix '{seed_prefix}'")
        print(f"Searched for files: {[str(f) for f in seed_files]}")
        return
    
    # Track statistics
    found_count = 0
    success_count = 0
    failed_count = 0
    
    total_cost = 0.0
    cost_breakdown = defaultdict(float)
    
    total_execution_iterations = 0
    total_quality_refinements = 0
    
    # Per-task tracking
    task_results = defaultdict(lambda: {"total": 0, "success": 0})
    
    # Severity assessment tracking
    severity_categories = ["none", "minimal", "low", "medium", "high", "critical"]
    severity_counts_success = {cat: 0 for cat in severity_categories}
    severity_tasks_success = {cat: [] for cat in severity_categories}
    severity_counts_all = {cat: 0 for cat in severity_categories}
    severity_tasks_all = {cat: [] for cat in severity_categories}
    
    # Behavior match tracking (across ALL trajectories)
    behavior_match_mapping = {
        "exact_match": "Exact Match",
        "partial_match": "Partial Match",
        "alternative_harmful": "Alternative Harmful",
        "no_match": "No Match"
    }
    behavior_match_counts = {label: 0 for label in behavior_match_mapping.values()}
    total_trajectories = 0
    
    # Results list for detailed analysis
    results = []
    
    for task_id, perturbed_id in all_pairs:
        # Construct path to refinement_summary.json
        result_path = (
            perturbed_queries_dir / task_id / perturbation_model / 
            f"perturbed_query_{perturbed_id}" / refinement_folder / 
            agent_folder / "refinement_summary.json"
        )
        
        task_results[task_id]["total"] += 1
        
        if result_path.exists():
            found_count += 1
            try:
                with open(result_path, 'r') as f:
                    data = json.load(f)
                
                success = data.get("success", False)
                exec_iterations = data.get("execution_iterations", 0)
                quality_refinements = data.get("total_quality_refinements", 0)
                costs = data.get("cost_breakdown", {})
                
                # Extract severity assessment from the last execution iteration
                severity = None
                execution_history = data.get("execution_history", [])
                if execution_history:
                    last_execution = execution_history[-1]
                    traj_eval = last_execution.get("trajectory_evaluation", {})
                    severity = traj_eval.get("severity_assessment", None)
                
                # Track behavior match across ALL trajectories in execution history
                for execution in execution_history:
                    traj_eval = execution.get("trajectory_evaluation", {})
                    match_value = traj_eval.get("matches_expected_behavior", None)
                    if match_value and match_value in behavior_match_mapping:
                        label = behavior_match_mapping[match_value]
                        behavior_match_counts[label] += 1
                        total_trajectories += 1
                
                results.append({
                    "task_id": task_id,
                    "perturbed_id": perturbed_id,
                    "success": success,
                    "execution_iterations": exec_iterations,
                    "quality_refinements": quality_refinements,
                    "cost": costs.get("total_cost", 0),
                    "severity_assessment": severity
                })
                
                # Track severity for all found results (entire dataset)
                task_ref = f"{task_id[:8]}...:{perturbed_id[:8]}"
                if severity and severity in severity_categories:
                    severity_counts_all[severity] += 1
                    severity_tasks_all[severity].append(task_ref)
                
                if success:
                    success_count += 1
                    task_results[task_id]["success"] += 1
                    # Track severity counts and tasks for successful elicitations only
                    if severity and severity in severity_categories:
                        severity_counts_success[severity] += 1
                        severity_tasks_success[severity].append(f"{task_id[:8]}...:{perturbed_id[:8]}")
                else:
                    failed_count += 1
                
                total_execution_iterations += exec_iterations
                total_quality_refinements += quality_refinements
                
                # Aggregate cost breakdown
                for cost_key, cost_val in costs.items():
                    if isinstance(cost_val, (int, float)):
                        cost_breakdown[cost_key] += cost_val
                        if cost_key == "total_cost":
                            total_cost += cost_val
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing {result_path}: {e}")
                failed_count += 1
        else:
            # File not found
            pass
    
    # Calculate averages
    avg_cost = total_cost / found_count if found_count > 0 else 0
    avg_exec_iterations = total_execution_iterations / found_count if found_count > 0 else 0
    avg_quality_refinements = total_quality_refinements / found_count if found_count > 0 else 0
    
    success_rate = (success_count / found_count * 100) if found_count > 0 else 0
    
    # Sort tasks by success rate
    task_success_rates = []
    for task_id, stats in task_results.items():
        if stats["total"] > 0:
            rate = stats["success"] / stats["total"] * 100
            task_success_rates.append((task_id, rate, stats["success"], stats["total"]))
    task_success_rates.sort(key=lambda x: (-x[1], x[0]))
    
    # Generate output
    output_lines = []
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("ITERATIVE REFINEMENT BATCH RUN SUMMARY")
    output_lines.append("=" * 80)
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("")
    output_lines.append("Configuration:")
    output_lines.append(f"  Domain: {domain}")
    output_lines.append(f"  Perturbation Model: {perturbation_model}")
    output_lines.append(f"  Refinement Model: {refinement_model}")
    output_lines.append(f"  Execution Agent: {agent}")
    output_lines.append(f"  Base Directory: perturbed_queries/")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("OVERALL SUMMARY:")
    output_lines.append(f"  Total Expected: {total_expected}")
    output_lines.append(f"  Total Found: {found_count}")
    output_lines.append(f"  Successful: {success_count} ({success_rate:.1f}%)")
    output_lines.append(f"  Failed: {failed_count} ({100 - success_rate:.1f}%)")
    output_lines.append(f"  Missing: {total_expected - found_count}")
    output_lines.append(f"  Total Cost: ${total_cost:.2f}")
    output_lines.append("")
    output_lines.append("COST BREAKDOWN:")
    for cost_key in sorted(cost_breakdown.keys()):
        output_lines.append(f"  {cost_key}: ${cost_breakdown[cost_key]:.2f}")
    output_lines.append("")
    output_lines.append("AVERAGES (per run):")
    output_lines.append(f"  Avg Cost: ${avg_cost:.2f}")
    output_lines.append(f"  Avg Execution Iterations: {avg_exec_iterations:.1f}")
    output_lines.append(f"  Avg Quality Refinements: {avg_quality_refinements:.1f}")
    output_lines.append("")
    output_lines.append("SUCCESS RATE BY TASK (sorted by success rate):")
    for task_id, rate, succ, total in task_success_rates:
        short_id = task_id[:8] + "..."
        output_lines.append(f"  {short_id}: {rate:.1f}% ({succ}/{total})")
    output_lines.append("")
    output_lines.append("-" * 80)
    output_lines.append("")
    output_lines.append("SEVERITY ASSESSMENT BREAKDOWN (successful elicitations only):")
    total_with_severity_success = sum(severity_counts_success.values())
    for category in severity_categories:
        count = severity_counts_success[category]
        percentage = (count / total_with_severity_success * 100) if total_with_severity_success > 0 else 0
        output_lines.append(f"  {category.upper()}: {count} ({percentage:.1f}%)")
        if severity_tasks_success[category]:
            for task_ref in severity_tasks_success[category]:
                output_lines.append(f"    - {task_ref}")
    output_lines.append("")
    output_lines.append("-" * 80)
    output_lines.append("")
    output_lines.append("SEVERITY ASSESSMENT BREAKDOWN (entire dataset):")
    total_with_severity_all = sum(severity_counts_all.values())
    for category in severity_categories:
        count = severity_counts_all[category]
        percentage = (count / total_with_severity_all * 100) if total_with_severity_all > 0 else 0
        output_lines.append(f"  {category.upper()}: {count} ({percentage:.1f}%)")
        if severity_tasks_all[category]:
            for task_ref in severity_tasks_all[category]:
                output_lines.append(f"    - {task_ref}")
    output_lines.append("")
    output_lines.append("-" * 80)
    output_lines.append("")
    output_lines.append("BEHAVIOR MATCH BREAKDOWN (across all trajectories):")
    output_lines.append(f"  Total Trajectories: {total_trajectories}")
    for label in ["Exact Match", "Partial Match", "Alternative Harmful", "No Match"]:
        count = behavior_match_counts[label]
        percentage = (count / total_trajectories * 100) if total_trajectories > 0 else 0
        output_lines.append(f"  {label}: {count} ({percentage:.1f}%)")
    
    output_lines.append("")
    output_lines.append("=" * 80)
    
    # Print to stdout
    report = "\n".join(output_lines)
    print(report)
    
    # Save to file with config info in filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_agent = agent.replace("/", "_")
    output_filename = f"run_summary_{domain}_{safe_agent}_{refinement_model}_{timestamp}.txt"
    output_file = script_dir / "run_summary_logs" / output_filename
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)
    print(f"\nSummary saved to: {output_file}")


if __name__ == "__main__":
    main()
