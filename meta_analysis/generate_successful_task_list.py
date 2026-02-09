#!/usr/bin/env python3
"""
Generate a JSON file containing all successful elicitation runs for a specific configuration.

This script searches through the perturbed_queries directory to find successful
elicitation runs matching the specified perturbation model and agent. It discovers
all refinement models automatically and creates separate task lists for each.

Supports two optional filtering mechanisms:
1. Baseline harm rate filter: Exclude tasks with baseline harm rate > threshold
2. Human annotation filter: Exclude examples marked as "False Positive" by human annotators

Usage:
    # Basic usage with defaults (Claude Haiku 4.5 agent, discovers all refinement models)
    python generate_successful_task_list.py
    
    # With data filter for 0% baseline harm rate
    python generate_successful_task_list.py --data_filter data_filter_harm_threshold_0%_final.json
    
    # With human annotation filter to exclude false positives
    python generate_successful_task_list.py \\
        --human_annotation_filter /path/to/final_decisions.json
    
    # With both filters combined
    python generate_successful_task_list.py \\
        --domain multi_apps_test \\
        --agent claude-haiku-4-5-20251001 \\
        --data_filter data_filter_harm_threshold_0%_final.json \\
        --human_annotation_filter /path/to/final_decisions.json \\
        --output task_list_claude_haiku_filtered.json
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple


def load_data_filter(filter_path: str, domain: str, agent: str) -> Optional[set]:
    """
    Load the data filter file and extract the set of task IDs to EXCLUDE for the 
    specified domain and agent.
    
    The data filter file lists task IDs where the baseline harm rate EXCEEDS the
    threshold (e.g., >0%), meaning these tasks should be EXCLUDED when filtering
    for tasks with 0% baseline harm rate.
    
    Args:
        filter_path: Path to the data filter JSON file
        domain: The domain to filter for (e.g., "multi_apps_test")
        agent: The agent name (e.g., "claude-haiku-4-5-20251001")
        
    Returns:
        Set of task IDs to EXCLUDE, or None if no filter should be applied
    """
    if not filter_path:
        return None
    
    filter_file = Path(filter_path)
    if not filter_file.exists():
        print(f"Warning: Data filter file not found: {filter_path}")
        return None
    
    try:
        with open(filter_file, 'r') as f:
            filter_data = json.load(f)
        
        # Construct the agent key format used in the filter file
        agent_key = f"agent_{agent}"
        
        if domain not in filter_data:
            print(f"Warning: Domain '{domain}' not found in data filter file")
            print(f"  Available domains: {list(filter_data.keys())}")
            return set()
        
        if agent_key not in filter_data[domain]:
            print(f"Warning: Agent '{agent_key}' not found in data filter for domain '{domain}'")
            print(f"  Available agents: {list(filter_data[domain].keys())}")
            return set()
        
        task_ids = set(filter_data[domain][agent_key])
        print(f"Loaded {len(task_ids)} task IDs to EXCLUDE from data filter for {domain}/{agent_key}")
        print(f"  (These tasks have baseline harm rate > threshold)")
        return task_ids
        
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading data filter file: {e}")
        return None


def load_human_annotation_filter(
    annotation_path: str, 
    agent: str
) -> Tuple[Optional[Set[str]], Optional[Dict[str, dict]]]:
    """
    Load the human annotation decisions file and extract examples to EXCLUDE.
    
    Excludes examples where the final_decision is "False Positive".
    
    The annotation file uses keys in the format:
        {task_id}:{perturbed_id}:{refinement_model}:{execution_agent}
    
    Args:
        annotation_path: Path to the final_decisions.json file
        agent: The execution agent to filter for
        
    Returns:
        Tuple of:
        - Set of example keys to EXCLUDE (task_id:perturbed_id:refinement_model format)
        - Dict of excluded example details for reporting, or None if no filter
    """
    if not annotation_path:
        return None, None
    
    annotation_file = Path(annotation_path)
    if not annotation_file.exists():
        print(f"Warning: Human annotation file not found: {annotation_path}")
        return None, None
    
    try:
        with open(annotation_file, 'r') as f:
            annotation_data = json.load(f)
        
        # Find all false positives for the specified agent
        false_positives: Set[str] = set()
        false_positive_details: Dict[str, dict] = {}
        
        for example_key, example_data in annotation_data.items():
            # Parse the example key: task_id:perturbed_id:refinement_model:execution_agent
            parts = example_key.split(":")
            if len(parts) != 4:
                continue
            
            task_id, perturbed_id, refinement_model, exec_agent = parts
            
            # Only consider examples for the specified agent
            if exec_agent != agent:
                continue
            
            # Check if this is a false positive
            final_decision = example_data.get("final_decision", "")
            if final_decision == "False Positive":
                # Store key without the agent (task_id:perturbed_id:refinement_model)
                # This format allows checking during the search
                fp_key = f"{task_id}:{perturbed_id}:{refinement_model}"
                false_positives.add(fp_key)
                false_positive_details[fp_key] = {
                    "task_id": task_id,
                    "perturbed_id": perturbed_id,
                    "refinement_model": refinement_model,
                    "execution_agent": exec_agent,
                    "severity": example_data.get("severity", "unknown"),
                    "vote_count": example_data.get("vote_count", 0),
                }
        
        if false_positives:
            print(f"Loaded {len(false_positives)} false positive examples to EXCLUDE for agent {agent}")
            print(f"  (These were marked as 'False Positive' by human annotators)")
        else:
            print(f"No false positive examples found for agent {agent} in annotation file")
        
        return false_positives, false_positive_details
        
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading human annotation file: {e}")
        return None, None


def find_successful_elicitation_runs(
    base_dir: str = "../perturbed_queries",
    domain: Optional[str] = None,
    perturbation_model: str = "o4-mini-2025-04-16",
    refinement_model: Optional[str] = None,
    agent: str = "claude-haiku-4-5-20251001",
    excluded_task_ids: Optional[Set[str]] = None,
    excluded_false_positives: Optional[Set[str]] = None,
) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """
    Find all successful elicitation runs matching the specified configuration.
    
    If refinement_model is None, discovers all refinement models automatically and
    returns results organized by refinement model.
    
    Args:
        base_dir: Base directory containing perturbed_queries
        domain: Specific domain to search (if None, searches all domains)
        perturbation_model: The perturbation model used
        refinement_model: The refinement model used (if None, discovers all)
        agent: The execution agent
        excluded_task_ids: Optional set of task IDs to EXCLUDE (those with baseline harm > threshold)
        excluded_false_positives: Optional set of example keys to EXCLUDE (human-annotated false positives)
                                  Format: task_id:perturbed_id:refinement_model
        
    Returns:
        Tuple of:
        - Dictionary mapping refinement_model -> list of task specification details
        - List of examples excluded due to human annotation (false positives)
    """
    # Results organized by refinement model
    results_by_refinement: Dict[str, List[dict]] = defaultdict(list)
    excluded_baseline_count = 0
    excluded_annotation_list: List[dict] = []
    discovered_refinement_models: Set[str] = set()
    
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return {}, []
    
    # Determine which domains to search
    if domain:
        domains_to_search = [base_path / domain]
    else:
        domains_to_search = [d for d in base_path.iterdir() if d.is_dir()]
    
    for domain_dir in domains_to_search:
        if not domain_dir.is_dir():
            continue
        
        domain_name = domain_dir.name
        
        # Skip non-domain directories (like zip files, python scripts, etc.)
        if any(domain_name.endswith(ext) for ext in ['.zip', '.py', '.txt', '.json']):
            continue
        
        print(f"\nSearching domain: {domain_name}")
        
        # Iterate through all task IDs in the domain
        for task_dir in domain_dir.iterdir():
            if not task_dir.is_dir():
                continue
            
            task_id = task_dir.name
            
            # Apply exclusion filter if provided (EXCLUDE tasks with baseline harm > threshold)
            if excluded_task_ids is not None and task_id in excluded_task_ids:
                excluded_baseline_count += 1
                continue
            
            # Check for the perturbation model directory
            perturbation_dir = task_dir / perturbation_model
            if not perturbation_dir.exists():
                continue
            
            # Iterate through all perturbed queries
            for perturbed_query_dir in perturbation_dir.iterdir():
                if not perturbed_query_dir.is_dir():
                    continue
                
                # Extract perturbed_id from directory name (perturbed_query_{perturbed_id})
                if not perturbed_query_dir.name.startswith("perturbed_query_"):
                    continue
                
                perturbed_id = perturbed_query_dir.name.replace("perturbed_query_", "")
                
                # Determine which refinement models to search
                if refinement_model:
                    # Specific refinement model requested
                    refinement_models_to_check = [refinement_model]
                else:
                    # Discover all refinement models in this perturbed query directory
                    refinement_models_to_check = []
                    for item in perturbed_query_dir.iterdir():
                        if item.is_dir() and item.name.startswith("iterative_refinement_"):
                            rm = item.name.replace("iterative_refinement_", "")
                            # Skip attempt folders (e.g., iterative_refinement_model_attempt_1)
                            if "_attempt_" not in rm:
                                refinement_models_to_check.append(rm)
                                discovered_refinement_models.add(rm)
                
                # Check each refinement model
                for rm in refinement_models_to_check:
                    refinement_dir = perturbed_query_dir / f"iterative_refinement_{rm}"
                    if not refinement_dir.exists():
                        continue
                    
                    # Check for the agent directory
                    agent_dir = refinement_dir / f"agent_{agent}"
                    if not agent_dir.exists():
                        continue
                    
                    # Check for refinement_summary.json
                    summary_file = agent_dir / "refinement_summary.json"
                    if not summary_file.exists():
                        continue
                    
                    # Check if the run was successful
                    try:
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                        
                        if summary.get("success", False):
                            # Check if this example is a human-annotated false positive
                            fp_key = f"{task_id}:{perturbed_id}:{rm}"
                            if excluded_false_positives is not None and fp_key in excluded_false_positives:
                                excluded_annotation_list.append({
                                    "domain": domain_name,
                                    "task_id": task_id,
                                    "perturbed_id": perturbed_id,
                                    "refinement_model": rm,
                                    "task_spec": f"{domain_name}:{task_id}:{perturbed_id}",
                                    "fp_key": fp_key,
                                })
                                continue
                            
                            task_entry = {
                                "domain": domain_name,
                                "task_id": task_id,
                                "perturbed_id": perturbed_id,
                                "task_spec": f"{domain_name}:{task_id}:{perturbed_id}",
                                "refinement_model": rm,
                                "refinement_summary_path": str(summary_file),
                                "execution_iterations": summary.get("execution_iterations", 0),
                                "final_score": summary.get("final_score", 0),
                            }
                            results_by_refinement[rm].append(task_entry)
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"  Warning: Could not read {summary_file}: {e}")
                        continue
    
    if excluded_task_ids is not None:
        print(f"\n  Excluded {excluded_baseline_count} task directories (baseline harm > threshold)")
    
    if excluded_false_positives is not None and excluded_annotation_list:
        print(f"  Excluded {len(excluded_annotation_list)} examples (human-annotated false positives)")
    
    if not refinement_model and discovered_refinement_models:
        print(f"\n  Discovered {len(discovered_refinement_models)} refinement model(s):")
        for rm in sorted(discovered_refinement_models):
            count = len(results_by_refinement.get(rm, []))
            print(f"    - {rm}: {count} successful runs")
    
    return dict(results_by_refinement), excluded_annotation_list


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a task list of successful elicitation runs for meta-analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate task list for all refinement models with 0% baseline harm rate filter
    python generate_successful_task_list.py \\
        --domain multi_apps_test \\
        --agent claude-haiku-4-5-20251001 \\
        --data_filter data_filter_harm_threshold_0%_final.json
        
    # Generate task list with human annotation filter to exclude false positives
    python generate_successful_task_list.py \\
        --agent claude-haiku-4-5-20251001 \\
        --human_annotation_filter /path/to/final_decisions.json
        
    # Generate task list with both filters combined
    python generate_successful_task_list.py \\
        --domain multi_apps_test \\
        --agent claude-haiku-4-5-20251001 \\
        --data_filter data_filter_harm_threshold_0%_final.json \\
        --human_annotation_filter /path/to/final_decisions.json
        
    # Generate task list for all successful runs (no filter, all refinement models)
    python generate_successful_task_list.py --agent claude-haiku-4-5-20251001
        """
    )
    
    parser.add_argument(
        '--base_dir', 
        type=str, 
        default="../perturbed_queries",
        help='Base directory containing perturbed_queries (default: ../perturbed_queries)'
    )
    parser.add_argument(
        '--domain', 
        type=str, 
        default="multi_apps_test",
        help='Domain to search (default: multi_apps_test, use "all" to search all domains)'
    )
    parser.add_argument(
        '--perturbation_model', 
        type=str, 
        default="o4-mini-2025-04-16",
        help='Perturbation model used (default: o4-mini-2025-04-16)'
    )
    parser.add_argument(
        '--refinement_model', 
        type=str, 
        default=None,
        help='Refinement model used for iterative refinement (default: None = discover all)'
    )
    parser.add_argument(
        '--agent', 
        type=str, 
        default="claude-haiku-4-5-20251001",
        choices=[
            "claude-haiku-4-5-20251001",
            "claude-opus-4-5-20251101",
            "computer-use-preview",
        ],
        help='Execution agent (default: claude-haiku-4-5-20251001)'
    )
    parser.add_argument(
        '--data_filter', 
        type=str, 
        default=None,
        help='Path to data filter JSON file for 0%% baseline harm rate filtering'
    )
    parser.add_argument(
        '--human_annotation_filter', 
        type=str, 
        default=None,
        help='Path to human annotation decisions JSON file (final_decisions.json) to exclude false positives'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Output JSON file path (default: auto-generated based on configuration)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Handle domain argument
    domain = None if args.domain.lower() == "all" else args.domain
    
    # Generate default output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filter_suffix = "_0pct_baseline" if args.data_filter else ""
        annotation_suffix = "_human_filtered" if args.human_annotation_filter else ""
        domain_suffix = args.domain if domain else "all_domains"
        agent_suffix = args.agent.replace("-", "_")
        refinement_suffix = f"_{args.refinement_model}" if args.refinement_model else "_all_refinement_models"
        output_file = f"task_list_{domain_suffix}_{agent_suffix}{refinement_suffix}{filter_suffix}{annotation_suffix}_{timestamp}.json"
    else:
        output_file = args.output
    
    print("=" * 80)
    print("Generate Successful Elicitation Task List")
    print("=" * 80)
    print(f"Base Directory:           {args.base_dir}")
    print(f"Domain:                   {args.domain}")
    print(f"Perturbation Model:       {args.perturbation_model}")
    print(f"Refinement Model:         {args.refinement_model or 'All (auto-discover)'}")
    print(f"Agent:                    {args.agent}")
    print(f"Data Filter:              {args.data_filter or 'None (no filtering)'}")
    print(f"Human Annotation Filter:  {args.human_annotation_filter or 'None (no filtering)'}")
    print(f"Output File:              {output_file}")
    print("=" * 80)
    
    # Load data filter if provided (contains task IDs to EXCLUDE based on baseline harm)
    excluded_task_ids = None
    if args.data_filter and domain:
        excluded_task_ids = load_data_filter(args.data_filter, domain, args.agent)
        if excluded_task_ids is not None:
            print(f"\nWill exclude {len(excluded_task_ids)} task IDs with baseline harm > threshold")
    
    # Load human annotation filter if provided (contains false positives to EXCLUDE)
    excluded_false_positives = None
    false_positive_details = None
    if args.human_annotation_filter:
        excluded_false_positives, false_positive_details = load_human_annotation_filter(
            args.human_annotation_filter, args.agent
        )
        if excluded_false_positives:
            print(f"Will exclude {len(excluded_false_positives)} human-annotated false positives")
    
    # Find successful runs (returns dict organized by refinement model)
    results_by_refinement, excluded_annotation_list = find_successful_elicitation_runs(
        base_dir=args.base_dir,
        domain=domain,
        perturbation_model=args.perturbation_model,
        refinement_model=args.refinement_model,
        agent=args.agent,
        excluded_task_ids=excluded_task_ids,
        excluded_false_positives=excluded_false_positives,
    )
    
    # Sort tasks within each refinement model for consistent output
    for rm in results_by_refinement:
        results_by_refinement[rm].sort(key=lambda x: x["task_spec"])
    
    # Calculate total across all refinement models
    total_successful = sum(len(tasks) for tasks in results_by_refinement.values())
    
    # Create task lists organized by refinement model
    task_lists_by_refinement = {}
    for rm, tasks in results_by_refinement.items():
        task_lists_by_refinement[rm] = [task["task_spec"] for task in tasks]
    
    # Organize excluded false positives by refinement model for reporting
    excluded_fp_by_refinement: Dict[str, List[dict]] = defaultdict(list)
    for excluded in excluded_annotation_list:
        excluded_fp_by_refinement[excluded["refinement_model"]].append(excluded)
    
    # Create output JSON with metadata
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "base_dir": args.base_dir,
            "domain": args.domain,
            "perturbation_model": args.perturbation_model,
            "refinement_model": args.refinement_model,
            "agent": args.agent,
            "data_filter": args.data_filter,
            "human_annotation_filter": args.human_annotation_filter,
            "excluded_task_ids_baseline_harm": sorted(list(excluded_task_ids)) if excluded_task_ids else [],
            "num_excluded_tasks_baseline_harm": len(excluded_task_ids) if excluded_task_ids else 0,
            "num_excluded_false_positives": len(excluded_annotation_list),
            "excluded_false_positives_by_refinement_model": {
                rm: len(fps) for rm, fps in excluded_fp_by_refinement.items()
            } if excluded_annotation_list else {},
            "refinement_models_found": sorted(list(results_by_refinement.keys())),
            "total_successful_runs": total_successful,
            "successful_runs_by_refinement_model": {
                rm: len(tasks) for rm, tasks in results_by_refinement.items()
            },
        },
        # Task lists organized by refinement model
        "task_lists_by_refinement_model": task_lists_by_refinement,
        # Task details organized by refinement model
        "task_details_by_refinement_model": results_by_refinement,
        # Excluded false positives (for transparency)
        "excluded_false_positives": excluded_annotation_list if excluded_annotation_list else [],
    }
    
    # Write to file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total successful runs found: {total_successful}")
    print(f"Refinement models found: {len(results_by_refinement)}")
    for rm in sorted(results_by_refinement.keys()):
        count = len(results_by_refinement[rm])
        print(f"  - {rm}: {count} successful runs")
    
    if excluded_task_ids is not None:
        print(f"\nExcluded (baseline harm > threshold): {len(excluded_task_ids)} task IDs")
        for tid in sorted(excluded_task_ids):
            print(f"    ✗ {tid}")
    
    if excluded_annotation_list:
        print(f"\nExcluded (human-annotated false positives): {len(excluded_annotation_list)} examples")
        for rm in sorted(excluded_fp_by_refinement.keys()):
            fps = excluded_fp_by_refinement[rm]
            print(f"  [{rm}]: {len(fps)} false positives")
            for fp in fps:
                fp_detail = false_positive_details.get(fp["fp_key"], {}) if false_positive_details else {}
                severity = fp_detail.get("severity", "unknown")
                print(f"    ✗ {fp['task_id']}:{fp['perturbed_id']} (severity: {severity})")
    
    print(f"\nOutput written to: {output_file}")
    print("=" * 80)
    
    # Print task breakdown by refinement model and task_id
    for rm in sorted(results_by_refinement.keys()):
        tasks = results_by_refinement[rm]
        if tasks:
            print(f"\n[{rm}] Task breakdown ({len(tasks)} successful):")
            task_id_counts = {}
            for task in tasks:
                tid = task["task_id"]
                task_id_counts[tid] = task_id_counts.get(tid, 0) + 1
            
            for tid, count in sorted(task_id_counts.items()):
                print(f"  ✓ {tid}: {count} successful perturbed instruction(s)")


if __name__ == "__main__":
    main()

