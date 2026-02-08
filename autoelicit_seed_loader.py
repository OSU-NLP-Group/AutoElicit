#!/usr/bin/env python3
"""
AutoElicit-Seed Dataset Loader

A lightweight utility to download the AutoElicit-Seed dataset from HuggingFace
and reconstruct it in the directory structure required by AutoElicit's
iterative refinement pipeline.

Directory Structure:
    {output_dir}/{domain}/{task_id}/{perturbation_model}/perturbed_query_{perturbation_id}/perturbed_query_{perturbation_id}.json

Usage:
    python autoelicit_seed_loader.py                          # Default: ./autoelicit_seed
    python autoelicit_seed_loader.py --output-dir ./my_seeds  # Custom output directory
    python autoelicit_seed_loader.py --domain multi_apps      # Filter by domain
"""

import json
import os
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any


def load_dataset_from_hf(repo_id: str = "osunlp/AutoElicit-Seed") -> List[Dict[str, Any]]:
    """Load the AutoElicit-Seed dataset from HuggingFace.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        
    Returns:
        List of seed perturbation dictionaries
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the 'datasets' library: pip install datasets"
        )
    
    print(f"Loading dataset from HuggingFace: {repo_id}...")
    ds = load_dataset(repo_id, split="train")
    return list(ds)


def reconstruct_seed_file(seed: Dict[str, Any], output_dir: Path) -> Path:
    """Reconstruct a single seed perturbation file in the expected directory structure.
    
    Args:
        seed: Seed perturbation data from the HuggingFace dataset
        output_dir: Base output directory
        
    Returns:
        Path to the created JSON file
    """
    # Extract fields from HuggingFace format
    task_id = seed["task_id"]
    domain = seed["domain"]
    perturbation_model = seed["perturbation_model"]
    perturbation_id = seed["perturbation_id"]
    
    # Create directory structure:
    # {output_dir}/{domain}/{task_id}/{perturbation_model}/perturbed_query_{perturbation_id}/
    query_dir = output_dir / domain / task_id / perturbation_model / f"perturbed_query_{perturbation_id}"
    query_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the JSON file in the format expected by iterative_refinement.py
    seed_data = {
        "id": perturbation_id,
        "task_id": task_id,
        "domain": domain,
        "original_instruction": seed["original_instruction"],
        "perturbed_instruction": seed["perturbed_instruction"],
        "reasoning": seed["perturbation_reasoning"],
        "plausible_unintended_behavior": seed["plausible_unintended_behavior"],
        "elicitation_strategy": seed["elicitation_strategy"],
    }
    
    # Write the JSON file
    json_path = query_dir / f"perturbed_query_{perturbation_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(seed_data, f, indent=2, ensure_ascii=False)
    
    return json_path


def reconstruct_dataset(
    output_dir: Path,
    domain_filter: Optional[str] = None,
    repo_id: str = "osunlp/AutoElicit-Seed"
) -> Dict[str, int]:
    """Download and reconstruct the full AutoElicit-Seed dataset.
    
    Args:
        output_dir: Base output directory for reconstructed files
        domain_filter: Optional domain to filter by (e.g., "multi_apps", "os")
        repo_id: HuggingFace dataset repository ID
        
    Returns:
        Dictionary with statistics about the reconstruction
    """
    # Load dataset from HuggingFace
    seeds = load_dataset_from_hf(repo_id)
    print(f"Loaded {len(seeds)} seed perturbations")
    
    # Filter by domain if specified
    if domain_filter:
        seeds = [s for s in seeds if s["domain"] == domain_filter]
        print(f"Filtered to {len(seeds)} seeds in domain: {domain_filter}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Reconstruct each seed file
    stats = {"total": 0, "domains": {}, "tasks": set()}
    
    for i, seed in enumerate(seeds):
        json_path = reconstruct_seed_file(seed, output_dir)
        
        # Update statistics
        stats["total"] += 1
        domain = seed["domain"]
        stats["domains"][domain] = stats["domains"].get(domain, 0) + 1
        stats["tasks"].add(seed["task_id"])
        
        # Progress indicator
        if (i + 1) % 50 == 0 or (i + 1) == len(seeds):
            print(f"  Processed {i + 1}/{len(seeds)} seeds...")
    
    stats["tasks"] = len(stats["tasks"])
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download and reconstruct AutoElicit-Seed dataset for use with AutoElicit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python autoelicit_seed_loader.py
  python autoelicit_seed_loader.py --output-dir ./my_seeds
  python autoelicit_seed_loader.py --domain multi_apps
  python autoelicit_seed_loader.py --domain os --output-dir ./os_seeds

After reconstruction, you can use the seeds with iterative_refinement.py:
  python iterative_refinement.py --task-id <task_id> --domain <domain> \\
      --perturbed-id <perturbation_id> --base-dir ./autoelicit_seed
        """
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./autoelicit_seed",
        help="Output directory for reconstructed dataset (default: ./autoelicit_seed)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter by domain (e.g., 'multi_apps', 'os')"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="osunlp/AutoElicit-Seed",
        help="HuggingFace dataset repository ID"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AutoElicit-Seed Dataset Loader")
    print("=" * 60)
    
    # Reconstruct the dataset
    stats = reconstruct_dataset(
        output_dir=Path(args.output_dir),
        domain_filter=args.domain,
        repo_id=args.repo_id
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Reconstruction Complete!")
    print("=" * 60)
    print(f"  Output directory: {args.output_dir}")
    print(f"  Total seeds: {stats['total']}")
    print(f"  Unique tasks: {stats['tasks']}")
    print(f"  Domains:")
    for domain, count in sorted(stats["domains"].items()):
        print(f"    - {domain}: {count} seeds")
    
    print(f"\nUsage with iterative_refinement.py:")
    print(f"  python iterative_refinement.py --task-id <task_id> --domain <domain> \\")
    print(f"      --perturbed-id <perturbation_id> --base-dir {args.output_dir}")


if __name__ == "__main__":
    main()
