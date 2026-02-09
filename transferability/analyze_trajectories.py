"""
Analyze trajectories from robustness evaluation results.

This script performs two-stage analysis:
1. Generate trajectory summaries using GPT-5-mini
2. Classify trajectories as safe/unsafe using the summaries

Usage:
    # Analyze all examples for an agent
    python analyze_trajectories.py --agent_model claude-haiku-4-5-20251001

    # Analyze specific examples only
    python analyze_trajectories.py \
        --agent_model claude-opus-4-5-20251101 \
        --limit 10

    # Resume interrupted run (skip existing summaries/classifications)
    python analyze_trajectories.py \
        --agent_model claude-sonnet-4-5-20250929 \
        --skip_existing

    # Only generate summaries (skip classification)
    python analyze_trajectories.py \
        --agent_model computer-use-preview \
        --summarize_only

    # Only classify (assumes summaries exist)
    python analyze_trajectories.py \
        --agent_model claude-haiku-4-5-20251001 \
        --classify_only
"""

import argparse
import json
import os
import sys
import base64
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for model_pricing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)

from utils.model_pricing import calculate_cost, format_cost

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai library not installed. Install with: pip install openai")
    sys.exit(1)


def load_trajectory(example_dir: str) -> Dict[str, Any]:
    """Load trajectory data from example directory."""
    traj_file = os.path.join(example_dir, "traj.jsonl")
    metadata_file = os.path.join(example_dir, "execution_metadata.json")

    if not os.path.exists(traj_file):
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load trajectory steps
    steps = []
    with open(traj_file, 'r') as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))

    return {
        "metadata": metadata,
        "steps": steps,
        "example_dir": example_dir
    }


def format_trajectory_for_summary(trajectory_data: Dict[str, Any]) -> tuple[str, List[str]]:
    """
    Format trajectory data for summarization prompt.

    Returns:
        tuple: (formatted_text, list of screenshot paths)
    """
    steps = trajectory_data["steps"]
    metadata = trajectory_data["metadata"]
    example_dir = trajectory_data["example_dir"]

    trajectory_text = f"**Instruction**: {metadata['perturbed_instruction']}\n\n"
    trajectory_text += f"**Total Steps**: {len(steps)}\n\n"
    trajectory_text += "---\n\n"

    screenshot_paths = []

    for i, step in enumerate(steps, 1):
        step_num = step.get("step_num", i)
        response = step.get("response", "")
        action = step.get("action", {})
        screenshot_file = step.get("screenshot_file", "")

        trajectory_text += f"### Step {step_num}\n\n"

        if response:
            trajectory_text += f"**Agent Reasoning**: {response}\n\n"

        # Format action
        if isinstance(action, dict):
            action_input = action.get("input", {})
            if isinstance(action_input, dict):
                action_type = action_input.get("action", "unknown")
                trajectory_text += f"**Action**: {action_type}"

                if action_type in ["left_click", "right_click", "double_click"]:
                    coord = action_input.get("coordinate", [])
                    trajectory_text += f" at coordinate {coord}"
                elif action_type == "type":
                    text = action_input.get("text", "")
                    trajectory_text += f" - Text: '{text}'"
                elif action_type == "key":
                    key = action_input.get("text", "")
                    trajectory_text += f" - Key: '{key}'"
                elif action_type == "wait":
                    duration = action_input.get("duration", 0)
                    trajectory_text += f" - Duration: {duration}s"

                trajectory_text += "\n\n"

        # Add screenshot reference
        if screenshot_file:
            screenshot_path = os.path.join(example_dir, screenshot_file)
            if os.path.exists(screenshot_path):
                trajectory_text += f"**Screenshot**: `{screenshot_file}`\n\n"
                screenshot_paths.append(screenshot_path)

        trajectory_text += "---\n\n"

    return trajectory_text, screenshot_paths


def load_prompt_template(prompt_file: str) -> str:
    """Load prompt template from file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def generate_trajectory_summary(
    trajectory_data: Dict[str, Any],
    api_client: OpenAI,
    model: str = "gpt-5-mini-2025-08-07",
    max_tokens: int = 16384
) -> tuple[str, int, int]:
    """
    Generate trajectory summary using GPT-5-mini with vision.

    Returns:
        tuple: (summary_text, input_tokens, output_tokens)
    """
    # Load summarization prompt
    prompt_template = load_prompt_template("prompt/trajectory_summarization_prompt.md")

    # Format trajectory
    trajectory_text, screenshot_paths = format_trajectory_for_summary(trajectory_data)

    # Replace placeholder
    full_prompt = prompt_template.replace("{TRAJECTORY_STEPS}", trajectory_text)

    # Build message content with screenshots
    message_content = []

    # Add text prompt first
    message_content.append({
        "type": "text",
        "text": full_prompt
    })

    # Add screenshots (sample every N steps to avoid token limits)
    # For 50 steps, sample every 5 steps = 10 images
    sample_rate = max(1, len(screenshot_paths) // 10)

    for i, screenshot_path in enumerate(screenshot_paths):
        if i % sample_rate == 0:  # Sample screenshots
            with open(screenshot_path, 'rb') as f:
                image_data = base64.standard_b64encode(f.read()).decode('utf-8')

            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}",
                    "detail": "low"  # Use low detail to reduce token usage
                }
            })

    # Call API
    # GPT-5 models use max_completion_tokens instead of max_tokens
    if "gpt-5" in model.lower():
        response = api_client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": message_content
            }],
            max_completion_tokens=max_tokens,
        )
    else:
        response = api_client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": message_content
            }],
            max_tokens=max_tokens,
        )

    summary = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    return summary, input_tokens, output_tokens


def classify_trajectory_safety(
    instruction: str,
    trajectory_summary: str,
    api_client: OpenAI,
    model: str = "gpt-5-2025-08-07",
    max_tokens: int = 4096
) -> tuple[Dict[str, Any], int, int]:
    """
    Classify trajectory as safe/unsafe using the summary.

    Returns:
        tuple: (classification_result, input_tokens, output_tokens)
    """
    # Load classification prompt
    prompt_template = load_prompt_template("prompt/trajectory_evaluation.md")

    # Replace placeholders
    full_prompt = prompt_template.replace("{ORIGINAL_INSTRUCTION}", instruction)
    full_prompt = full_prompt.replace("{TRAJECTORY_SUMMARY}", trajectory_summary)

    # Call API
    # GPT-5 models use max_completion_tokens instead of max_tokens
    if "gpt-5" in model.lower():
        response = api_client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": full_prompt
            }],
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
    else:
        response = api_client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": full_prompt
            }],
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

    classification_text = response.choices[0].message.content
    classification_result = json.loads(classification_text)

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    return classification_result, input_tokens, output_tokens


def process_example(
    example_dir: str,
    api_client: OpenAI,
    summary_model: str = "gpt-5-mini-2025-08-07",
    classification_model: str = "gpt-5-2025-08-07",
    skip_existing: bool = False,
    summarize_only: bool = False,
    classify_only: bool = False
) -> Dict[str, Any]:
    """Process a single example: summarize and classify."""

    example_id = os.path.basename(example_dir)
    summary_file = os.path.join(example_dir, "trajectory_summary.md")
    classification_file = os.path.join(example_dir, "safety_classification.json")
    cost_file = os.path.join(example_dir, "analysis_cost.json")

    result = {
        "example_id": example_id,
        "summary_generated": False,
        "classification_generated": False,
        "total_cost": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0
    }

    # Skip if both exist and skip_existing is True
    if skip_existing and os.path.exists(summary_file) and os.path.exists(classification_file):
        print(f"  ⊘ Skipping (already analyzed)")
        return result

    # Load trajectory
    try:
        trajectory_data = load_trajectory(example_dir)
        instruction = trajectory_data["metadata"]["perturbed_instruction"]
    except Exception as e:
        print(f"  ✗ Error loading trajectory: {e}")
        result["error"] = str(e)
        return result

    # Step 1: Generate summary (unless classify_only)
    summary = None
    if not classify_only:
        if skip_existing and os.path.exists(summary_file):
            print(f"  ⊘ Summary exists, loading...")
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = f.read()
        else:
            try:
                print(f"  → Generating summary...")
                summary, input_tokens, output_tokens = generate_trajectory_summary(
                    trajectory_data, api_client, model=summary_model
                )

                # Save summary
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary)

                cost = calculate_cost(summary_model, input_tokens, output_tokens)
                result["summary_generated"] = True
                result["summary_input_tokens"] = input_tokens
                result["summary_output_tokens"] = output_tokens
                result["summary_cost"] = cost
                result["total_cost"] += cost
                result["total_input_tokens"] += input_tokens
                result["total_output_tokens"] += output_tokens

                print(f"  ✓ Summary saved ({input_tokens:,} in / {output_tokens:,} out / {format_cost(cost)})")

            except Exception as e:
                print(f"  ✗ Error generating summary: {e}")
                result["error"] = str(e)
                return result
    else:
        # Load existing summary
        if not os.path.exists(summary_file):
            print(f"  ✗ Summary not found (required for classify_only mode)")
            result["error"] = "Summary not found"
            return result
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = f.read()

    # Step 2: Classify safety (unless summarize_only)
    if not summarize_only:
        if skip_existing and os.path.exists(classification_file):
            print(f"  ⊘ Classification exists")
        else:
            try:
                print(f"  → Classifying safety...")
                classification, input_tokens, output_tokens = classify_trajectory_safety(
                    instruction, summary, api_client, model=classification_model
                )

                # Save classification
                with open(classification_file, 'w', encoding='utf-8') as f:
                    json.dump(classification, f, indent=2, ensure_ascii=False)

                cost = calculate_cost(classification_model, input_tokens, output_tokens)
                result["classification_generated"] = True
                result["classification_input_tokens"] = input_tokens
                result["classification_output_tokens"] = output_tokens
                result["classification_cost"] = cost
                result["classification_result"] = classification.get("classification", "UNKNOWN")
                result["contains_unsafe_action"] = classification.get("contains_unsafe_action", -1)
                result["risk_severity"] = classification.get("risk_severity", "unknown")
                result["total_cost"] += cost
                result["total_input_tokens"] += input_tokens
                result["total_output_tokens"] += output_tokens

                print(f"  ✓ Classification: {classification.get('classification', 'UNKNOWN')} "
                      f"(severity: {classification.get('risk_severity', 'unknown')}) "
                      f"({input_tokens:,} in / {output_tokens:,} out / {format_cost(cost)})")

            except Exception as e:
                print(f"  ✗ Error classifying: {e}")
                result["error"] = str(e)
                return result

    # Save cost summary
    with open(cost_file, 'w') as f:
        json.dump({
            "summary_model": summary_model,
            "classification_model": classification_model,
            "total_input_tokens": result["total_input_tokens"],
            "total_output_tokens": result["total_output_tokens"],
            "total_cost_usd": result["total_cost"],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectories from robustness evaluation"
    )

    parser.add_argument(
        "--agent_model",
        type=str,
        required=True,
        help="Agent model to analyze (e.g., claude-haiku-4-5-20251001)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory (default: ./results/)"
    )
    parser.add_argument(
        "--summary_model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="Model for trajectory summarization (default: gpt-5-mini-2025-08-07)"
    )
    parser.add_argument(
        "--classification_model",
        type=str,
        default="gpt-5-2025-08-07",
        help="Model for safety classification (default: gpt-5-2025-08-07)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of examples to analyze"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip examples that already have analysis results"
    )
    parser.add_argument(
        "--summarize_only",
        action="store_true",
        help="Only generate summaries, skip classification"
    )
    parser.add_argument(
        "--classify_only",
        action="store_true",
        help="Only classify (assumes summaries exist)"
    )

    args = parser.parse_args()

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Initialize API client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        sys.exit(1)

    api_client = OpenAI(api_key=api_key)

    print("=" * 80)
    print("TRAJECTORY ANALYSIS")
    print("=" * 80)
    print(f"Agent:               {args.agent_model}")
    print(f"Summary model:       {args.summary_model}")
    print(f"Classification model: {args.classification_model}")
    print(f"Skip existing:       {args.skip_existing}")
    print(f"Summarize only:      {args.summarize_only}")
    print(f"Classify only:       {args.classify_only}")
    print("=" * 80)
    print()

    # Get agent results directory
    agent_dir = os.path.join(args.results_dir, args.agent_model)
    if not os.path.exists(agent_dir):
        print(f"Error: Agent directory not found: {agent_dir}")
        sys.exit(1)

    # Get all example directories
    example_dirs = []
    for item in os.listdir(agent_dir):
        item_path = os.path.join(agent_dir, item)
        if os.path.isdir(item_path) and ":" in item:  # Example ID format
            example_dirs.append(item_path)

    example_dirs.sort()

    if args.limit:
        example_dirs = example_dirs[:args.limit]

    print(f"Found {len(example_dirs)} examples to analyze\n")

    # Process each example
    results = []
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    summaries_generated = 0
    classifications_generated = 0

    for i, example_dir in enumerate(example_dirs, 1):
        example_id = os.path.basename(example_dir)
        print(f"[{i}/{len(example_dirs)}] {example_id[:80]}...")

        try:
            result = process_example(
                example_dir,
                api_client,
                summary_model=args.summary_model,
                classification_model=args.classification_model,
                skip_existing=args.skip_existing,
                summarize_only=args.summarize_only,
                classify_only=args.classify_only
            )

            results.append(result)
            total_cost += result.get("total_cost", 0)
            total_input_tokens += result.get("total_input_tokens", 0)
            total_output_tokens += result.get("total_output_tokens", 0)

            if result.get("summary_generated"):
                summaries_generated += 1
            if result.get("classification_generated"):
                classifications_generated += 1

        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results.append({
                "example_id": example_id,
                "error": str(e)
            })

        print()

    # Calculate safety statistics
    safe_count = 0
    unsafe_count = 0
    severity_counts = defaultdict(int)

    for result in results:
        if "contains_unsafe_action" in result:
            if result["contains_unsafe_action"] == 0:
                safe_count += 1
            elif result["contains_unsafe_action"] == 1:
                unsafe_count += 1

            # Count severity
            if "risk_severity" in result:
                severity_counts[result["risk_severity"]] += 1

    # Calculate percentages
    total_classified = safe_count + unsafe_count
    safe_percentage = round(safe_count / total_classified * 100, 1) if total_classified > 0 else 0
    unsafe_percentage = round(unsafe_count / total_classified * 100, 1) if total_classified > 0 else 0

    # Save analysis summary
    summary_file = os.path.join(agent_dir, "analysis_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            "agent_model": args.agent_model,
            "summary_model": args.summary_model,
            "classification_model": args.classification_model,
            "timestamp": datetime.now().isoformat(),
            "total_examples": len(example_dirs),
            "summaries_generated": summaries_generated,
            "classifications_generated": classifications_generated,
            "safe_count": safe_count,
            "unsafe_count": unsafe_count,
            "safe_percentage": safe_percentage,
            "unsafe_percentage": unsafe_percentage,
            "severity_distribution": dict(severity_counts),
            "total_cost": total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "results": results
        }, f, indent=2)

    # Print summary
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total examples:           {len(example_dirs)}")
    print(f"Summaries generated:      {summaries_generated}")
    print(f"Classifications generated: {classifications_generated}")
    print(f"Total input tokens:       {total_input_tokens:,}")
    print(f"Total output tokens:      {total_output_tokens:,}")
    print(f"Total cost:               {format_cost(total_cost)}")
    print(f"Summary saved:            {summary_file}")
    print("=" * 80)

    # Print classification breakdown if classifications were generated
    if classifications_generated > 0 and not args.summarize_only:
        print()
        print("SAFETY CLASSIFICATION SUMMARY")
        print("=" * 80)

        print(f"Safe:    {safe_count} ({safe_percentage}%)")
        print(f"Unsafe:  {unsafe_count} ({unsafe_percentage}%)")

        # Severity breakdown
        severity_counts = {}
        for r in results:
            severity = r.get("risk_severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        print()
        print("Severity breakdown:")
        for severity in ["critical", "high", "medium", "low", "minimal", "none", "unknown"]:
            count = severity_counts.get(severity, 0)
            if count > 0:
                print(f"  {severity:12s}: {count:3d} ({100*count/classifications_generated:.1f}%)")

        print("=" * 80)


if __name__ == "__main__":
    main()
