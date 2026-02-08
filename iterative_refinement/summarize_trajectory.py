"""
Summarize agent execution trajectories using an LLM.

This script loads a trajectory file (traj.jsonl) with screenshots and uses
an LLM to generate a comprehensive summary of what the agent did, including
safety analysis.

Features:
- Load trajectory and screenshots from perturbed query execution
- Generate comprehensive summaries with multimodal LLMs
- Analyze safety: Did the agent fall into the trap?
- Cost tracking and token usage reporting

Usage:
    # Summarize a specific execution
    python summarize_trajectory.py \
        --task_id a4d98375-215b-4a4d-aee9-3d4370fccc41 \
        --domain os \
        --perturbed_id 38ea3a0e \
        --agent_model claude-4-sonnet-20250514 \
        --api anthropic

    # Summarize with specific model
    python summarize_trajectory.py \
        --task_id a4d98375-215b-4a4d-aee9-3d4370fccc41 \
        --domain os \
        --perturbed_id 38ea3a0e \
        --agent_model claude-4-sonnet-20250514 \
        --api openai \
        --model gpt-4o

    # Include only every 3rd screenshot (reduces cost)
    python summarize_trajectory.py \
        --task_id a4d98375-215b-4a4d-aee9-3d4370fccc41 \
        --domain os \
        --perturbed_id 38ea3a0e \
        --agent_model claude-4-sonnet-20250514 \
        --api anthropic \
        --screenshot_interval 3

    # Custom max_screenshots limit (override default of 20)
    python summarize_trajectory.py \
        --task_id a4d98375-215b-4a4d-aee9-3d4370fccc41 \
        --domain os \
        --perturbed_id 38ea3a0e \
        --agent_model claude-4-sonnet-20250514 \
        --api anthropic \
        --max_screenshots 50

    # Combine both: every 2nd screenshot, but max 15 total
    python summarize_trajectory.py \
        --task_id a4d98375-215b-4a4d-aee9-3d4370fccc41 \
        --domain os \
        --perturbed_id 38ea3a0e \
        --agent_model claude-4-sonnet-20250514 \
        --api anthropic \
        --screenshot_interval 2 \
        --max_screenshots 15

Output:
    Saves summary to:
    perturbed_queries/{domain}/{task_id}/{gen_model}/perturbed_query_{id}/{agent_model}/trajectory_summary.md
"""

import argparse
import json
import os
import base64
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import imagehash

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.model_pricing import calculate_cost, format_cost

# Load environment variables
load_dotenv()

# Import API clients
try:
    from anthropic import Anthropic, AnthropicBedrock
except ImportError:
    print("Warning: anthropic library not installed. Install with: pip install anthropic")
    Anthropic = None
    AnthropicBedrock = None

try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    print("Warning: openai library not installed. Install with: pip install openai")
    OpenAI = None
    AzureOpenAI = None


def get_api_client(api_type: str):
    """Get the appropriate API client based on the API type."""
    if api_type == "openai":
        if OpenAI is None:
            raise ImportError("openai library not installed")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return OpenAI(api_key=api_key)

    elif api_type == "azure":
        if AzureOpenAI is None:
            raise ImportError("openai library not installed")
        api_key = os.getenv("AZURE_API_KEY")
        endpoint = os.getenv("AZURE_ENDPOINT")
        api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

        if not api_key or not endpoint:
            raise ValueError("AZURE_API_KEY and AZURE_ENDPOINT must be set in environment")

        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )

    elif api_type == "anthropic":
        if Anthropic is None:
            raise ImportError("anthropic library not installed")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        return Anthropic(api_key=api_key)

    elif api_type == "anthropic_bedrock":
        if AnthropicBedrock is None:
            raise ImportError("anthropic library not installed")
        # Use same environment variable names as AWS SDK and AnthropicAgent
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION", "us-east-1")

        if not aws_access_key or not aws_secret_key:
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set")

        return AnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region
        )

    else:
        raise ValueError(f"Invalid API type: {api_type}")


def load_trajectory_data(
    task_id: str,
    domain: str,
    perturbed_id: str,
    agent_model: str,
    perturbed_queries_dir: str = "./perturbed_queries"
) -> Dict[str, Any]:
    """Load trajectory data and perturbed query metadata."""

    # Find the generation model directory
    task_dir = os.path.join(perturbed_queries_dir, domain, task_id)

    # Find which generation model was used
    gen_model_dirs = [d for d in os.listdir(task_dir)
                      if os.path.isdir(os.path.join(task_dir, d))
                      and not d.startswith('.')]

    # Search for the perturbed query
    perturbed_dir = None
    gen_model = None
    for gen_model_candidate in gen_model_dirs:
        candidate_dir = os.path.join(
            task_dir,
            gen_model_candidate,
            f"perturbed_query_{perturbed_id}"
        )
        if os.path.exists(candidate_dir):
            perturbed_dir = candidate_dir
            gen_model = gen_model_candidate
            break

    if not perturbed_dir:
        raise FileNotFoundError(
            f"Could not find perturbed_query_{perturbed_id} for task {task_id}"
        )

    # Path to agent execution results
    agent_dir = os.path.join(perturbed_dir, agent_model)
    if not os.path.exists(agent_dir):
        raise FileNotFoundError(
            f"Could not find execution results for {agent_model} in {perturbed_dir}"
        )

    # Load perturbed query metadata
    metadata_file = os.path.join(perturbed_dir, f"perturbed_query_{perturbed_id}.json")
    if not os.path.exists(metadata_file):
        # Try in agent_dir
        metadata_file = os.path.join(agent_dir, "perturbed_metadata.json")

    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Load trajectory
    traj_file = os.path.join(agent_dir, "traj.jsonl")
    trajectory = []
    with open(traj_file, 'r', encoding='utf-8') as f:
        for line in f:
            trajectory.append(json.loads(line))

    # Load result
    result_file = os.path.join(agent_dir, "result.txt")
    with open(result_file, 'r') as f:
        result = f.read().strip()

    return {
        "task_id": task_id,
        "domain": domain,
        "perturbed_id": perturbed_id,
        "agent_model": agent_model,
        "gen_model": gen_model,
        "metadata": metadata,
        "trajectory": trajectory,
        "result": result,
        "agent_dir": agent_dir,
        "perturbed_dir": perturbed_dir
    }


def compute_image_hash(screenshot_path: str) -> Optional[imagehash.ImageHash]:
    """
    Compute perceptual hash for quick similarity comparison.
    Uses average hash (aHash) - fast and good for detecting similar images.

    Returns:
        ImageHash object or None if file not found/error
    """
    try:
        if not os.path.exists(screenshot_path):
            return None
        with Image.open(screenshot_path) as img:
            # Use average hash - very fast, good for our use case
            return imagehash.average_hash(img, hash_size=8)
    except Exception:
        return None


def load_screenshot(agent_dir: str, screenshot_filename: str) -> Tuple[Optional[str], int]:
    """
    Load and base64 encode a screenshot.

    Returns:
        Tuple of (base64_encoded_data, size_in_bytes) or (None, 0) if not found
    """
    screenshot_path = os.path.join(agent_dir, screenshot_filename)
    if not os.path.exists(screenshot_path):
        return None, 0

    with open(screenshot_path, 'rb') as f:
        image_bytes = f.read()
        encoded = base64.standard_b64encode(image_bytes).decode('utf-8')
        return encoded, len(encoded)


def remove_similar_consecutive_screenshots(
    trajectory: List[Dict[str, Any]],
    agent_dir: str,
    candidate_steps: List[int],
    similarity_threshold: int = 5
) -> List[int]:
    """
    Remove visually similar consecutive screenshots using perceptual hashing.

    Strategy:
        - Compare each screenshot with the next one in sequence
        - If hamming distance < threshold, they're similar → remove one
        - Always keep first and last screenshots
        - Keep the first occurrence of similar sequences

    Args:
        trajectory: Full trajectory list
        agent_dir: Directory containing screenshots
        candidate_steps: Sorted list of candidate step indices
        similarity_threshold: Hamming distance threshold (0-64, lower = more similar)
                            Default 5 means very similar (< 8% different)

    Returns:
        Filtered list of step indices with similar consecutive screenshots removed
    """
    if len(candidate_steps) <= 2:
        return candidate_steps  # Too few to deduplicate

    # Compute hashes for all candidates
    hashes = {}  # idx -> hash
    for idx in candidate_steps:
        step = trajectory[idx]
        screenshot_file = step.get("screenshot_file", "")
        if screenshot_file:
            screenshot_path = os.path.join(agent_dir, screenshot_file)
            img_hash = compute_image_hash(screenshot_path)
            if img_hash is not None:
                hashes[idx] = img_hash

    # Filter out similar consecutive screenshots
    filtered_steps = []
    prev_hash = None

    for i, idx in enumerate(candidate_steps):
        # Always keep first and last
        if i == 0 or i == len(candidate_steps) - 1:
            filtered_steps.append(idx)
            if idx in hashes:
                prev_hash = hashes[idx]
            continue

        # Check similarity with previous kept screenshot
        if idx in hashes and prev_hash is not None:
            hamming_dist = hashes[idx] - prev_hash
            if hamming_dist >= similarity_threshold:
                # Different enough, keep it
                filtered_steps.append(idx)
                prev_hash = hashes[idx]
            # else: too similar, skip it
        else:
            # No hash available, keep by default
            filtered_steps.append(idx)
            if idx in hashes:
                prev_hash = hashes[idx]

    return filtered_steps


def adaptive_screenshot_selection(
    trajectory: List[Dict[str, Any]],
    agent_dir: str,
    candidate_steps: List[int],
    max_payload_mb: float = 50.0,
    use_similarity_dedup: bool = True,
    similarity_threshold: int = 5
) -> Tuple[Dict[int, Tuple[str, int]], Dict[str, Any]]:
    """
    Smart adaptive screenshot selection with similarity-based deduplication.

    Strategy:
        1. Load ALL candidates and measure sizes
        2. If under 50 MB limit → KEEP ALL (no reduction)
        3. If over 50 MB limit → Apply smart reduction:
           a. First: Remove visually similar consecutive screenshots
           b. If still over: Apply importance-based reduction
        4. ALWAYS keep first and last screenshots (critical context)

    Args:
        trajectory: Full trajectory list
        agent_dir: Directory containing screenshots
        candidate_steps: Initial candidate step indices
        max_payload_mb: Maximum payload size in MB (default: 50 MB, OpenAI's limit)
        use_similarity_dedup: Enable similarity-based deduplication (default: True)
        similarity_threshold: Hamming distance threshold for similarity (default: 5)

    Returns:
        Tuple of (selected_screenshot_data, stats)
        - selected_screenshot_data: Dict[idx -> (base64_data, size_bytes)]
        - stats: Dict with selection statistics
    """
    max_payload_bytes = int(max_payload_mb * 1024 * 1024)

    # Step 1: Load ALL candidates and measure sizes (no filtering yet)
    screenshot_data = {}  # idx -> (base64_data, size_bytes)
    total_size = 0

    for idx in candidate_steps:
        step = trajectory[idx]
        screenshot_file = step.get("screenshot_file", "")
        if screenshot_file:
            data, size = load_screenshot(agent_dir, screenshot_file)
            if data:
                screenshot_data[idx] = (data, size)
                total_size += size

    initial_count = len(screenshot_data)
    initial_size_mb = total_size / (1024 * 1024)

    stats = {
        "initial_count": initial_count,
        "initial_size_mb": initial_size_mb,
        "max_payload_mb": max_payload_mb,
        "similarity_dedup_removed": 0,
        "reduced": False,
        "final_count": initial_count,
        "final_size_mb": initial_size_mb
    }

    # Step 2: If under limit, KEEP ALL - no reduction needed!
    if total_size <= max_payload_bytes:
        return screenshot_data, stats

    # Step 3: Over limit - need reduction
    stats["reduced"] = True

    # Step 3a: Try similarity deduplication first (keeps first/last automatically)
    dedup_count = 0
    if use_similarity_dedup and len(screenshot_data) > 2:
        print(f"  ⚠️  Payload size ({initial_size_mb:.1f} MB) exceeds {max_payload_mb} MB limit")
        print(f"  Strategy 1: Removing similar consecutive screenshots (threshold={similarity_threshold})...")

        # Identify similar pairs, but remove them one-by-one until under limit
        # This keeps as many screenshots as possible
        candidate_indices = sorted(screenshot_data.keys())

        # Compute hashes for all screenshots
        hashes = {}
        for idx in candidate_indices:
            step = trajectory[idx]
            screenshot_file = step.get("screenshot_file", "")
            if screenshot_file:
                screenshot_path = os.path.join(agent_dir, screenshot_file)
                img_hash = compute_image_hash(screenshot_path)
                if img_hash is not None:
                    hashes[idx] = img_hash

        # Identify similar consecutive pairs (but don't remove first/last)
        similar_pairs = []  # (idx, similarity_score, size)
        first_idx = candidate_indices[0]
        last_idx = candidate_indices[-1]

        for i in range(len(candidate_indices) - 1):
            idx1 = candidate_indices[i]
            idx2 = candidate_indices[i + 1]

            # Skip if either is first or last
            if idx1 == first_idx or idx1 == last_idx or idx2 == first_idx or idx2 == last_idx:
                continue

            if idx1 in hashes and idx2 in hashes:
                hamming_dist = hashes[idx1] - hashes[idx2]
                if hamming_dist < similarity_threshold:
                    # They're similar - candidate for removal
                    # Remove the second one (keep first occurrence)
                    _, size = screenshot_data[idx2]
                    similar_pairs.append((idx2, hamming_dist, size))

        # Sort by similarity (most similar first) to maximize space savings
        similar_pairs.sort(key=lambda x: x[1])

        # Remove similar screenshots one by one until under limit
        for idx, _, size in similar_pairs:  # sim_score not needed here
            if total_size <= max_payload_bytes:
                break  # Already under limit, stop removing!

            # Remove this screenshot
            del screenshot_data[idx]
            total_size -= size
            dedup_count += 1

        if dedup_count > 0:
            print(f"  → Removed {dedup_count} similar screenshots (now {total_size / (1024 * 1024):.1f} MB)")

        stats["similarity_dedup_removed"] = dedup_count

        # Check if we're now under limit
        if total_size <= max_payload_bytes:
            stats["final_count"] = len(screenshot_data)
            stats["final_size_mb"] = total_size / (1024 * 1024)
            return screenshot_data, stats

    # Step 3b: Still over limit - apply importance-based reduction
    print(f"  Strategy 2: Importance-based reduction...")
    selected_indices = set(screenshot_data.keys())

    # Protect first and last steps (critical for understanding)
    protected = set()
    if selected_indices:
        protected.add(min(selected_indices))
        protected.add(max(selected_indices))

    # Calculate "importance" for each step
    # More important = keep longer
    def calculate_importance(idx: int) -> float:
        """
        Calculate importance score for a step.
        Higher score = more important to keep.
        """
        total_steps = len(trajectory)

        # Factor 1: Distance from nearest neighbor (uniqueness)
        # Steps far from others are more valuable
        sorted_indices = sorted(selected_indices)
        idx_pos = sorted_indices.index(idx)

        dist_to_prev = idx - sorted_indices[idx_pos - 1] if idx_pos > 0 else idx
        dist_to_next = sorted_indices[idx_pos + 1] - idx if idx_pos < len(sorted_indices) - 1 else total_steps - idx
        min_neighbor_dist = min(dist_to_prev, dist_to_next)

        # Factor 2: Position in trajectory (prefer even distribution)
        # Steps in under-represented regions are more valuable
        position_score = min(idx / total_steps, (total_steps - idx) / total_steps) * 2  # Peaks at 0.5

        # Combined score
        return min_neighbor_dist * 2 + position_score * 10

    # Remove steps one by one, starting with least important
    while total_size > max_payload_bytes and len(selected_indices - protected) > 0:
        # Find least important removable step
        removable = selected_indices - protected
        least_important_idx = min(removable, key=calculate_importance)

        # Remove it
        data, size = screenshot_data[least_important_idx]
        del screenshot_data[least_important_idx]
        selected_indices.remove(least_important_idx)
        total_size -= size

    # Update stats
    stats["final_count"] = len(screenshot_data)
    stats["final_size_mb"] = total_size / (1024 * 1024)
    stats["reduction_pct"] = (1 - stats["final_count"] / stats["initial_count"]) * 100 if stats["initial_count"] > 0 else 0

    return screenshot_data, stats


def format_trajectory_steps(
    trajectory: List[Dict[str, Any]],
    agent_dir: str,
    screenshot_interval: int = 1,
    max_screenshots: int = 40,
    max_payload_mb: float = 50.0
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """
    Format trajectory steps with smart adaptive screenshot sampling.

    Args:
        trajectory: List of trajectory steps
        agent_dir: Directory containing screenshots
        screenshot_interval: Include every Nth screenshot (1 = all, 2 = every other, etc.)
        max_screenshots: Maximum number of screenshots (hard cap before size check)
        max_payload_mb: Maximum payload size in MB (default: 50 MB, OpenAI's limit)

    Returns:
        Tuple of (content_blocks, screenshots_b64, selection_stats)

    Smart Selection Logic:
        1. Initial: Apply interval + max_screenshots to get candidates
        2. Adaptive: Use adaptive_screenshot_selection() to respect payload limit
        3. Build: Interleave text and selected screenshots

    Examples:
        - 10 steps, small images -> all 10 screenshots
        - 50 steps, large images -> adaptively reduced to fit 50 MB
        - 100 steps, huge images -> intelligently sampled to stay under limit
    """
    content_blocks = []
    screenshots_b64 = []

    total_steps = len(trajectory)

    # Step 1: Initial candidate selection based on interval/max
    if screenshot_interval <= 1:
        interval_based_steps = list(range(total_steps))
    else:
        interval_based_steps = list(range(0, total_steps, screenshot_interval))

    if len(interval_based_steps) <= max_screenshots:
        candidate_steps = interval_based_steps
    else:
        # Even sampling to respect max_screenshots
        candidate_steps = [
            int(i * total_steps / max_screenshots)
            for i in range(max_screenshots)
        ]

    # Step 2: Apply adaptive selection (respects payload size)
    screenshot_data, selection_stats = adaptive_screenshot_selection(
        trajectory=trajectory,
        agent_dir=agent_dir,
        candidate_steps=candidate_steps,
        max_payload_mb=max_payload_mb
    )

    # Print selection stats
    if not selection_stats['reduced']:
        # Case 1: All screenshots fit, no reduction needed
        print(f"  ✓ All {selection_stats['final_count']} screenshots fit within {selection_stats['max_payload_mb']} MB limit ({selection_stats['final_size_mb']:.1f} MB total)")
    else:
        # Case 2: Had to apply reduction
        if selection_stats.get('similarity_dedup_removed', 0) > 0:
            print(f"  ✓ Similarity dedup: removed {selection_stats['similarity_dedup_removed']} similar screenshots")

        if selection_stats.get('reduction_pct', 0) > 0:
            print(f"  ⚠️  Size reduction: removed {selection_stats['reduction_pct']:.0f}% more to fit limit")

        print(f"  Final: {selection_stats['final_count']} screenshots, {selection_stats['final_size_mb']:.1f} MB")

    screenshot_steps = set(screenshot_data.keys())

    # Step 3: Build content blocks with text and selected screenshots
    for idx, step in enumerate(trajectory):
        step_num = step.get("step_num", idx + 1)
        action = step.get("action", {})
        response = step.get("response", "")

        # Format action
        if isinstance(action, dict):
            action_type = action.get("action_type", "unknown")
            action_input = action.get("input", {})

            # Extract action details
            if "action" in action_input:
                action_name = action_input["action"]
                if action_name == "screenshot":
                    action_str = "Took screenshot"
                elif action_name == "left_click":
                    coord = action_input.get("coordinate", [])
                    action_str = f"Left click at {coord}"
                elif action_name == "right_click":
                    coord = action_input.get("coordinate", [])
                    action_str = f"Right click at {coord}"
                elif action_name == "type":
                    text = action_input.get("text", "")
                    action_str = f"Typed: '{text}'"
                elif action_name == "wait":
                    duration = action_input.get("duration", 0)
                    action_str = f"Waited {duration} seconds"
                elif action_name == "key":
                    key = action_input.get("text", "")
                    action_str = f"Pressed key: {key}"
                else:
                    action_str = f"Action: {action_name}"
            else:
                action_str = str(action)
        elif action == "DONE":
            action_str = "Task completed (DONE)"
        else:
            action_str = str(action)

        # Build step description
        step_text = f"### Step {step_num}\n\n"
        step_text += f"**Action:** {action_str}\n\n"

        if response:
            # Truncate long responses
            if len(response) > 500:
                response = response[:500] + "..."
            step_text += f"**Agent Reasoning:** {response}\n\n"

        # Add step text as a content block
        content_blocks.append({"type": "text", "text": step_text})

        # Add screenshot immediately after the step description if selected
        if idx in screenshot_steps:
            # Use preloaded screenshot data (already size-checked)
            screenshot_b64, _ = screenshot_data[idx]  # size not needed here
            screenshots_b64.append(screenshot_b64)
            content_blocks.append({
                "type": "image",
                "data": screenshot_b64,
                "caption": f"Screenshot for Step {step_num}"
            })

    # Add selection stats to return
    selection_stats["selected_steps"] = sorted(screenshot_steps)

    return content_blocks, screenshots_b64, selection_stats


def load_prompt_template() -> str:
    """Load the trajectory summarization prompt template."""
    prompt_path = os.path.join(
        os.path.dirname(__file__),
        "prompts",
        "trajectory_summarization_prompt.md"
    )
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def generate_summary(
    api_client,
    api_type: str,
    trajectory_data: Dict[str, Any],
    model: str,
    screenshot_interval: int = 1,
    max_screenshots: int = 40,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    max_payload_mb: float = 50.0
) -> Tuple[str, int, int, int]:
    """
    Generate trajectory summary using LLM with smart screenshot sampling.

    Args:
        max_payload_mb: Maximum payload size in MB (default: 50 MB, OpenAI's limit)

    Returns:
        Tuple of (summary_text, input_tokens, output_tokens, total_tokens)
    """
    # Load prompt template
    prompt_template = load_prompt_template()

    # Format trajectory with smart adaptive screenshot selection
    content_blocks, _, _ = format_trajectory_steps(
        trajectory_data["trajectory"],
        trajectory_data["agent_dir"],
        screenshot_interval=screenshot_interval,
        max_screenshots=max_screenshots,
        max_payload_mb=max_payload_mb
    )  # selection_stats printed inside format_trajectory_steps

    # Call API
    if api_type in ["openai", "azure"]:
        # Build messages with interleaved text and images for OpenAI
        content = [{"type": "text", "text": prompt_template}]

        # Add interleaved content blocks
        for block in content_blocks:
            if block["type"] == "text":
                content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{block['data']}"
                    },
                })

        # Handle different model types
        if "gpt-5" in model.lower() or "o4" in model.lower():
            response = api_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=max_tokens,
                temperature=temperature
            )
        else:
            response = api_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=temperature
            )

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        summary = response.choices[0].message.content

    elif api_type in ["anthropic", "anthropic_bedrock"]:
        # Build content with interleaved images and text for Anthropic
        content = [{"type": "text", "text": prompt_template}]

        # Add interleaved content blocks
        for block in content_blocks:
            if block["type"] == "text":
                content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": block["data"]
                    }
                })

        response = api_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content}]
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens
        summary = response.content[0].text

    else:
        raise ValueError(f"Unknown API type: {api_type}")

    return summary, input_tokens, output_tokens, total_tokens


def save_summary(
    trajectory_data: Dict[str, Any],
    summary: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost: float
) -> str:
    """Save the generated summary to file."""
    agent_dir = trajectory_data["agent_dir"]
    output_path = os.path.join(agent_dir, "trajectory_summary.md")

    # Add metadata header
    header = f"""# Trajectory Summary

**Task ID:** {trajectory_data['task_id']}
**Domain:** {trajectory_data['domain']}
**Perturbed Query ID:** {trajectory_data['perturbed_id']}
**Agent Model:** {trajectory_data['agent_model']}
**Summary Model:** {model}
**Input Tokens:** {input_tokens:,}
**Output Tokens:** {output_tokens:,}
**Cost:** {format_cost(cost)}

---

"""

    full_content = header + summary

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_content)

    print(f"Summary saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Summarize agent execution trajectories using an LLM"
    )

    # Required arguments
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
        required=True,
        help="Perturbed query ID (e.g., 38ea3a0e)"
    )
    parser.add_argument(
        "--agent_model",
        type=str,
        required=True,
        help="Agent model that executed the task"
    )

    # API configuration
    parser.add_argument(
        "--api",
        type=str,
        default="anthropic",
        choices=["openai", "azure", "anthropic", "anthropic_bedrock"],
        help="API to use for summarization"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: claude-sonnet-4-5-20250929 for anthropic, gpt-4o for openai)"
    )

    # Screenshot configuration
    parser.add_argument(
        "--screenshot_interval",
        type=int,
        default=1,
        help="Include every Nth screenshot (default: 1 = all, 2 = every other, 3 = every third, etc.)"
    )
    parser.add_argument(
        "--max_screenshots",
        type=int,
        default=20,
        help="Maximum number of screenshots to include (default: 20). Acts as hard cap - if interval-based selection exceeds this, even sampling is used instead"
    )

    # Model parameters
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens for response"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Sampling temperature"
    )

    # Directory paths
    parser.add_argument(
        "--perturbed_queries_dir",
        type=str,
        default="./perturbed_queries",
        help="Directory containing perturbed queries"
    )

    args = parser.parse_args()

    # Set default model based on API
    if args.model is None:
        if args.api == "anthropic":
            args.model = "claude-sonnet-4-5-20250929"
        elif args.api in ["openai", "azure"]:
            args.model = "gpt-4o"
        else:
            raise ValueError(f"Must specify --model for API type: {args.api}")

    print(f"Loading trajectory data...")
    print(f"  Task: {args.task_id}")
    print(f"  Domain: {args.domain}")
    print(f"  Perturbed ID: {args.perturbed_id}")
    print(f"  Agent: {args.agent_model}")

    # Load trajectory data
    trajectory_data = load_trajectory_data(
        args.task_id,
        args.domain,
        args.perturbed_id,
        args.agent_model,
        args.perturbed_queries_dir
    )

    print(f"\nTrajectory loaded:")
    print(f"  Steps: {len(trajectory_data['trajectory'])}")
    print(f"  Result: {trajectory_data['result']}")
    print(f"  Generation model: {trajectory_data['gen_model']}")

    # Get API client
    print(f"\nInitializing {args.api} API client...")
    api_client = get_api_client(args.api)

    # Generate summary
    print(f"Generating summary with {args.model}...")

    # Calculate expected screenshot count for user feedback
    total_steps = len(trajectory_data['trajectory'])
    if args.screenshot_interval <= 1:
        interval_count = total_steps
    else:
        interval_count = (total_steps + args.screenshot_interval - 1) // args.screenshot_interval

    expected_screenshots = min(interval_count, args.max_screenshots)

    print(f"  Trajectory steps: {total_steps}")
    print(f"  Screenshot interval: every {args.screenshot_interval} step(s)")
    print(f"  Max screenshots limit: {args.max_screenshots}")
    print(f"  Expected screenshots to send: ~{expected_screenshots}")

    if interval_count > args.max_screenshots:
        print(f"  Note: Interval selection ({interval_count}) exceeds max_screenshots, using even sampling")

    summary, input_tokens, output_tokens, total_tokens = generate_summary(
        api_client,
        args.api,
        trajectory_data,
        args.model,
        screenshot_interval=args.screenshot_interval,
        max_screenshots=args.max_screenshots,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    # Calculate cost
    cost = calculate_cost(args.model, input_tokens, output_tokens)

    print(f"\nSummary generated:")
    print(f"  Input tokens: {input_tokens:,}")
    print(f"  Output tokens: {output_tokens:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Cost: {format_cost(cost)}")

    # Save summary
    output_path = save_summary(
        trajectory_data,
        summary,
        args.model,
        input_tokens,
        output_tokens,
        cost
    )

    print(f"\n✅ Summary complete!")
    print(f"📄 Saved to: {output_path}")


if __name__ == "__main__":
    main()
