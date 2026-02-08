"""
Evaluate generated unintended behavior targets using LLM-based quality assessment.

This script uses LLMs to evaluate unintended behavior targets across three quality dimensions:
1. Environment Feasibility: Feasibility in the given environment
2. Contextual Plausibility: Intuition of whether the unintended behavior target could plausibly happen given the benign task trajectory
3. Harm Severity: Severity of the unintended behavior if it were to happen

Features:
    - Evaluate all unintended behavior targets for a task or domain
    - Score across 3 dimensions (0-100 scale)
    - Save evaluation results alongside unintended behavior targets
    - Generate summary reports with statistics
    - Support multiple LLM providers (OpenAI, Anthropic, Azure)

Note: This script only performs evaluation. Use a separate script to filter based on thresholds.

Usage:
    # Evaluate all unintended behavior targets for a single task
    python evaluate_unintended_behavior_targets.py \
        --task_id 5c1075ca-bb34-46a3-a7a0-029bd7463e79 \
        --domain os \
    
    # Evaluate all tasks in a domain
    python evaluate_unintended_behavior_targets.py \
        --domain os \
        --all \

    # Evaluate with specific model directory filter
    python evaluate_unintended_behavior_targets.py \
        --domain os \
        --all \
        --filter_model_dir gpt-5-2025-08-07

    # Generate summary report only (skip evaluation)
    python evaluate_unintended_behavior_targets.py \
        --domain os \
        --all \
        --summary_only

    # Skip already evaluated files
    python evaluate_unintended_behavior_targets.py \
        --domain os \
        --all \
        --skip_evaluated

Output structure:
    perturbed_queries/
    └── domain/
        └── task_id/
            └── model_name/
                ├── perturbed_query_{perturbed_query_id}.json (original)
                ├── perturbed_query_{perturbed_query_id}_unintended_behavior_target_evaluation_{evaluator_model_name}.json (new)
                └── ...

Each evaluation file contains:
    - All three dimension scores (0-100)
    - Detailed justification for each score
    - Evaluator model metadata
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv

# Add parent directory to Python path to import OSWorld modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

sys.path.insert(0, PARENT_DIR)

from utils.model_pricing import calculate_cost, format_cost
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# Import API clients with error handling
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
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")

        if not aws_access_key or not aws_secret_key:
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in environment")

        return AnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region
        )

    elif api_type == "vllm":
        if OpenAI is None:
            raise ImportError("openai library not installed")
        return OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY"
        )

    else:
        raise ValueError(f"Invalid API type: {api_type}")

def load_evaluation_prompt_template() -> str:
    """Load the evaluation prompt template."""
    prompt_path = "prompts/unintended_behavior_target_eval.md"
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Evaluation prompt template not found at {prompt_path}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_benign_task_trajectory(execution_cua: str, domain: str, task_id: str, results_base_dir: str) -> Dict[str, Any]:
    """Load the benign task trajectory."""
    print(f"Loading benign task trajectory from {results_base_dir} for {execution_cua} {domain} {task_id}")
    trajectory_path = os.path.join(results_base_dir, execution_cua, domain, task_id, "response.json")
    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(f"Benign task trajectory not found at {trajectory_path}")

    with open(trajectory_path, 'r', encoding='utf-8') as f:
        benign_task_trajectory = json.load(f)
        return benign_task_trajectory[0]

def load_task_data(task_id: str, domain: str, initial_states_dir: str, examples_dir: str) -> Dict[str, Any]:
    """Load task data including original instruction and environment description."""
    # Paths
    state_dir = os.path.join(initial_states_dir, domain, task_id)
    task_file = os.path.join(examples_dir, domain, f"{task_id}.json")

    # Load task config
    if not os.path.exists(task_file):
        raise FileNotFoundError(f"Task file not found: {task_file}")

    with open(task_file, 'r') as f:
        task_config = json.load(f)

    # Load initial state description
    state_desc_path = os.path.join(state_dir, "initial_state_description.md")
    if not os.path.exists(state_desc_path):
        raise FileNotFoundError(f"Initial state description not found: {state_desc_path}")

    with open(state_desc_path, 'r', encoding='utf-8') as f:
        initial_state_description = f.read()

    return {
        "task_id": task_id,
        "domain": domain,
        "original_instruction": task_config.get("instruction", ""),
        "initial_state_description": initial_state_description
    }

def find_perturbed_queries(queries_dir: str, task_id: str, domain: str,
                          filter_model_dir: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Find all perturbed query files for a task.

    Expected structure: model_dir/perturbed_query_HASH/perturbed_query_HASH.json

    Args:
        queries_dir: Base perturbed queries directory
        task_id: Task ID to find queries for
        domain: Domain of the task
        filter_model_dir: Optional model directory name to filter by

    Returns:
        List of (model_dir, file_path) tuples
    """
    task_dir = os.path.join(queries_dir, domain, task_id)
    if not os.path.exists(task_dir):
        return []

    results = []

    # Iterate through model directories
    for model_dir in os.listdir(task_dir):
        model_path = os.path.join(task_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        # Skip if filter is specified and doesn't match
        if filter_model_dir and model_dir != filter_model_dir:
            continue

        # Look for nested structure (perturbed_query_HASH folders)
        for item in os.listdir(model_path):
            item_path = os.path.join(model_path, item)

            if os.path.isdir(item_path) and item.startswith("perturbed_query_"):
                # Find the perturbed query JSON inside the subfolder
                for filename in os.listdir(item_path):
                    if filename.startswith("perturbed_query_") and filename.endswith(".json") and "_evaluation" not in filename:
                        file_path = os.path.join(item_path, filename)
                        results.append((model_dir, file_path))

    return results

def format_evaluation_prompt(task_data: Dict[str, Any], unintended_behavior_target_data: Dict[str, Any], benign_task_trajectory: Dict[str, Any], template: str) -> str:
    """Format the unintended behavior target evaluation prompt with task and perturbation data."""    

    # Replace placeholders
    prompt = template.replace("{ORIGINAL_INSTRUCTION}", task_data["original_instruction"])
    prompt = prompt.replace("{ENV_DESCRIPTION}", task_data["initial_state_description"])
    prompt = prompt.replace("{BENIGN_TASK_TRAJECTORY}", json.dumps(benign_task_trajectory, indent=2))
    prompt = prompt.replace("{UNINTENDED_BEHAVIOR_TARGET}", unintended_behavior_target_data)

    return prompt

def call_llm_for_evaluation(api_client, hf_tokenizer, hf_model, api_type: str, prompt: str, model: str,
                            max_tokens: int = 4096, temperature: float = 0.3) -> Tuple[str, int, int, int]:
    """
    Call LLM to evaluate a perturbed query.

    Args:
        api_client: API client instance
        api_type: Type of API
        prompt: Formatted evaluation prompt
        model: Model name
        max_tokens: Maximum tokens for response
        temperature: Lower temperature for more consistent evaluation

    Returns:
        Tuple of (response_text, input_tokens, output_tokens, total_tokens)
    """
    if api_type in ["openai", "azure", "vllm"]:
        # For GPT-5-pro models
        if "gpt-5-pro" in model.lower():
            response = api_client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                        ],
                    }
                ],
                max_output_tokens=max_tokens,
                temperature=temperature,
            )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = response.usage.total_tokens
            response_text = response.output_text

            return response_text, input_tokens, output_tokens, total_tokens

        # For other GPT-5 and o4 models (use max_completion_tokens)
        elif "gpt-5" in model.lower() or "o4" in model.lower():
            response = api_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=max_tokens,
                temperature=temperature
            )

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            response_text = response.choices[0].message.content

            return response_text, input_tokens, output_tokens, total_tokens

        # For GPT-4 and other models (use max_tokens)
        else:
            response = api_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"}
            )

            if model == "Qwen/Qwen3-30B-A3B-Instruct-2507" or model == "Qwen/Qwen3-Next-80B-A3B-Instruct-2507":
                response_text = response.choices[0].message.reasoning_content
            else:
                response_text = response.choices[0].message.content

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            return response_text, input_tokens, output_tokens, total_tokens

    elif api_type in ["anthropic", "anthropic_bedrock"]:
        response = api_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        response_text = ""
        for event in response:
            if event.type == "content_block_delta":
                response_text += event.delta.text
            elif event.type == "message_stop":
                input_tokens = getattr(event, 'amazon-bedrock-invocationMetrics')['inputTokenCount']
                output_tokens = getattr(event, 'amazon-bedrock-invocationMetrics')['outputTokenCount']
                total_tokens = input_tokens + output_tokens
                break

        return response_text, input_tokens, output_tokens, total_tokens

    elif api_type == "huggingface":
        messages = [
            {"role": "user", "content": prompt},
        ]

        inputs = hf_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(hf_model.device)
        
        outputs = hf_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        response = hf_tokenizer.decode(outputs[0])
        print("Model response:", response.split("<|channel|>final<|message|>")[-1].strip())
        # TODO: Parse the output to remove <|return|> suffix
        raise ValueError("Stop here")

    else:
        raise ValueError(f"Unsupported API type: {api_type}")

def parse_evaluation_response(response_text: str) -> Dict[str, Any]:
    """Parse the LLM evaluation response to extract scores and justification."""
    try:
        # Extract JSON block between ```json and ```
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback: try to find any JSON object in the text
            match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            json_str = match.group(1) if match else None

        if not json_str:
            raise ValueError("No JSON found in response")

        parsed = json.loads(json_str)

        # Validate required fields
        required_fields = [
            "environment_feasibility_score",
            "contextual_plausibility_score",
            "harm_severity_score",
            "justification"
        ]

        for field in required_fields:
            if field not in parsed:
                print(f"    Warning: Missing field '{field}' in evaluation response")
                parsed[field] = 0 if field != "justification" else "Unable to parse"

        return parsed

    except json.JSONDecodeError as e:
        print(f"    Warning: Failed to parse JSON from evaluation response: {e}")
        return {
            "environment_feasibility_score": 0,
            "contextual_plausibility_score": 0,
            "harm_severity_score": 0,
            "justification": "Unable to parse evaluation response",
            "raw_response": response_text
        }
    else:
        print(f"    Warning: No JSON found in evaluation response")
        return {
            "environment_feasibility_score": 0,
            "contextual_plausibility_score": 0,
            "harm_severity_score": 0,
            "justification": "Unable to parse evaluation response",
            "raw_response": response_text
        }

def save_evaluation(unintended_behavior_target_file_path: str, evaluation_data: Dict[str, Any], evaluator_model: str):
    """Save evaluation data alongside the perturbed query file with evaluator model name.

    Args:
        unintended_behavior_target_file_path: Path to the unintended behavior target JSON file
        evaluation_data: Evaluation data to save
        evaluator_model: Name of the model used for evaluation

    Returns:
        Path to saved evaluation file
    """
    # Get the directory and base filename
    file_dir = os.path.dirname(unintended_behavior_target_file_path)
    base_filename = os.path.basename(unintended_behavior_target_file_path)

    # Remove .json extension
    base_name = os.path.splitext(base_filename)[0]

    # Clean model name for filename (replace special chars with underscores)
    clean_model_name = evaluator_model.replace(".", "_").replace(":", "_").replace("/", "_")

    # Create evaluation filename with model name
    eval_filename = f"{base_name}_unintended_behavior_target_evaluation_{clean_model_name}.json"
    eval_file_path = os.path.join(file_dir, eval_filename)

    with open(eval_file_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, indent=2, fp=f)

    return eval_file_path

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate unintended behavior targets"
    )

    # Task selection
    parser.add_argument("--task_id", type=str,
                       help="Specific task ID to evaluate")
    parser.add_argument("--domain", type=str, required=True,
                       help="Domain (os, chrome, vs_code, etc.)")
    parser.add_argument("--all", action="store_true",
                       help="Evaluate all tasks in the domain")

    
    # Directory configuration
    parser.add_argument("--queries_dir", type=str, default=os.path.join(PARENT_DIR, "perturbed_queries"),
                       help="Directory containing perturbed queries (default: ../perturbed_queries)")
    parser.add_argument("--initial_states_dir", type=str, default="initial_states",
                       help="Directory with initial state descriptions (default: ./initial_states)")
    parser.add_argument("--examples_dir", type=str, default=os.path.join(PARENT_DIR, "evaluation_examples", "examples"),
                       help="Directory with evaluation examples (default: ../evaluation_examples/examples)")
    parser.add_argument("--summary_dir", type=str, default=os.path.join(PARENT_DIR, "evaluation_summary"),
                       help="Directory for summary reports (default: ../evaluation_summary)")

    parser.add_argument("--results_base_dir", type=str, default=os.path.join(PARENT_DIR, "results", "pyautogui", "screenshot"),
                       help="Base directory for results (default: ../results/pyautogui/screenshot)")
    parser.add_argument('--execution_cua', type=str, default="claude-haiku-4-5-20251001", 
                       help='The CUA used to create the gold trajectory for the benign task instruction.')

    # Filtering options
    parser.add_argument("--filter_model_dir", type=str,
                       help="Only evaluate perturbations from this model directory (e.g., gpt-5-2025-08-07)")
    parser.add_argument("--skip_evaluated", action="store_true",
                       help="Skip perturbations that already have evaluation files")

    # API configuration
    parser.add_argument("--api", type=str, choices=["openai", "azure", "anthropic", "anthropic_bedrock", "huggingface", "vllm"],
                       default="anthropic",
                       help="API provider to use for evaluation (default: anthropic)")
    parser.add_argument("--model", type=str,
                       choices=[
                           # Models used in example_scripts
                           "gpt-5-2025-08-07",
                           "gpt-5-pro-2025-10-06",
                           "gpt-5-mini-2025-08-07",
                           "o4-mini-2025-04-16",
                           "us.anthropic.claude-sonnet-4-20250514-v1:0",
                           "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                           "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                           "us.anthropic.claude-opus-4-1-20250805-v1:0",
                           # Open Source models (e.g, vllm, huggingface)
                           "openai/gpt-oss-20b",
                           "Qwen/Qwen3-30B-A3B-Instruct-2507",
                           "Qwen/Qwen3-Next-80B-A3B-Instruct"
                       ],
                       default="o4-mini-2025-04-16",
                       help="Model name")
    parser.add_argument("--max_tokens", type=int, default=32768,
                       help="Maximum tokens for LLM response (default: 32768)")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for LLM sampling (default: 0.3 for consistent evaluation)")

    # Summary mode
    parser.add_argument("--summary_only", action="store_true",
                       help="Only generate summary reports, skip evaluation")

    args = parser.parse_args()

    # Determine which tasks to process
    if args.all:
        state_domain_dir = os.path.join(args.initial_states_dir, args.domain)
        if not os.path.exists(state_domain_dir):
            print(f"Error: Domain directory not found: {state_domain_dir}")
            return
        task_ids = [d for d in os.listdir(state_domain_dir)
                   if os.path.isdir(os.path.join(state_domain_dir, d))]
    else:
        if not args.task_id:
            print("Error: Must provide --task_id or --all")
            return
        task_ids = [args.task_id]

    print(f"Processing {len(task_ids)} tasks from domain '{args.domain}'...")
    print()

    # Initialize API client if not summary-only mode
    if not args.summary_only:
        if args.api == "huggingface":
            hf_tokenizer = AutoTokenizer.from_pretrained(args.model)
            hf_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype="auto",
                device_map="auto"
                # attn_implementation="flash_attention_2"
            )
            # print(hf_model.config._attn_implementation)
            api_client = None
        else:
            try:
                api_client = get_api_client(args.api)
                print(f"Using {args.api} API with model {args.model} for evaluation")
                hf_tokenizer = None
                hf_model = None
            except Exception as e:
                print(f"Error initializing API client: {e}")
                return

        # Load filter prompt template
        try:
            evaluation_prompt_template = load_evaluation_prompt_template()
            print(f"Loaded evaluation prompt template from prompts/unintended_behavior_target_eval.md")
        except Exception as e:
            print(f"Error loading evaluation prompt template: {e}")
            return
        print()

    # Process each task
    total_evaluated = 0
    total_skipped = 0
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for i, task_id in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] Processing task {task_id}...")

        # Load benign task trajectory
        try:
            benign_task_trajectory = load_benign_task_trajectory(args.execution_cua, args.domain, task_id, args.results_base_dir)
            print(f"Loaded benign task trajectory from {args.execution_cua}")
        except Exception as e:
            print(f"Error loading benign task trajectory: {e}")
            return
        print()

        try:
            # Load task data
            task_data = load_task_data(
                task_id, args.domain,
                args.initial_states_dir, args.examples_dir
            )

            # Find all perturbed queries for this task
            perturbed_files = find_perturbed_queries(
                args.queries_dir, task_id, args.domain,
                args.filter_model_dir
            )

            if not args.summary_only:
                # Evaluate each unintended behavior target
                task_evaluated = 0
                task_skipped = 0

                for model_dir, unintended_behavior_target_file in perturbed_files:
                    # Check if evaluation already exists (with this evaluator model)
                    file_dir = os.path.dirname(unintended_behavior_target_file)
                    base_filename = os.path.basename(unintended_behavior_target_file)
                    base_name = os.path.splitext(base_filename)[0]
                    clean_model_name = args.model.replace(".", "_").replace(":", "_").replace("/", "_")
                    eval_filename = f"{base_name}_unintended_behavior_target_evaluation_{clean_model_name}.json"
                    eval_file = os.path.join(file_dir, eval_filename)

                    if args.skip_evaluated and os.path.exists(eval_file):
                        print(f"  [{model_dir}] Skipping {os.path.basename(eval_file)} (already evaluated by {args.model})")
                        task_skipped += 1
                        continue

                    # Load unintended behavior target
                    with open(unintended_behavior_target_file, 'r') as f:
                        perturbed_query_data = json.load(f)
                        unintended_behavior_target_data = perturbed_query_data["plausible_unintended_behavior"]

                    print(f"  [{model_dir}] Evaluating unintended behavior target in {os.path.basename(unintended_behavior_target_file)}...")

                    evaluation_prompt = format_evaluation_prompt(task_data, unintended_behavior_target_data, benign_task_trajectory, evaluation_prompt_template)

                    # Call LLM for evaluation
                    try:
                        response_text, input_tokens, output_tokens, total_tokens = call_llm_for_evaluation(
                            api_client, hf_tokenizer, hf_model, args.api, evaluation_prompt, args.model,
                            args.max_tokens, args.temperature
                        )

                        # Parse evaluation response
                        evaluation = parse_evaluation_response(response_text)

                        # Add metadata
                        evaluation_data = {
                            "unintended_behavior_target_file": os.path.basename(unintended_behavior_target_file),
                            "unintended_behavior_target_id": perturbed_query_data.get("id", "unknown"),
                            "original_instruction": task_data["original_instruction"],
                            "plausible_unintended_behavior": unintended_behavior_target_data,
                            "environment_feasibility_score": evaluation["environment_feasibility_score"],
                            "contextual_plausibility_score": evaluation["contextual_plausibility_score"],
                            "harm_severity_score": evaluation["harm_severity_score"],
                            "justification": evaluation["justification"],
                            "evaluator_metadata": {
                                "model": args.model,
                                "api": args.api,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "total_tokens": total_tokens,
                                "cost": calculate_cost(args.model, input_tokens, output_tokens)
                            }
                        }

                        # Save evaluation with evaluator model name
                        eval_path = save_evaluation(unintended_behavior_target_file, evaluation_data, args.model)

                        # Update statistics
                        task_evaluated += 1
                        print(f"    ✓ Scores: E={evaluation['environment_feasibility_score']} C={evaluation['contextual_plausibility_score']} H={evaluation['harm_severity_score']}")

                        # Track costs
                        cost = calculate_cost(args.model, input_tokens, output_tokens)
                        total_cost += cost
                        total_input_tokens += input_tokens
                        total_output_tokens += output_tokens

                    except Exception as e:
                        print(f"    Error evaluating: {e}")
                        continue

                print(f"  Task summary: {task_evaluated} evaluated, {task_skipped} skipped")
                total_evaluated += task_evaluated
                total_skipped += task_skipped

        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue
        except Exception as e:
            print(f"  Error processing task: {e}")
            import traceback
            traceback.print_exc()
            continue

        print()

        # Print final summary
        print("=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        if not args.summary_only:
            print(f"Total evaluated: {total_evaluated}")
            print(f"Total skipped: {total_skipped}")
            print(f"Total cost: {format_cost(total_cost)}")
            print(f"Total tokens: {total_input_tokens + total_output_tokens:,} ({total_input_tokens:,} in, {total_output_tokens:,} out)")
        print(f"Summary reports saved to: {args.summary_dir}/{args.domain}/")


if __name__ == "__main__":
    main()
                    