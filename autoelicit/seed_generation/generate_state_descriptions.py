"""
Generate comprehensive initial state descriptions using an LLM.

Supports multiple APIs with credentials loaded from .env file:
- OpenAI (openai)
- Azure OpenAI (azure)
- Anthropic Direct (anthropic)
- Anthropic via AWS Bedrock (anthropic_bedrock)

Features:
- Automatic cost calculation and tracking using model_pricing.py
- Token usage reporting for each task
- Summary statistics with total and average costs

Usage:
    # Store API keys in .env file:
    # OPENAI_API_KEY=sk-...
    # ANTHROPIC_API_KEY=sk-ant-...
    # AZURE_API_KEY=...
    # AZURE_ENDPOINT=...
    # AZURE_API_VERSION=...
    # AWS_ACCESS_KEY=...
    # AWS_SECRET_KEY=...
    # AWS_REGION=us-east-1

    # Single task
    python generate_state_descriptions.py \
        --task_id 4d117223-a354-47fb-8b45-62ab1390a95f \
        --domain os 

    # All tasks
    python generate_state_descriptions.py \
        --domain os \
        --all 
"""

import argparse
import json
import os
import base64
import sys
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Add parent directory to Python path to import OSWorld modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.model_pricing import calculate_cost, format_cost

# Load environment variables from .env file
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
    """
    Get the appropriate API client based on the API type.

    Args:
        api_type: One of "openai", "azure", "anthropic", "anthropic_bedrock"

    Returns:
        API client instance
    """
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
        aws_access_key = os.getenv("AWS_ACCESS_KEY")
        aws_secret_key = os.getenv("AWS_SECRET_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")

        if not aws_access_key or not aws_secret_key:
            raise ValueError("AWS_ACCESS_KEY and AWS_SECRET_KEY must be set in environment")

        return AnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region
        )

    else:
        raise ValueError(f"Invalid API type: {api_type}")


def load_initial_state_description_prompt_template() -> str:
    """Load the initial state description prompt template."""
    # Use local prompts folder in perturbation_generation
    prompt_path = "prompts/comprehensive_state_description_prompt.md"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_task_data(task_id: str, domain: str, initial_states_dir: str, examples_dir: str) -> Dict[str, Any]:
    """Load all data for a task."""
    # Paths
    state_dir = os.path.join(initial_states_dir, domain, task_id)
    task_file = os.path.join(examples_dir, domain, f"{task_id}.json")

    # Load task config
    with open(task_file, 'r') as f:
        task_config = json.load(f)

    setup_sh_content = None
    setup_sh_url = None

    # Finds the setup.sh program in the environment setup config
    for step in task_config.get("config", []):
        if step.get("type") == "download":
            files = step.get("parameters", {}).get("files", [])
            for f in files:
                if f["path"] == "setup.sh":
                    setup_sh_url = f["url"]

    # Downloads the setup.sh program from the URL
    print(f"Setup.sh URL: {setup_sh_url}")
    if setup_sh_url:
        resp = requests.get(setup_sh_url)
        if resp.status_code == 200:
            setup_sh_content = resp.text
            print(f"Setup.sh content: {setup_sh_content}")
        else:
            setup_sh_content = f"[Failed to download setup.sh: HTTP {resp.status_code}]"

    # Load metadata
    metadata_path = os.path.join(state_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load SoM elements
    som_elements_path = os.path.join(state_dir, "initial_som_elements.txt")
    with open(som_elements_path, 'r') as f:
        som_elements = f.read()

    # Load screenshot (base64 encoded) - prefer plain screenshot by default
    screenshot_path = os.path.join(state_dir, "initial_screenshot.png")
    if not os.path.exists(screenshot_path):
        screenshot_path = os.path.join(state_dir, "initial_som_screenshot.png")

    with open(screenshot_path, 'rb') as f:
        screenshot_base64 = base64.standard_b64encode(f.read()).decode('utf-8')

    # Load a11y tree (optional, for reference)
    a11y_path = os.path.join(state_dir, "initial_a11y_tree.json")
    with open(a11y_path, 'r') as f:
        a11y_tree = f.read()

    return {
        "task_id": task_id,
        "domain": domain,
        "task_config": task_config,
        "setup_sh_content": setup_sh_content,
        "metadata": metadata,
        "som_elements": som_elements,
        "screenshot_base64": screenshot_base64,
        "a11y_tree": a11y_tree,
        "state_dir": state_dir
    }


def format_task_context(task_data: Dict[str, Any], include_som: bool = False) -> str:
    """Format task context information for the prompt."""
    task_config = task_data["task_config"]

    # Extract and format setup steps in human-readable way
    setup_steps = []
    for idx, step in enumerate(task_config.get("config", []), 1):
        step_type = step.get("type", "unknown")
        params = step.get("parameters", {})

        if step_type == "download":
            files = params.get("files", [])
            for file_info in files:
                filename = file_info.get('path', 'unknown file')
                setup_steps.append(f"{idx}. Downloaded file: '{filename}'")

        elif step_type == "execute":
            cmd = params.get("command", "")

            # Format command nicely
            if isinstance(cmd, list):
                # Handle Python commands specially
                if len(cmd) >= 3 and cmd[0] == "python" and cmd[1] == "-c":
                    python_code = cmd[2]
                    # Simplify common patterns
                    if "pyautogui" in python_code:
                        if "hotkey('ctrl', 'alt', 't')" in python_code:
                            setup_steps.append(f"{idx}. Opened Terminal application (Ctrl+Alt+T)")
                        elif "click(" in python_code:
                            setup_steps.append(f"{idx}. Clicked on desktop")
                        elif "write(" in python_code:
                            # Extract what was written
                            import re
                            write_match = re.search(r"write\('([^']+)'\)", python_code)
                            if write_match:
                                text = write_match.group(1)
                                setup_steps.append(f"{idx}. Typed command: '{text}'")
                        elif "press('enter')" in python_code:
                            setup_steps.append(f"{idx}. Pressed Enter key")
                        else:
                            setup_steps.append(f"{idx}. Executed PyAutoGUI automation")
                    else:
                        setup_steps.append(f"{idx}. Executed Python code")
                else:
                    cmd_str = " ".join(cmd)
                    setup_steps.append(f"{idx}. Executed: {cmd_str}")
            else:
                setup_steps.append(f"{idx}. Executed: {cmd}")

        elif step_type == "launch":
            cmd = params.get("command", [])
            if isinstance(cmd, list):
                app = cmd[0] if cmd else "application"
                if len(cmd) > 1:
                    setup_steps.append(f"{idx}. Launched {app} with file: {cmd[1]}")
                else:
                    setup_steps.append(f"{idx}. Launched {app}")
            else:
                setup_steps.append(f"{idx}. Launched: {cmd}")

        elif step_type == "activate_window":
            window = params.get("window_name", "")
            setup_steps.append(f"{idx}. Activated (focused) window: '{window}'")

        elif step_type == "open":
            path = params.get("path", "")
            setup_steps.append(f"{idx}. Opened: {path}")

        else:
            setup_steps.append(f"{idx}. {step_type}: {params}")

    # Format setup description
    setup_description = "\n".join(setup_steps) if setup_steps else "No setup steps were performed."

    setup_description += f"\n\nSetup.sh content: {task_data['setup_sh_content']}"

    # SoM section (optional)
    som_section = ""
    if include_som:
        som_section = f"""
### Interactive UI Elements

The screenshot has numbered red bounding boxes around interactive elements.
Here is the complete mapping:

```
{task_data['som_elements']}
```

Reference these element numbers (e.g., "Element #10") when describing UI components.
"""

    # Build the initial state description prompt
    initial_state_description_prompt_template = load_initial_state_description_prompt_template()

    # Replace placeholders
    full_initial_state_description_prompt = initial_state_description_prompt_template.replace("{SETUP_DESCRIPTION}", setup_description)
    full_initial_state_description_prompt = full_initial_state_description_prompt.replace("{DOMAIN}", task_data['domain'])
    full_initial_state_description_prompt = full_initial_state_description_prompt.replace("{TASK_ID}", task_data['task_id'])
    full_initial_state_description_prompt = full_initial_state_description_prompt.replace("{RELATED_APPS}", ', '.join(task_config.get('related_apps', [])))
    full_initial_state_description_prompt = full_initial_state_description_prompt.replace("{SOM_SECTION}", som_section)

    return full_initial_state_description_prompt


def generate_description(api_client, api_type: str, task_data: Dict[str, Any], model: str, max_tokens: int = 4096, temperature: float = 0.0, include_som: bool = False) -> str:
    """
    Generate description using the specified API client.

    Args:
        api_client: API client instance
        api_type: Type of API ("openai", "azure", "anthropic", "anthropic_bedrock")
        task_data: Task data dictionary
        model: Model name
        max_tokens: Maximum tokens for response
        temperature: Sampling temperature
        include_som: Include Set-of-Mark elements

    Returns:
        Generated description string
    """
    # Get full prompt
    full_initial_state_description_prompt = format_task_context(task_data, include_som=include_som)    # Call appropriate API
    if api_type in ["openai", "azure"]:
        # GPT-5-pro models use the /v1/responses endpoint
        if "gpt-5-pro" in model.lower():
            response = api_client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": full_initial_state_description_prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{task_data['screenshot_base64']}"
                            },
                        ],
                    }
                ],
                max_output_tokens=max_tokens,  # renamed from max_completion_tokens
                temperature=temperature,
            )

            # Extract token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = response.usage.total_tokens

            # Extract text output
            initial_state_description_assistant_content = response.output_text  # convenient helper property
            
            return initial_state_description_assistant_content, input_tokens, output_tokens, total_tokens


        # For other GPT-5 and o4 models, use max_completion_tokens instead of max_tokens
        elif "gpt-5" in model.lower() or "o4" in model.lower():
            response = api_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": full_initial_state_description_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{task_data['screenshot_base64']}"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=max_tokens,
                temperature=temperature
            )

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            initial_state_response = response.choices[0].message.content

            return initial_state_response, input_tokens, output_tokens, total_tokens
        else:
            response = api_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": full_initial_state_description_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{task_data['screenshot_base64']}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Extract token usage
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            initial_state_response = response.choices[0].message.content


            return initial_state_response, input_tokens, output_tokens, total_tokens

    elif api_type in ["anthropic", "anthropic_bedrock"]:
        initial_state_description_response = api_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": task_data["screenshot_base64"],
                            },
                        },
                        {
                            "type": "text",
                            "text": full_initial_state_description_prompt
                        }
                    ]
                }
            ]
        )

        initial_state_description_response_text = ""
        for event in initial_state_description_response:
            if event.type == "content_block_delta":
                initial_state_description_response_text += event.delta.text
            elif event.type == "message_stop":
                input_tokens = getattr(event, 'amazon-bedrock-invocationMetrics')['inputTokenCount']
                output_tokens = getattr(event, 'amazon-bedrock-invocationMetrics')['outputTokenCount']
                total_tokens = input_tokens + output_tokens
                break

        return initial_state_description_response_text, input_tokens, output_tokens, total_tokens

    else:
        raise ValueError(f"Unknown API type: {api_type}")


def save_description(task_data: Dict[str, Any], initial_state_description: str) -> str:
    """Save the generated description to file."""
    initial_state_description_output_path = os.path.join(task_data["state_dir"], "initial_state_description.md")

    with open(initial_state_description_output_path, 'w', encoding='utf-8') as f:
        f.write(initial_state_description)

    return initial_state_description_output_path

def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive initial state descriptions"
    )

    # Task selection
    parser.add_argument("--task_id", type=str, help="Single task ID to process")
    parser.add_argument("--domain", type=str, required=True, help="Task domain")
    parser.add_argument("--all", action="store_true", help="Process all tasks in domain")

    # Directories
    parser.add_argument("--initial_states_dir", type=str, default="../initial_states",
                       help="Directory with captured initial states (default: ../initial_states/)")
    parser.add_argument("--examples_dir", type=str, default="../../evaluation_examples/examples",
                       help="Directory with task JSON files (default: ../../evaluation_examples/examples)")

    # API configuration
    parser.add_argument("--api", type=str,
                       choices=["openai", "azure", "anthropic", "anthropic_bedrock"],
                       required=True,
                       help="Which API to use")
    parser.add_argument("--model", type=str,
                       choices=[
                           # Models used in example_scripts
                           "gpt-4o",
                           "gpt-4o-2024-08-06",
                           "gpt-4.1-2025-04-14",
                           "gpt-5-2025-08-07",
                           "gpt-5-pro-2025-10-06",
                           "gpt-5-nano-2025-08-07",
                           "o4-mini-2025-04-16",
                           "us.anthropic.claude-sonnet-4-20250514-v1:0",
                           # Additional common models
                           "gpt-4o-2024-11-20",
                           "gpt-4o-mini",
                           "o1-preview",
                           "o1-mini",
                           "gpt-4-turbo",
                           "claude-3-5-sonnet-20241022",
                           "claude-3-7-sonnet-20250219",
                           "claude-3-5-haiku-20241022",
                       ],
                       default="gpt-5-nano-2025-08-07",
                       help="Model name")
    parser.add_argument("--max_tokens", type=int, default=32768,
                       help="Maximum tokens for LLM response (default: 32768)")
    parser.add_argument("--temperature", type=float, default=1,
                       help="Temperature for LLM sampling (default: 0.0 for deterministic output)")
    parser.add_argument("--include_som", action="store_true",
                       help="Include Set-of-Mark elements in the prompt (default: False)")

    args = parser.parse_args()

    # Get API client
    try:
        api_client = get_api_client(args.api)
        print(f"Using {args.api} API with model {args.model}")
    except Exception as e:
        print(f"Error initializing API client: {e}")
        print("Please check your .env file has the required credentials:")
        if args.api == "openai":
            print("  - OPENAI_API_KEY")
        elif args.api == "azure":
            print("  - AZURE_API_KEY")
            print("  - AZURE_ENDPOINT")
            print("  - AZURE_API_VERSION (optional)")
        elif args.api == "anthropic":
            print("  - ANTHROPIC_API_KEY")
        elif args.api == "anthropic_bedrock":
            print("  - AWS_ACCESS_KEY")
            print("  - AWS_SECRET_KEY")
            print("  - AWS_REGION")
        return

    # Determine which tasks to process
    if args.all:
        state_domain_dir = os.path.join(args.initial_states_dir, args.domain)
        task_ids = [d for d in os.listdir(state_domain_dir)
                   if os.path.isdir(os.path.join(state_domain_dir, d))]
    else:
        if not args.task_id:
            print("Error: Must provide --task_id or --all")
            return
        task_ids = [args.task_id]

    print(f"Processing {len(task_ids)} tasks from domain '{args.domain}'...")
    print(f"SoM included: {args.include_som}")
    print()

    # Process each task
    successful = 0
    failed = 0
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for i, task_id in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] Processing {task_id}...")

        try:
            # Load task data
            task_data = load_task_data(
                task_id, args.domain,
                args.initial_states_dir, args.examples_dir
            )

            # Generate description
            print(f"  Calling {args.api} API...")
            initial_state_description, input_tokens, output_tokens, total_tokens = generate_description(
                api_client, args.api, task_data, args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                include_som=args.include_som
            )
            initial_state_description_output_path = save_description(task_data, initial_state_description)

            # Calculate cost
            cost = calculate_cost(args.model, input_tokens, output_tokens)
            total_cost += cost
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            print(f"  ✓ Saved initial state description: {initial_state_description_output_path}")
            print(f"  ℹ Tokens: {input_tokens:,} in / {output_tokens:,} out / {total_tokens:,} total")
            print(f"  ℹ Cost: {format_cost(cost)}")
            successful += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
            continue

        print()

    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tasks:  {len(task_ids)}")
    print(f"Successful:   {successful}")
    print(f"Failed:       {failed}")
    print(f"Descriptions saved to: initial_states/{args.domain}/*/initial_state_description.md")
    if successful > 0:
        print()
        print(f"Token usage:")
        print(f"  Input:      {total_input_tokens:,}")
        print(f"  Output:     {total_output_tokens:,}")
        print(f"  Total:      {total_input_tokens + total_output_tokens:,}")
        print(f"Total cost:   {format_cost(total_cost)}")
        if successful > 1:
            avg_cost = total_cost / successful
            print(f"Average cost: {format_cost(avg_cost)} per task")
    print("="*60)


if __name__ == "__main__":
    main()
