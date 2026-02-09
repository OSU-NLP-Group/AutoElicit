"""
Cluster elicitation run categories using an LLM (GPT-5) based on the category_clustering prompt.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to Python path to import OSWorld modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

sys.path.insert(0, PARENT_DIR)

from utils.model_pricing import calculate_cost, format_cost

load_dotenv()

# Import API clients
try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    print("Warning: openai library not installed. Install with: pip install openai")
    OpenAI = None
    AzureOpenAI = None

try:
    from anthropic import Anthropic, AnthropicBedrock
except ImportError:
    print("Warning: anthropic library not installed. Install with: pip install anthropic")
    Anthropic = None
    AnthropicBedrock = None


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


def call_llm(client, api_type: str, model_name: str, prompt: str, max_tokens: int, temperature: float):
    """
    Call the LLM with the given prompt.
    
    Returns:
        Tuple of (response_text, input_tokens, output_tokens, total_tokens)
    """
    if api_type in ["openai", "azure"]:
        # gpt-5-pro models use the responses API instead of chat completions
        if "gpt-5-pro" in model_name.lower():
            response = client.responses.create(
                model=model_name,
                input=prompt,
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens
            response_text = response.output_text
        # Other gpt-5 and o4 models use chat completions with max_completion_tokens
        elif "gpt-5" in model_name.lower() or "o4" in model_name.lower():
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                temperature=temperature
            )
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            response_text = response.choices[0].message.content
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            response_text = response.choices[0].message.content

    elif api_type in ["anthropic", "anthropic_bedrock"]:
        response = client.messages.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens
        response_text = response.content[0].text

    return response_text, input_tokens, output_tokens, total_tokens


def parse_json_response(response_text: str) -> dict:
    """Parse JSON from LLM response with error handling."""
    # Look for JSON object in the response
    json_match = re.search(r'\{[\s\S]*\}', response_text)

    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            return parsed
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON: {e}")
            print(f"Response text (first 500 chars): {response_text[:500]}")
            return {"error": f"JSON decode error: {e}", "raw_response": response_text}
    else:
        print(f"Warning: No JSON found in response")
        print(f"Response text (first 500 chars): {response_text[:500]}")
        return {"error": "No JSON found in response", "raw_response": response_text}


def format_categories_for_prompt(categories: list) -> str:
    """Format categories as a string for the prompt."""
    formatted = ""
    for i, cat in enumerate(categories, 1):
        formatted += f"\n### Category {i}: {cat['category_name']}\n"
        formatted += f"**Definition:** {cat['definition']}\n"
        formatted += f"**Examples ({len(cat.get('examples', []))}):**\n"
        
        for j, example in enumerate(cat.get("examples", [])[:5], 1):  # Show up to 5 examples
            formatted += f"  {j}. Trigger: \"{example.get('trigger_phrase', 'N/A')}\"\n"
            formatted += f"     Justification: {example.get('justification', 'N/A')}\n"
        
        if len(cat.get("examples", [])) > 5:
            formatted += f"  ... and {len(cat['examples']) - 5} more examples\n"
        
        formatted += "\n"
    
    return formatted


def load_clustering_prompt(categories_text: str) -> str:
    """Load and populate the clustering prompt."""
    prompt_file = os.path.join(PARENT_DIR, "meta_analysis", "prompts", "category_clustering.md")
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    prompt = prompt.replace("{BENIGN_INPUT_VULNERABILITY_CATEGORIES}", categories_text)
    return prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster elicitation run categories using an LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with GPT-5
    python llm_category_clustering.py \\
        --input_path elicitation_run_categorization/.../categorization.json

    # Use a different model
    python llm_category_clustering.py \\
        --input_path elicitation_run_categorization/.../categorization.json \\
        --model gpt-5-pro-2025-10-06
        """
    )

    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to categorization JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (defaults to clusters_{model_name} in same dir as input)')

    # API configuration
    parser.add_argument("--api", type=str, 
                       choices=["openai", "azure", "anthropic", "anthropic_bedrock"],
                       default="openai",
                       help="API provider (default: openai)")
    parser.add_argument("--model", type=str,
                       choices=[
                           "gpt-5-2025-08-07",
                           "gpt-5-pro-2025-10-06",
                           "gpt-5-mini-2025-08-07",
                           "o4-mini-2025-04-16",
                           "us.anthropic.claude-sonnet-4-20250514-v1:0",
                           "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                           "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                           "us.anthropic.claude-opus-4-1-20250805-v1:0",
                       ],
                       default="gpt-5-2025-08-07",
                       help="Model name for clustering (default: gpt-5-2025-08-07)")
    parser.add_argument("--max_tokens", type=int, default=64000,
                       help="Maximum tokens for LLM response (default: 32768)")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for LLM sampling (default: 1.0)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("LLM-based Category Clustering")
    print("=" * 70)
    print(f"Input: {args.input_path}")
    print(f"Model: {args.model}")
    print(f"API: {args.api}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print("=" * 70)

    # Load the categorization data
    print("\nLoading categorization data...")
    with open(args.input_path, 'r') as f:
        data = json.load(f)

    categories = data["categorization"]["categories"]
    print(f"Loaded {len(categories)} categories")
    print(f"Total examples: {sum(len(c.get('examples', [])) for c in categories)}")

    # Format categories for the prompt
    print("\nFormatting categories for prompt...")
    categories_text = format_categories_for_prompt(categories)

    # Load the clustering prompt
    print("Loading clustering prompt...")
    prompt = load_clustering_prompt(categories_text)
    print(f"Prompt length: {len(prompt)} characters")

    # Get API client
    print(f"\nInitializing {args.api} client...")
    client = get_api_client(args.api)

    # Call the LLM
    print(f"Calling {args.model} for clustering...")
    response_text, input_tokens, output_tokens, total_tokens = call_llm(
        client, args.api, args.model, prompt, args.max_tokens, args.temperature
    )
    print(f"Response received: {len(response_text)} characters")

    # Parse the response
    print("\nParsing JSON response...")
    clustering_result = parse_json_response(response_text)

    # Calculate cost
    cost = calculate_cost(args.model, input_tokens, output_tokens)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create clusters_{model_name} folder in the same directory as input
        input_dir = Path(args.input_path).parent
        # Clean model name for folder (replace special chars)
        model_folder_name = args.model.replace(":", "_").replace("/", "_").replace(".", "_")
        output_dir = input_dir / f"clusters_{model_folder_name}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"llm_clusters_{timestamp}.json"

    # Build output data
    output_data = {
        "clusters": clustering_result.get("clusters", []),
        "statistics": {
            "input_categories": len(categories),
            "output_clusters": len(clustering_result.get("clusters", [])),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": format_cost(cost)
        },
        "metadata": {
            "input_path": str(Path(args.input_path).resolve()),
            "api_type": args.api,
            "model_name": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "generated_at": datetime.now().isoformat()
        }
    }

    # Handle parse errors
    if "error" in clustering_result:
        output_data["parse_error"] = clustering_result["error"]
        output_data["raw_response"] = clustering_result.get("raw_response", response_text)

    # Save output
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    # Print summary
    print("\n" + "=" * 70)
    print("CLUSTERING COMPLETE")
    print("=" * 70)
    print(f"Input categories: {len(categories)}")
    print(f"Output clusters: {len(clustering_result.get('clusters', []))}")
    print(f"Input tokens: {input_tokens:,}")
    print(f"Output tokens: {output_tokens:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Cost: {format_cost(cost)}")
    print(f"\nOutput saved to: {output_path}")

    # Print cluster summary
    if "clusters" in clustering_result:
        print("\n--- Cluster Summary ---")
        for i, cluster in enumerate(clustering_result["clusters"], 1):
            member_count = len(cluster.get("member_categories", []))
            print(f"{i}. {cluster.get('cluster_name', 'Unnamed')} ({member_count} members)")

    print("=" * 70)


if __name__ == "__main__":
    main()
