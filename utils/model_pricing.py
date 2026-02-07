# Pricing: (input_price_per_1M, output_price_per_1M)
MODEL_PRICING = {
    # OpenAI GPT-5 family (October 2025)
    "gpt-5-2025-08-07": (1.25, 10.00),
    "gpt-5-pro-2025-10-06": (15, 120),
    "gpt-5-mini-2025-08-07": (0.25, 2.00),
    "gpt-5-nano-2025-08-07": (0.05, 0.40),

    # OpenAI GPT-4o models (latest published rates October 2025)
    "gpt-4o": (5.00, 15.00),
    "gpt-4o-2024-11-20": (5.00, 15.00),
    "gpt-4o-2024-08-06": (5.00, 15.00),
    "gpt-4o-2024-05-13": (5.00, 15.00),
    "gpt-4o-mini": (0.60, 2.40),
    "gpt-4o-mini-2024-07-18": (0.60, 2.40),

    # OpenAI o1 models
    "o1-preview": (15.00, 60.00),
    "o1-preview-2024-09-12": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o1-mini-2024-09-12": (3.00, 12.00),

    # OpenAI GPT-4 Turbo
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4-turbo-2024-04-09": (10.00, 30.00),
    "gpt-4-turbo-preview": (10.00, 30.00),
    "gpt-4-0125-preview": (10.00, 30.00),
    "gpt-4-1106-preview": (10.00, 30.00),

    # OpenAI GPT-4, GPT-4.1 (latest, historical rates)
    "gpt-4": (30.00, 60.00),
    "gpt-4-0613": (30.00, 60.00),
    "gpt-4.1-2025-04-14": (3.00, 12.00),

    # OpenAI o4-mini
    "o4-mini-2025-04-16": (1.00, 4.00),

    # OpenAI CUA models
    "computer-use-preview": (3.00, 12.00),

    # Anthropic Claude 4 family (Updated December 2025)
    # Claude Opus 4.5: $5.00 input / $25.00 output per MTok
    "claude-opus-4-5-20251101": (5.00, 25.00),
    "us.anthropic.claude-opus-4-5-20251101-v1:0": (5.00, 25.00),

    # Claude Opus 4.1: $15.00 input / $75.00 output per MTok
    "claude-opus-4-1-20250805": (15.00, 75.00),
    "us.anthropic.claude-opus-4-1-20250805-v1:0": (15.00, 75.00),

    # Claude Sonnet 4.5: $3.00 input / $15.00 output per MTok
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.00, 15.00),

    # Claude Sonnet 4: $3.00 input / $15.00 output per MTok
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "us.anthropic.claude-sonnet-4-20250514-v1:0": (3.00, 15.00),

    # Claude Haiku 4.5: $1.00 input / $5.00 output per MTok
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (1.00, 5.00),

    # Claude Opus 4.6: $5.00 input / $25.00 output per MTok
    "claude-opus-4-6": (5.00, 25.00),
    "us.anthropic.claude-opus-4-6-v1": (5.00, 25.00),

    # Anthropic Claude 3 family (legacy)
    "claude-3-7-sonnet-20250219": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.25, 1.25),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-sonnet-20240229": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": (3.00, 15.00),
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": (0.25, 1.25),
    "anthropic.claude-3-opus-20240229-v1:0": (15.00, 75.00),
    "anthropic.claude-3-sonnet-20240229-v1:0": (3.00, 15.00),
    "anthropic.claude-3-haiku-20240307-v1:0": (0.25, 1.25),
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost for a given model and token usage.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    if model not in MODEL_PRICING:
        # Return 0 for unknown models
        return 0.0

    input_price, output_price = MODEL_PRICING[model]

    # Calculate cost (prices are per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price

    return input_cost + output_cost

def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"
