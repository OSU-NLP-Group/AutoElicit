#!/bin/bash

# Generate Summary Report for Iterative Refinement Experiments
# This script runs the Python summary generator with configurable parameters

set -e

# Default values
DOMAIN="os"
PERTURBATION_MODEL="o4-mini-2025-04-16"
REFINEMENT_MODEL="gpt-5-2025-08-07"
AGENT="claude-haiku-4-5-20251001"
NUM_PARTS=5
SEED_PREFIX="refinement_batch_part"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --perturbation_model)
            PERTURBATION_MODEL="$2"
            shift 2
            ;;
        --refinement_model)
            REFINEMENT_MODEL="$2"
            shift 2
            ;;
        --agent_model)
            AGENT="$2"
            shift 2
            ;;
        --num_parts)
            NUM_PARTS="$2"
            shift 2
            ;;
        --seed_prefix)
            SEED_PREFIX="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Generate a summary report for iterative refinement experiments."
            echo ""
            echo "Options:"
            echo "  --domain DOMAIN              Domain name (default: os)"
            echo "  --perturbation_model MODEL   Perturbation model (default: o4-mini-2025-04-16)"
            echo "  --refinement_model MODEL     Refinement model identifier"
            echo "                               (default: us_anthropic_claude-haiku-4-5-20251001-v1_0)"
            echo "  --agent_model AGENT_MODEL    Execution agent (default: claude-haiku-4-5-20251001)"
            echo "  --num_parts N                Number of seed file parts (default: 5)"
            echo "  --seed_prefix PREFIX         Prefix for seed files (default: refinement_batch_part)"
            echo "  --help, -h                   Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --domain os --agent_model claude-haiku-4-5-20251001"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Change to script directory
cd "$(dirname "$0")"

echo "========================================="
echo "Generating Run Summary"
echo "Started: $(date)"
echo "========================================="
echo "Domain: $DOMAIN"
echo "Perturbation Model: $PERTURBATION_MODEL"
echo "Refinement Model: $REFINEMENT_MODEL"
echo "Agent: $AGENT"
echo "Seed File Parts: $NUM_PARTS"
echo "========================================="
echo ""

python3 generate_summary.py \
    --domain "$DOMAIN" \
    --perturbation_model "$PERTURBATION_MODEL" \
    --refinement_model "$REFINEMENT_MODEL" \
    --agent_model "$AGENT_MODEL" \
    --num_parts "$NUM_PARTS" \
    --seed_prefix "$SEED_PREFIX"

echo ""
echo "========================================="
echo "Summary Generation Complete"
echo "Completed: $(date)"
echo "========================================="
