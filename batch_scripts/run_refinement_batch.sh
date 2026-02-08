#!/bin/bash

# Generic Iterative Refinement Batch Script
# Runs refinement pipeline over a batch file of task_id:perturbed_id pairs.
# Supports custom batch files and hyperparameters.

set -e

usage() {
    cat << EOF
Usage: $0 BATCH_FILE [OPTIONS]

Required:
  BATCH_FILE    Path to batch file with one "task_id:perturbed_id" per line

Options (with defaults):
  --domain DOMAIN                    Domain name (default: os)
  --agent-model MODEL                Agent/executor model (default: claude-4-5-haiku-20251001)
  --refinement-model MODEL           Refinement model (default: gpt-5-2025-08-07)
  --refinement-api API               Refinement API provider (default: openai)
  --max-iterations N                 Max refinement iterations (default: 10)
  --max-quality-refinements N        Max quality refinements (default: 10)
  --progress-interval N              Print progress summary every N runs (default: 15)
  --perturbed-queries-dir DIR        Dir for perturbed queries (default: ../perturbed_queries)
  -h, --help                         Show this help

Example:
  $0 batch_scripts/my_batch.txt --domain my_domain --refinement-model gpt-4 --max-iterations 5
EOF
    exit 0
}

# Defaults
DOMAIN="os"
AGENT_MODEL="claude-4-5-haiku-20251001"
REFINEMENT_MODEL="gpt-5-2025-08-07"
REFINEMENT_API="openai"
MAX_ITERATIONS=10
MAX_QUALITY_REFINEMENTS=10
LOG_PREFIX="refinement_batch_part"
PROGRESS_INTERVAL=15
PERTURBED_QUERIES_DIR="../perturbed_queries"

# Parse positional argument
BATCH_FILE=""
if [ $# -eq 0 ]; then
    usage
fi

# First arg must be batch file (or --help)
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
fi

BATCH_FILE="$1"
shift

# Parse optional arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --agent_model)
            AGENT_MODEL="$2"
            shift 2
            ;;
        --refinement_model)
            REFINEMENT_MODEL="$2"
            shift 2
            ;;
        --refinement_api)
            REFINEMENT_API="$2"
            shift 2
            ;;
        --max_iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --max_quality_refinements)
            MAX_QUALITY_REFINEMENTS="$2"
            shift 2
            ;;
        --progress_interval)
            PROGRESS_INTERVAL="$2"
            shift 2
            ;;
        --perturbed_queries_dir)
            PERTURBED_QUERIES_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Change to parent directory (iterative_refinement.py expects to run from perturbation_generation/)
cd "$(dirname "$0")/.."

# Validate batch file
if [ ! -f "$BATCH_FILE" ]; then
    echo "Error: Batch file not found: $BATCH_FILE"
    exit 1
fi

# Extract log prefix from batch filename (basename without extension)
BATCH_BASENAME=$(basename "$BATCH_FILE")
LOG_PREFIX="${BATCH_BASENAME%.*}"

# Load tasks into array
mapfile -t TASKS < "$BATCH_FILE"

# Filter out empty lines for seed count
SEED_COUNT=0
for t in "${TASKS[@]}"; do
    [ -n "$t" ] && SEED_COUNT=$((SEED_COUNT + 1))
done

# Setup logging
LOG_DIR="batch_scripts/batch_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/${LOG_PREFIX}_${TIMESTAMP}.log"

echo "=========================================" | tee "$MAIN_LOG"
echo "Iterative Refinement Batch Run" | tee -a "$MAIN_LOG"
echo "Started: $(date)" | tee -a "$MAIN_LOG"
echo "=========================================" | tee -a "$MAIN_LOG"
echo "Batch File: $BATCH_FILE" | tee -a "$MAIN_LOG"
echo "Seeds: $SEED_COUNT" | tee -a "$MAIN_LOG"
echo "Domain: $DOMAIN" | tee -a "$MAIN_LOG"
echo "Agent Model: $AGENT_MODEL" | tee -a "$MAIN_LOG"
echo "Refinement Model: $REFINEMENT_MODEL ($REFINEMENT_API)" | tee -a "$MAIN_LOG"
echo "Max Iterations: $MAX_ITERATIONS" | tee -a "$MAIN_LOG"
echo "Max Quality Refinements: $MAX_QUALITY_REFINEMENTS" | tee -a "$MAIN_LOG"
echo "Base Directory: $PERTURBED_QUERIES_DIR/" | tee -a "$MAIN_LOG"
echo "=========================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Counters
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0
CURRENT=0
TOTAL_EXPECTED=$SEED_COUNT

# Run all seeds
for task_pair in "${TASKS[@]}"; do
    # Skip empty lines
    if [ -z "$task_pair" ]; then
        continue
    fi

    IFS=':' read -r task_id perturbed_id <<< "$task_pair"
    CURRENT=$((CURRENT + 1))

    echo "[$CURRENT/$TOTAL_EXPECTED] Task: $task_id | Query: $perturbed_id" | tee -a "$MAIN_LOG"

    # Check if already completed
    result_dir="${PERTURBED_QUERIES_DIR}/${DOMAIN}/${task_id}/*/perturbed_query_${perturbed_id}/iterative_refinement_${REFINEMENT_MODEL//\//_}/agent_${AGENT_MODEL//\//_}"

    if compgen -G "${result_dir}/refinement_metadata.json" > /dev/null 2>&1; then
        if grep -q '"status": "completed"' ${result_dir}/refinement_metadata.json 2>/dev/null; then
            echo "  ⏭  SKIPPED (already completed)" | tee -a "$MAIN_LOG"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi

    # Run refinement
    run_log="$LOG_DIR/${LOG_PREFIX}_${task_id}_${perturbed_id}_${TIMESTAMP}.log"
    start_time=$(date +%s)

    if python iterative_refinement/iterative_refinement.py \
        --task_id "$task_id" \
        --domain "$DOMAIN" \
        --perturbed_id "$perturbed_id" \
        --perturbed_queries_dir "$PERTURBED_QUERIES_DIR" \
        --agent_model "$AGENT_MODEL" \
        --refinement_model "$REFINEMENT_MODEL" \
        --refinement_api "$REFINEMENT_API" \
        --max_iterations "$MAX_ITERATIONS" \
        --max_quality_refinements "$MAX_QUALITY_REFINEMENTS" \
        > "$run_log" 2>&1; then

        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "  ✓ SUCCESS (${duration}s)" | tee -a "$MAIN_LOG"
        SUCCESS=$((SUCCESS + 1))
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "  ✗ FAILED (${duration}s) - See: $run_log" | tee -a "$MAIN_LOG"
        FAILED=$((FAILED + 1))
    fi

    TOTAL=$((TOTAL + 1))

    # Print periodic summary
    if [ "$PROGRESS_INTERVAL" -gt 0 ] && [ $((CURRENT % PROGRESS_INTERVAL)) -eq 0 ]; then
        echo "" | tee -a "$MAIN_LOG"
        echo "--- Progress Update ---" | tee -a "$MAIN_LOG"
        echo "Completed: $CURRENT/$TOTAL_EXPECTED" | tee -a "$MAIN_LOG"
        echo "Success: $SUCCESS | Failed: $FAILED | Skipped: $SKIPPED" | tee -a "$MAIN_LOG"
        if [ $((TOTAL - SKIPPED)) -gt 0 ]; then
            SUCCESS_RATE=$(awk "BEGIN {printf \"%.1f\", ($SUCCESS / ($TOTAL - $SKIPPED)) * 100}")
            echo "Success Rate: ${SUCCESS_RATE}%" | tee -a "$MAIN_LOG"
        fi
        echo "----------------------" | tee -a "$MAIN_LOG"
        echo "" | tee -a "$MAIN_LOG"
    fi
done

# Final summary
echo "" | tee -a "$MAIN_LOG"
echo "=========================================" | tee -a "$MAIN_LOG"
echo "FINAL SUMMARY - $LOG_PREFIX" | tee -a "$MAIN_LOG"
echo "=========================================" | tee -a "$MAIN_LOG"
echo "Completed: $(date)" | tee -a "$MAIN_LOG"
echo "Seeds: $SEED_COUNT" | tee -a "$MAIN_LOG"
echo "Total Runs: $TOTAL" | tee -a "$MAIN_LOG"
echo "Successful: $SUCCESS" | tee -a "$MAIN_LOG"
echo "Failed: $FAILED" | tee -a "$MAIN_LOG"
echo "Skipped: $SKIPPED" | tee -a "$MAIN_LOG"
if [ $((TOTAL - SKIPPED)) -gt 0 ]; then
    SUCCESS_RATE=$(awk "BEGIN {printf \"%.1f\", ($SUCCESS / ($TOTAL - $SKIPPED)) * 100}")
    echo "Success Rate: ${SUCCESS_RATE}%" | tee -a "$MAIN_LOG"
fi
echo "=========================================" | tee -a "$MAIN_LOG"
