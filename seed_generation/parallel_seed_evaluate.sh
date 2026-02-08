#!/bin/bash

# Parallel evaluation script for perturbed queries
# Usage: ./parallel_seed_evaluate.sh [options]

# Perturbed Queries Evaluation Usage: 
# ./parallel_seed_evaluate.sh     \
#     --eval_script evaluate_perturbed_queries.py  \
#     --domain os    \
#     --model gpt-5-2025-08-07     \
#     --max_parallel 20

# Unintended Behavior Targets Evaluation Usage: 
# ./parallel_seed_evaluate.sh     \
#     --eval_script evaluate_unintended_behavior_targets.py  \
#     --domain os    \
#     --model gpt-5-2025-08-07     \
#     --execution_cua claude-haiku-4-5-20251001 \
#     --max_parallel 20


set -e

# Default values
EVAL_SCRIPT="evaluate_perturbed_queries.py"
DOMAIN="os"
API="openai"
MODEL="gpt-5-2025-08-07"
QUERIES_DIR="../perturbed_queries"
SKIP_EVALUATED="--skip_evaluated"
MAX_PARALLEL=10
EXAMPLES_DIR="../evaluation_examples/examples"
LOG_DIR="../seed_evaluation_logs"
EXECUTION_CUA=""
RESULTS_BASE_DIR="../results/pyautogui/screenshot"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --eval_script)
            EVAL_SCRIPT="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --api)
            API="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --queries_dir)
            QUERIES_DIR="$2"
            shift 2
            ;;
        --execution_cua)
            EXECUTION_CUA="$2"
            shift 2
            ;;
        --results_base_dir)
            RESULTS_BASE_DIR="$2"
            shift 2
            ;;
        --max_parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --examples_dir)
            EXAMPLES_DIR="$2"
            shift 2
            ;;
        --no_skip_evaluated)
            SKIP_EVALUATED=""
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --eval_script SCRIPT         Evaluation script to run:"
            echo "                                 - evaluate_perturbed_queries.py (default)"
            echo "                                 - evaluate_unintended_behavior_targets.py"
            echo "  --domain DOMAIN              Domain to evaluate (default: os)"
            echo "  --api API                    API provider (default: openai)"
            echo "  --model MODEL                Model name (default: gpt-5-2025-08-07)"
            echo "  --queries_dir DIR            Queries directory (default: perturbed_queries_revised)"
            echo "  --execution_cua CUA          Execution CUA (for unintended_behavior_targets only)"
            echo "                                 Example: 'aws | us.anthropic.claude-sonnet-4-20250514-v1:0 | cua'"
            echo "  --results_base_dir DIR       Results base directory (default: results/pyautogui/screenshot)"
            echo "  --max_parallel N             Max parallel jobs (default: 10)"
            echo "  --examples_dir DIR           Examples directory (default: evaluation_examples/examples)"
            echo "  --no_skip_evaluated          Don't skip already evaluated tasks"
            echo "  --help                       Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Evaluate perturbed queries"
            echo "  $0 --eval_script evaluate_perturbed_queries.py --domain os --api openai --model gpt-5-2025-08-07"
            echo ""
            echo "  # Evaluate unintended behavior targets"
            echo "  $0 --eval_script evaluate_unintended_behavior_targets.py --domain os --api openai --model gpt-5-2025-08-07 \\"
            echo "     --execution_cua 'aws | us.anthropic.claude-sonnet-4-20250514-v1:0 | cua'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# Validate evaluation script
if [[ "$EVAL_SCRIPT" != "evaluate_perturbed_queries.py" && "$EVAL_SCRIPT" != "evaluate_unintended_behavior_targets.py" ]]; then
    echo "Error: Invalid evaluation script. Must be either:"
    echo "  - evaluate_perturbed_queries.py"
    echo "  - evaluate_unintended_behavior_targets.py"
    exit 1
fi

# Check if script exists
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

# Validate execution_cua for unintended_behavior_targets
if [[ "$EVAL_SCRIPT" == "evaluate_unintended_behavior_targets.py" && -z "$EXECUTION_CUA" ]]; then
    echo "Error: --execution_cua is required when using evaluate_unintended_behavior_targets.py"
    echo "Example: --execution_cua 'aws | us.anthropic.claude-sonnet-4-20250514-v1:0 | cua'"
    exit 1
fi

# Get the domain path
DOMAIN_PATH="${QUERIES_DIR}/${DOMAIN}"

# Check if domain directory exists
if [ ! -d "$DOMAIN_PATH" ]; then
    echo "Error: Domain directory not found: $DOMAIN_PATH"
    exit 1
fi

# Get all task IDs (subdirectories in the domain folder)
echo "Scanning for task IDs in: $DOMAIN_PATH"
TASK_IDS=($(find "$DOMAIN_PATH" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort))

if [ ${#TASK_IDS[@]} -eq 0 ]; then
    echo "Error: No task IDs found in $DOMAIN_PATH"
    exit 1
fi

echo "Found ${#TASK_IDS[@]} task IDs in domain '$DOMAIN'"
echo "Task IDs: ${TASK_IDS[@]}"
echo "Running with max $MAX_PARALLEL parallel jobs"
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Create a timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG_DIR="${LOG_DIR}/run_${TIMESTAMP}_${DOMAIN}"
mkdir -p "$RUN_LOG_DIR"

# Function to run evaluation for a single task
run_task() {
    local task_id=$1
    local log_file="${RUN_LOG_DIR}/${task_id}.log"
    local status_file="${RUN_LOG_DIR}/${task_id}.status"
    local cost_file="${RUN_LOG_DIR}/${task_id}.cost"
    
    # Create a header for both log and stdout
    local header="[${task_id}] START: $(date +%Y-%m-%d\ %H:%M:%S)"
    echo "$header"
    echo "$header" > "$log_file"
    echo "---" >> "$log_file"
    
    # Build command based on evaluation script
    local cmd="stdbuf -oL -eL python -u $EVAL_SCRIPT \
        --task_id \"$task_id\" \
        --domain \"$DOMAIN\" \
        --api \"$API\" \
        --model \"$MODEL\" \
        --queries_dir \"$QUERIES_DIR\" \
        --examples_dir \"$EXAMPLES_DIR\" \
        $SKIP_EVALUATED"
    
    # Add script-specific arguments
    if [[ "$EVAL_SCRIPT" == "evaluate_unintended_behavior_targets.py" ]]; then
        cmd="$cmd --execution_cua \"$EXECUTION_CUA\" --results_base_dir \"$RESULTS_BASE_DIR\""
    fi
    
    # Execute the command with real-time output
    if eval "$cmd" 2>&1 | while IFS= read -r line; do
            # Write to log file
            echo "$line" >> "$log_file"
            # Write to stdout with prefix
            echo "[${task_id}] $line"
            
            # Extract cost information if present
            if [[ "$line" =~ Total\ cost:\ \$([0-9]+\.[0-9]+) ]]; then
                echo "${BASH_REMATCH[1]}" > "$cost_file"
            fi
        done; then
        echo "SUCCESS" > "$status_file"
        local success_msg="[${task_id}] ✓ COMPLETED SUCCESSFULLY at $(date +%H:%M:%S)"
        echo "$success_msg"
        echo "$success_msg" >> "$log_file"
    else
        echo "FAILED" > "$status_file"
        local fail_msg="[${task_id}] ✗ FAILED at $(date +%H:%M:%S) - check $log_file"
        echo "$fail_msg"
        echo "$fail_msg" >> "$log_file"
    fi
    
    echo "---" >> "$log_file"
    echo "[${task_id}] END: $(date +%Y-%m-%d\ %H:%M:%S)" >> "$log_file"
}

export -f run_task
export EVAL_SCRIPT DOMAIN API MODEL QUERIES_DIR SKIP_EVALUATED EXAMPLES_DIR RUN_LOG_DIR EXECUTION_CUA RESULTS_BASE_DIR

# Print configuration summary
echo "Configuration:"
echo "  Eval script:  $EVAL_SCRIPT"
echo "  Domain:       $DOMAIN"
echo "  API:          $API"
echo "  Model:        $MODEL"
echo "  Queries dir:  $QUERIES_DIR"
echo "  Examples dir: $EXAMPLES_DIR"
if [[ "$EVAL_SCRIPT" == "evaluate_unintended_behavior_targets.py" ]]; then
    echo "  Execution CUA: $EXECUTION_CUA"
    echo "  Results dir:   $RESULTS_BASE_DIR"
fi
echo "  Skip eval:    $([ -n "$SKIP_EVALUATED" ] && echo "Yes" || echo "No")"
echo "  Log dir:      $RUN_LOG_DIR"
echo ""
echo "Starting evaluation..."
echo ""

# Run tasks in parallel using GNU parallel if available, otherwise use xargs
if command -v parallel &> /dev/null; then
    # Using GNU parallel (recommended)
    printf "%s\n" "${TASK_IDS[@]}" | parallel -j "$MAX_PARALLEL" run_task {}
else
    # Fallback to xargs with background processes
    echo "Note: GNU parallel not found, using xargs (install with: sudo apt-get install parallel)"
    printf "%s\n" "${TASK_IDS[@]}" | xargs -P "$MAX_PARALLEL" -I {} bash -c 'run_task "$@"' _ {}
fi

# Generate summary report
echo ""
echo "================================"
echo "Evaluation Complete!"
echo "================================"
echo ""

TOTAL=${#TASK_IDS[@]}
SUCCESS_COUNT=$(find "$RUN_LOG_DIR" -name "*.status" -exec grep -l "SUCCESS" {} \; 2>/dev/null | wc -l)
FAILED_COUNT=$(find "$RUN_LOG_DIR" -name "*.status" -exec grep -l "FAILED" {} \; 2>/dev/null | wc -l)

# Calculate total cost
TOTAL_COST=0
if command -v bc &> /dev/null; then
    for cost_file in "$RUN_LOG_DIR"/*.cost; do
        if [ -f "$cost_file" ]; then
            cost=$(cat "$cost_file")
            TOTAL_COST=$(echo "$TOTAL_COST + $cost" | bc)
        fi
    done
    TOTAL_COST_FORMATTED=$(printf "%.4f" $TOTAL_COST)
else
    # Fallback if bc is not available - use awk
    for cost_file in "$RUN_LOG_DIR"/*.cost; do
        if [ -f "$cost_file" ]; then
            cost=$(cat "$cost_file")
            TOTAL_COST=$(awk -v a="$TOTAL_COST" -v b="$cost" 'BEGIN {printf "%.4f", a + b}')
        fi
    done
    TOTAL_COST_FORMATTED=$TOTAL_COST
fi

echo "Summary:"
echo "  Total tasks:      $TOTAL"
echo "  Successful:       $SUCCESS_COUNT"
echo "  Failed:           $FAILED_COUNT"
echo "  Total cost:       \$$TOTAL_COST_FORMATTED"
echo ""

# Show per-task costs if available
if ls "$RUN_LOG_DIR"/*.cost &>/dev/null; then
    echo "Per-task costs:"
    for cost_file in "$RUN_LOG_DIR"/*.cost; do
        if [ -f "$cost_file" ]; then
            task_id=$(basename "$cost_file" .cost)
            cost=$(cat "$cost_file")
            printf "  %-20s \$%.4f\n" "$task_id:" "$cost"
        fi
    done
    echo ""
fi

if [ $FAILED_COUNT -gt 0 ]; then
    echo "Failed tasks:"
    find "$RUN_LOG_DIR" -name "*.status" -exec grep -l "FAILED" {} \; 2>/dev/null | while read status_file; do
        task_id=$(basename "$status_file" .status)
        echo "  - $task_id (log: ${RUN_LOG_DIR}/${task_id}.log)"
    done
    echo ""
fi

echo "All logs saved to: $RUN_LOG_DIR"
echo ""
echo "To view a specific log:"
echo "  cat ${RUN_LOG_DIR}/<task_id>.log"
echo ""
echo "To view all failed logs:"
echo "  for f in ${RUN_LOG_DIR}/*.log; do if grep -q FAILED \${f%.log}.status 2>/dev/null; then echo \"=== \$f ===\"; cat \"\$f\"; echo; fi; done"