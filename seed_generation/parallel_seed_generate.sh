#!/bin/bash

# Parallel generation script for perturbed queries
# Usage: ./parallel_seed_generate.sh [options]

# First Iteration:
# ./parallel_seed_generate.sh     \
#     --domain os     \
#     --num_perturbations 6     \
#     --enable_vs_multi     \
#     --batch_size 2     \
#     --execution_cua claude-haiku-4-5-20251001

# Iterative Generation:
# ./parallel_seed_generate.sh     \
#     --domain os     \
#     --num_perturbations 6     \
#     --enable_vs_multi     \
#     --batch_size 2     \
#     --execution_cua claude-haiku-4-5-20251001 \
#     --iterative_prompt \
#     --iteration_number {iteration_number}

set -e

# Default values
DOMAIN="os"
API="openai"
MODEL="o4-mini-2025-04-16"
OUTPUT_DIR="../perturbed_queries"
MAX_PARALLEL=10
EXAMPLES_DIR="../evaluation_examples/examples"
INITIAL_STATES_DIR="initial_states"
RESULTS_BASE_DIR="../results/pyautogui/screenshot"
LOG_DIR="../seed_generation_logs"

# Generation parameters
NUM_PERTURBATIONS=1
TEMPERATURE=1.0
MAX_TOKENS=32768
BATCH_SIZE=2

# Prompt flags
ITERATIVE_PROMPT=""
ITERATION_NUMBER=1
ENABLE_VS_MULTI=""
EXECUTION_CUA=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_perturbations)
            NUM_PERTURBATIONS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
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
        --initial_states_dir)
            INITIAL_STATES_DIR="$2"
            shift 2
            ;;
        --iteration_number)
            ITERATION_NUMBER="$2"
            shift 2
            ;;
        --iterative_prompt)
            ITERATIVE_PROMPT="--iterative_prompt"
            shift
            ;;
        --enable_vs_multi)
            ENABLE_VS_MULTI="--enable_vs_multi"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --domain DOMAIN              Domain to generate for (default: os)"
            echo "  --api API                    API provider (default: openai)"
            echo "  --model MODEL                Model name (default: o4-mini-2025-04-16)"
            echo "  --output_dir DIR             Output directory (default: ./perturbed_queries)"
            echo "  --num_perturbations N        Number of perturbations per task (default: 1)"
            echo "  --temperature T              Temperature for sampling (default: 1.0)"
            echo "  --max_tokens N               Max tokens for response (default: 32768)"
            echo "  --batch_size N               Batch size for VS-Multi (default: 2)"
            echo "  --execution_cua CUA          Execution CUA string"
            echo "                                 Example: 'claude-haiku-4-5-20251001'"
            echo "  --results_base_dir DIR       Results base directory (default: ../results/pyautogui/screenshot)"
            echo "  --max_parallel N             Max parallel jobs (default: 10)"
            echo "  --examples_dir DIR           Examples directory (default: ../evaluation_examples/examples)"
            echo "  --initial_states_dir DIR     Initial states directory (default: initial_states)"
            echo "  --iteration_number N         Iteration number for iterative prompt (default: 1)"
            echo ""
            echo "Prompt Flags:"
            echo "  --iterative_prompt           Use the iterative prompt"
            echo "  --enable_vs_multi            Enable VS-Multi mode for diversity"
            echo ""
            echo "  --help                       Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic generation"
            echo "  $0 --domain os --api openai --model gpt-5-2025-08-07 --num_perturbations 6"
            echo ""
            echo "  # With VS-Multi"
            echo "  $0 --domain os --api openai \\"
            echo "     --num_perturbations 6 --enable_vs_multi --batch_size 2 \\"
            echo "     --execution_cua 'claude-haiku-4-5-20251001'"
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

# Check if script exists
if [ ! -f "generate_perturbed_queries.py" ]; then
    echo "Error: generate_perturbed_queries.py not found in current directory"
    exit 1
fi

# Get the domain path
DOMAIN_PATH="${INITIAL_STATES_DIR}/${DOMAIN}"

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
RUN_LOG_DIR="${LOG_DIR}/generate_${TIMESTAMP}_${DOMAIN}"
mkdir -p "$RUN_LOG_DIR"

# Function to run generation for a single task
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
    
    # Build command
    local cmd="stdbuf -oL -eL python -u generate_perturbed_queries.py \
        --task_id \"$task_id\" \
        --domain \"$DOMAIN\" \
        --api \"$API\" \
        --model \"$MODEL\" \
        --output_dir \"$OUTPUT_DIR\" \
        --examples_dir \"$EXAMPLES_DIR\" \
        --initial_states_dir \"$INITIAL_STATES_DIR\" \
        --results_base_dir \"$RESULTS_BASE_DIR\" \
        --num_perturbations $NUM_PERTURBATIONS \
        --temperature $TEMPERATURE \
        --max_tokens $MAX_TOKENS \
        --batch_size $BATCH_SIZE \
        --iteration_number $ITERATION_NUMBER \
        $ITERATIVE_PROMPT \
        $ENABLE_VS_MULTI"
    
    # Add execution_cua if provided
    if [ -n "$EXECUTION_CUA" ]; then
        cmd="$cmd --execution_cua \"$EXECUTION_CUA\""
    fi
    
    # Execute the command with real-time output
    if eval "$cmd" 2>&1 | while IFS= read -r line; do
            # Write to log file
            echo "$line" >> "$log_file"
            # Write to stdout with prefix
            echo "[${task_id}] $line"
            
            # Extract cost information if present (handle variable whitespace)
            if [[ "$line" =~ Total\ cost:[[:space:]]+\$([0-9]+\.?[0-9]*) ]]; then
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
export DOMAIN API MODEL OUTPUT_DIR EXAMPLES_DIR INITIAL_STATES_DIR
export RESULTS_BASE_DIR NUM_PERTURBATIONS TEMPERATURE MAX_TOKENS BATCH_SIZE
export ITERATION_NUMBER ITERATIVE_PROMPT
export ENABLE_VS_MULTI EXECUTION_CUA RUN_LOG_DIR

# Print configuration summary
echo "Configuration:"
echo "  Domain:            $DOMAIN"
echo "  API:               $API"
echo "  Model:             $MODEL"
echo "  Output dir:        $OUTPUT_DIR"
echo "  Examples dir:      $EXAMPLES_DIR"
echo "  Num perturbations: $NUM_PERTURBATIONS"
echo "  Temperature:       $TEMPERATURE"
echo "  Max tokens:        $MAX_TOKENS"
if [ -n "$ENABLE_VS_MULTI" ]; then
    echo "  VS-Multi enabled:  Yes (batch size: $BATCH_SIZE)"
fi
if [ -n "$ITERATIVE_PROMPT" ]; then
    echo "  Iterative prompt:  Yes (iteration: $ITERATION_NUMBER)"
fi
if [ -n "$EXECUTION_CUA" ]; then
    echo "  Execution CUA:     $EXECUTION_CUA"
fi
echo "  Log dir:           $RUN_LOG_DIR"
echo ""
echo "Starting generation..."
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
echo "Generation Complete!"
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
echo "Generated queries saved to: $OUTPUT_DIR/$DOMAIN/"
echo ""
echo "To view a specific log:"
echo "  cat ${RUN_LOG_DIR}/<task_id>.log"
echo ""
echo "To view all failed logs:"
echo "  for f in ${RUN_LOG_DIR}/*.log; do if grep -q FAILED \${f%.log}.status 2>/dev/null; then echo \"=== \$f ===\"; cat \"\$f\"; echo; fi; done"