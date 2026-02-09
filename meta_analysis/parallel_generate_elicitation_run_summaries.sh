#!/bin/bash

# =============================================================================
# Generate Elicitation Run Summaries (Parallel Execution)
# =============================================================================
# This script calls elicitation_run_summary.py for each task in the task list.
# Supports the new JSON format with task_lists_by_refinement_model.
# Tasks are executed in parallel using GNU parallel (or xargs fallback).
# =============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Configuration
# =============================================================================

# Model configurations
PERTURBATION_MODEL="o4-mini-2025-04-16"
AGENT="claude-haiku-4-5-20251001"

# API configuration for summary generation
API="openai"
MODEL="gpt-5-2025-08-07"
MAX_TOKENS=32768
TEMPERATURE=1.0

# Parallel execution settings
MAX_PARALLEL=10
LOG_DIR="elicitation_summary_logs"

# Task list file (JSON format)
TASK_LIST_FILE=""

# Filter for specific refinement model (optional)
REFINEMENT_MODEL_FILTER=""

# =============================================================================
# Parse Command Line Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --task_list_file)
            TASK_LIST_FILE="$2"
            shift 2
            ;;
        --max_parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --perturbation_model)
            PERTURBATION_MODEL="$2"
            shift 2
            ;;
        --refinement_model_filter)
            REFINEMENT_MODEL_FILTER="$2"
            shift 2
            ;;
        --agent)
            AGENT="$2"
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
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --task_list_file <path> [options]"
            echo ""
            echo "Required:"
            echo "  --task_list_file FILE         Path to JSON file containing task list"
            echo "                                Supports format: {\"task_lists_by_refinement_model\": {...}}"
            echo ""
            echo "Options:"
            echo "  --max_parallel N              Max parallel jobs (default: 10)"
            echo "  --perturbation_model M        Perturbation model (default: o4-mini-2025-04-16)"
            echo "  --refinement_model_filter M   Only process tasks for this refinement model"
            echo "  --agent A                     Agent name (default: claude-haiku-4-5-20251001)"
            echo "  --api API                     API provider (default: openai)"
            echo "  --model MODEL                 Summary model (default: gpt-5-2025-08-07)"
            echo "  --max_tokens N                Max tokens (default: 32768)"
            echo "  --temperature T               Temperature (default: 1.0)"
            echo "  --help                        Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --task_list_file task_list_claude_haiku_0pct_baseline_human_filtered.json"
            echo "  $0 --task_list_file task_list.json --refinement_model_filter gpt-5-2025-08-07"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate Required Arguments and Load Task List
# =============================================================================
if [[ -z "$TASK_LIST_FILE" ]]; then
    echo "Error: --task_list_file is required"
    echo "Use --help for usage information"
    exit 1
fi

if [[ ! -f "$TASK_LIST_FILE" ]]; then
    echo "Error: Task list file not found: $TASK_LIST_FILE"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Create a timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Include agent name in directory for easier identification of multiple runs
RUN_LOG_DIR="${LOG_DIR}/summary_${AGENT}_${TIMESTAMP}"
mkdir -p "$RUN_LOG_DIR"

# =============================================================================
# Load task list from JSON file (supports both old and new formats)
# =============================================================================
echo "Loading task list from: $TASK_LIST_FILE"

# Python script to extract tasks with their refinement models
read -r -d '' PYTHON_EXTRACT_TASKS << 'PYTHON_EOF' || true
import json
import sys

task_list_file = sys.argv[1]
refinement_filter = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

with open(task_list_file, 'r') as f:
    data = json.load(f)

# Check format
if "task_details_by_refinement_model" in data:
    # New format with refinement model info
    for rm, tasks in data["task_details_by_refinement_model"].items():
        if refinement_filter and rm != refinement_filter:
            continue
        for task in tasks:
            # Output: domain:task_id:perturbed_id:refinement_model
            print(f"{task['domain']}:{task['task_id']}:{task['perturbed_id']}:{rm}")
elif "task_lists_by_refinement_model" in data:
    # Alternative format with just task specs
    for rm, task_specs in data["task_lists_by_refinement_model"].items():
        if refinement_filter and rm != refinement_filter:
            continue
        for spec in task_specs:
            # Output: spec:refinement_model
            print(f"{spec}:{rm}")
elif "task_list" in data:
    # Legacy format - requires refinement_model_filter
    if not refinement_filter:
        print("ERROR: Legacy format requires --refinement_model_filter", file=sys.stderr)
        sys.exit(1)
    for spec in data["task_list"]:
        print(f"{spec}:{refinement_filter}")
else:
    print("ERROR: Unknown task list format", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF

# Load tasks into array
TASK_LIST=()
while IFS= read -r task; do
    TASK_LIST+=("$task")
done < <(python3 -c "$PYTHON_EXTRACT_TASKS" "$TASK_LIST_FILE" "$REFINEMENT_MODEL_FILTER" 2>&1)

# Check for errors
if [[ ${#TASK_LIST[@]} -eq 0 ]]; then
    echo "Error: No tasks found in $TASK_LIST_FILE"
    exit 1
fi

if [[ "${TASK_LIST[0]}" == ERROR:* ]]; then
    echo "${TASK_LIST[0]}"
    exit 1
fi

echo "Loaded ${#TASK_LIST[@]} tasks from JSON file"
echo ""

# =============================================================================
# Function to run summary generation for a single task
# =============================================================================
run_task() {
    local task_spec=$1
    
    # Remove trailing comma if present
    task_spec="${task_spec%,}"
    
    # Skip empty lines or comments
    [[ -z "$task_spec" || "$task_spec" == \#* ]] && return 0
    
    # Parse domain, task_id, perturbed_id, refinement_model
    IFS=':' read -r domain task_id perturbed_id refinement_model <<< "$task_spec"
    
    # Sanity checks
    if [[ -z "$domain" || -z "$task_id" || -z "$perturbed_id" || -z "$refinement_model" ]]; then
        echo "[INVALID] Warning: invalid task format '$task_spec'"
        return 1
    fi
    
    # Create a unique identifier for logging
    local task_key="${task_id}_${perturbed_id}_${refinement_model##*_}"
    local log_file="${RUN_LOG_DIR}/${task_key}.log"
    local status_file="${RUN_LOG_DIR}/${task_key}.status"
    local cost_file="${RUN_LOG_DIR}/${task_key}.cost"
    
    # Create a header for both log and stdout
    local header="[${task_key}] START: $(date +%Y-%m-%d\ %H:%M:%S)"
    echo "$header"
    echo "$header" > "$log_file"
    echo "Domain: ${domain}" >> "$log_file"
    echo "Task ID: ${task_id}" >> "$log_file"
    echo "Perturbed ID: ${perturbed_id}" >> "$log_file"
    echo "Refinement Model: ${refinement_model}" >> "$log_file"
    echo "---" >> "$log_file"
    
    # Run the Python script
    if stdbuf -oL -eL python -u "${SCRIPT_DIR}/elicitation_run_summary.py" \
        --task_id "${task_id}" \
        --domain "${domain}" \
        --perturbed_id "${perturbed_id}" \
        --perturbation_model "${PERTURBATION_MODEL}" \
        --refinement_model "${refinement_model}" \
        --agent "${AGENT}" \
        --api "${API}" \
        --model "${MODEL}" \
        --max_tokens "${MAX_TOKENS}" \
        --temperature "${TEMPERATURE}" 2>&1 | while IFS= read -r line; do
            # Write to log file
            echo "$line" >> "$log_file"
            # Write to stdout with prefix
            echo "[${task_key}] $line"
        done; then
        
        echo "SUCCESS" > "$status_file"
        
        # Extract cost from the generated JSON file
        # Note: perturbed_queries is in parent directory (perturbation_generation), not meta_analysis_package
        local json_path="${SCRIPT_DIR}/../perturbed_queries/${domain}/${task_id}/${PERTURBATION_MODEL}/perturbed_query_${perturbed_id}/iterative_refinement_${refinement_model}/agent_${AGENT}/elicitation_run_summary.json"
        if [ -f "${json_path}" ]; then
            local task_cost=$(python3 -c "import json; print(json.load(open('${json_path}'))['cost'])" 2>/dev/null)
            if [ -n "${task_cost}" ]; then
                echo "${task_cost}" > "$cost_file"
                echo "[${task_key}] Cost: \$${task_cost}"
            fi
        fi
        
        local success_msg="[${task_key}] ✓ COMPLETED at $(date +%H:%M:%S)"
        echo "$success_msg"
        echo "$success_msg" >> "$log_file"
    else
        echo "FAILED" > "$status_file"
        local fail_msg="[${task_key}] ✗ FAILED at $(date +%H:%M:%S) - check $log_file"
        echo "$fail_msg"
        echo "$fail_msg" >> "$log_file"
    fi
    
    echo "---" >> "$log_file"
    echo "[${task_key}] END: $(date +%Y-%m-%d\ %H:%M:%S)" >> "$log_file"
}

export -f run_task
export SCRIPT_DIR PERTURBATION_MODEL AGENT API MODEL MAX_TOKENS TEMPERATURE RUN_LOG_DIR

# =============================================================================
# Main Execution
# =============================================================================

echo "=============================================="
echo "Generating Elicitation Run Summaries (Parallel)"
echo "=============================================="
echo "Task List File: ${TASK_LIST_FILE}"
echo "Perturbation Model: ${PERTURBATION_MODEL}"
echo "Refinement Model Filter: ${REFINEMENT_MODEL_FILTER:-All}"
echo "Agent: ${AGENT}"
echo "Summary API: ${API}"
echo "Summary Model: ${MODEL}"
echo "Max Parallel Jobs: ${MAX_PARALLEL}"
echo "Log Directory: ${RUN_LOG_DIR}"
echo "=============================================="
echo ""

# Filter out empty tasks and count
VALID_TASKS=()
for TASK in "${TASK_LIST[@]}"; do
    TASK="${TASK%,}"
    [[ -z "$TASK" || "$TASK" == \#* ]] && continue
    VALID_TASKS+=("$TASK")
done

# Count tasks by refinement model
echo "Tasks by refinement model:"
python3 -c "
import sys
from collections import Counter
tasks = sys.argv[1:]
models = [t.split(':')[-1] for t in tasks if t]
for model, count in sorted(Counter(models).items()):
    print(f'  - {model}: {count} tasks')
" "${VALID_TASKS[@]}"
echo ""

echo "Found ${#VALID_TASKS[@]} valid tasks to process"
echo "Running with max $MAX_PARALLEL parallel jobs"
echo ""
echo "Starting generation..."
echo ""

# Run tasks in parallel using GNU parallel if available, otherwise use xargs
if command -v parallel &> /dev/null; then
    # Using GNU parallel (recommended)
    printf "%s\n" "${VALID_TASKS[@]}" | parallel -j "$MAX_PARALLEL" run_task {}
else
    # Fallback to xargs with background processes
    echo "Note: GNU parallel not found, using xargs (install with: sudo apt-get install parallel)"
    printf "%s\n" "${VALID_TASKS[@]}" | xargs -P "$MAX_PARALLEL" -I {} bash -c 'run_task "$@"' _ {}
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Summary Generation Complete"
echo "=============================================="
echo ""

TOTAL=${#VALID_TASKS[@]}
SUCCESS_COUNT=$(find "$RUN_LOG_DIR" -name "*.status" -exec grep -l "SUCCESS" {} \; 2>/dev/null | wc -l)
FAILED_COUNT=$(find "$RUN_LOG_DIR" -name "*.status" -exec grep -l "FAILED" {} \; 2>/dev/null | wc -l)

# Calculate total cost with high precision using Python
TOTAL_COST=$(python3 -c "
from decimal import Decimal
import glob
total = Decimal('0')
for cost_file in glob.glob('$RUN_LOG_DIR/*.cost'):
    with open(cost_file) as f:
        try:
            total += Decimal(f.read().strip())
        except:
            pass
print(f'{total:.8f}')
" 2>/dev/null || echo "0.00000000")

echo "Summary:"
echo "  Total tasks:      $TOTAL"
echo "  Successful:       $SUCCESS_COUNT"
echo "  Failed:           $FAILED_COUNT"
echo "  Total cost:       \$$TOTAL_COST"
echo ""

# Save summary to JSON with agent and timestamp in filename
SUMMARY_JSON="${RUN_LOG_DIR}/run_summary_${AGENT}_${TIMESTAMP}.json"
python3 -c "
import json
import glob
from decimal import Decimal
from datetime import datetime

costs = {}
for cost_file in glob.glob('$RUN_LOG_DIR/*.cost'):
    task_key = cost_file.rsplit('/', 1)[-1].replace('.cost', '')
    with open(cost_file) as f:
        costs[task_key] = f.read().strip()

summary = {
    'timestamp': datetime.now().isoformat(),
    'task_list_file': '$TASK_LIST_FILE',
    'perturbation_model': '$PERTURBATION_MODEL',
    'agent': '$AGENT',
    'summary_api': '$API',
    'summary_model': '$MODEL',
    'statistics': {
        'total_tasks': $TOTAL,
        'successful': $SUCCESS_COUNT,
        'failed': $FAILED_COUNT,
        'total_cost': '$TOTAL_COST'
    },
    'per_task_costs': costs
}

with open('$SUMMARY_JSON', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Summary saved to: $SUMMARY_JSON')
"
echo ""

if [ "$FAILED_COUNT" -gt 0 ]; then
    echo "Failed tasks:"
    find "$RUN_LOG_DIR" -name "*.status" -exec grep -l "FAILED" {} \; 2>/dev/null | while read status_file; do
        task_key=$(basename "$status_file" .status)
        echo "  - $task_key (log: ${RUN_LOG_DIR}/${task_key}.log)"
    done
    echo ""
fi

echo "All logs saved to: $RUN_LOG_DIR"
echo ""
echo "To view a specific log:"
echo "  cat ${RUN_LOG_DIR}/<task_key>.log"
echo ""
echo "=============================================="

