#!/bin/bash

# Large-Scale Benign Task Reproducibility Analysis 
# Discovers all successful elicitations and tests reproducibility of their
# corresponding BENIGN (original) tasks to establish baseline behavior.
# Shows live progress updates during execution with colored output

set -e

# Change to perturbation_generation directory
cd "$(dirname "$0")/.."

# ============================================================================
# CONFIGURATION
# ============================================================================

DOMAIN="multi_apps"
BASE_DIR="perturbed_queries"
N_TRIALS=5  # Number of reproducibility trials per case

# Agent filter - only test these agents (leave empty array to test all agents)
AGENT_FILTER=("agent_claude-haiku-4-5-20251001" "agent_computer-use-preview")

# AWS Configuration
PROVIDER_NAME="aws"
REGION="us-east-1"

# ============================================================================
# LOGGING SETUP
# ============================================================================

LOG_DIR="batch_scripts/batch_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/benign_reproducibility_batch_${DOMAIN}_benign_${TIMESTAMP}.log"

# ============================================================================
# COLORS FOR TERMINAL OUTPUT
# ============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

is_agent_filtered() {
    local agent_name=$1
    
    # If filter is empty, allow all agents
    if [ ${#AGENT_FILTER[@]} -eq 0 ]; then
        return 0  # true - agent is allowed
    fi
    
    # Check if agent is in the filter list
    for filtered_agent in "${AGENT_FILTER[@]}"; do
        if [ "$agent_name" == "$filtered_agent" ]; then
            return 0  # true - agent is allowed
        fi
    done
    
    return 1  # false - agent is not allowed
}

print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}${BOLD}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

print_separator() {
    echo -e "${BLUE}───────────────────────────────────────────────────────────────${NC}"
}

log_both() {
    echo -e "$1" | tee -a "$MAIN_LOG"
}

log_only() {
    echo "$1" >> "$MAIN_LOG"
}

# ============================================================================
# START
# ============================================================================

print_header "Benign Task Reproducibility Analysis - $DOMAIN Domain"
log_both "Started: $(date)"
log_both ""
log_both "${CYAN}Configuration:${NC}"
log_both "  • Domain: $DOMAIN"
log_both "  • Base Directory: $BASE_DIR"
log_both "  • Trials per Case: $N_TRIALS"
log_both "  • Test Type: ${MAGENTA}BENIGN TASKS${NC}"
log_both "  • Provider: $PROVIDER_NAME (Region: $REGION)"
if [ ${#AGENT_FILTER[@]} -gt 0 ]; then
    log_both "  • ${YELLOW}Agent Filter: ${AGENT_FILTER[*]}${NC}"
else
    log_both "  • Agent Filter: ${GREEN}ALL AGENTS${NC}"
fi
log_both ""
print_separator

# ============================================================================
# DISCOVER ALL SUCCESSFUL CASES AND EXTRACT UNIQUE (task_id, agent_model) PAIRS
# ============================================================================

echo ""
log_both "${CYAN}${BOLD}Phase 1: Discovering Unique Benign Tasks${NC}"
log_both ""
log_both "Searching for final_successful_query.json files and extracting unique (task_id, agent_model) pairs..."

# Find all successful cases and extract unique (task_id, agent_model) pairs
# We only need ONE benign test per unique combination, not one per successful elicitation
declare -A UNIQUE_TASK_AGENTS
declare -A TASK_AGENT_CASE_PATH  # Store a representative case path for each unique pair

while IFS= read -r case_path; do
    agent_dir=$(dirname "$case_path")
    task_id=$(echo "$agent_dir" | awk -F'/' '{print $(NF-4)}')
    agent_name=$(basename "$agent_dir")

    # Check if agent passes filter
    if ! is_agent_filtered "$agent_name"; then
        FILTERED_OUT_COUNT=$((FILTERED_OUT_COUNT + 1))
        continue
    fi
    
    # Create a unique key for (task_id, agent_model) pair
    unique_key="${task_id}|${agent_name}"
    
    # Only store if we haven't seen this combination before
    if [ -z "${UNIQUE_TASK_AGENTS[$unique_key]}" ]; then
        UNIQUE_TASK_AGENTS[$unique_key]=1
        TASK_AGENT_CASE_PATH[$unique_key]="$agent_dir"
    fi
done < <(find "$BASE_DIR/$DOMAIN" -name "final_successful_query.json" -type f 2>/dev/null | sort)

TOTAL_UNIQUE=${#UNIQUE_TASK_AGENTS[@]}

if [ $TOTAL_UNIQUE -eq 0 ]; then
    log_both "${RED}Error: No successful cases found in $BASE_DIR/$DOMAIN${NC}"
    if [ ${#AGENT_FILTER[@]} -gt 0 ]; then
        log_both "${YELLOW}Note: Agent filter is active. Filtered agents: ${AGENT_FILTER[*]}${NC}"
    fi
    exit 1
fi

log_both "${GREEN}✓ Found $TOTAL_UNIQUE unique (task_id, agent_model) pairs for benign testing${NC}"
if [ ${#AGENT_FILTER[@]} -gt 0 ]; then
    log_both "${YELLOW}  (Filtered for agents: ${AGENT_FILTER[*]})${NC}"
    log_both "${YELLOW}  (Filtered out: $FILTERED_OUT_COUNT combinations)${NC}"
fi
log_both ""

# ============================================================================
# COUNT UNIQUE TASK IDs PER AGENT
# ============================================================================

echo ""
log_both "${CYAN}${BOLD}Unique Task IDs per Agent:${NC}"
log_both ""

# Create associative arrays to track task IDs per agent
declare -A AGENT_TASK_COUNTS
declare -A AGENT_TASKS

# Count unique task IDs for each agent
for unique_key in "${!UNIQUE_TASK_AGENTS[@]}"; do
    IFS='|' read -r task_id agent_name <<< "$unique_key"
    
    # Initialize agent entry if not exists
    if [ -z "${AGENT_TASK_COUNTS[$agent_name]}" ]; then
        AGENT_TASK_COUNTS[$agent_name]=0
        AGENT_TASKS[$agent_name]=""
    fi
    
    # Increment count for this agent
    AGENT_TASK_COUNTS[$agent_name]=$((AGENT_TASK_COUNTS[$agent_name] + 1))
    
    # Store task ID (for verification if needed)
    if [ -z "${AGENT_TASKS[$agent_name]}" ]; then
        AGENT_TASKS[$agent_name]="$task_id"
    else
        AGENT_TASKS[$agent_name]="${AGENT_TASKS[$agent_name]},$task_id"
    fi
done

# Display counts sorted by agent name
for agent_name in $(echo "${!AGENT_TASK_COUNTS[@]}" | tr ' ' '\n' | sort); do
    count=${AGENT_TASK_COUNTS[$agent_name]}
    log_both "  • ${agent_name}: ${GREEN}${count}${NC} unique tasks"
done

log_both ""
print_separator

# ============================================================================
# ANALYZE CASES TO FIND WHAT NEEDS TESTING
# ============================================================================

log_both "${CYAN}${BOLD}Phase 2: Checking Existing Benign Reproducibility Results${NC}"
log_both ""
log_both "Checking: $BASE_DIR/\$DOMAIN/\$task_id/benign_execution/\$agent_model/benign_reproducibility_analysis/"

PENDING_KEYS=()
COMPLETED_KEYS=()

for unique_key in "${!UNIQUE_TASK_AGENTS[@]}"; do
    IFS='|' read -r task_id agent_name <<< "$unique_key"
    
    repro_summary_path="$BASE_DIR/$DOMAIN/$task_id/benign_execution/$agent_name/benign_reproducibility_analysis/benign_reproducibility_summary.json"

    if [ -f "$repro_summary_path" ]; then
        # Check if test is complete (has all n_trials)
        completed=$(python3 -c "
import json
try:
    with open('$repro_summary_path') as f:
        data = json.load(f)
        if data.get('n_trials', 0) >= $N_TRIALS:
            print('yes')
        else:
            print('no')
except:
    print('no')
" 2>/dev/null || echo "no")

        if [ "$completed" == "yes" ]; then
            COMPLETED_KEYS+=("$unique_key")
        else
            PENDING_KEYS+=("$unique_key")
        fi
    else
        PENDING_KEYS+=("$unique_key")
    fi
done

TOTAL_PENDING=${#PENDING_KEYS[@]}
TOTAL_COMPLETED=${#COMPLETED_KEYS[@]}

log_both "  • Total Unique (task_id, agent) Pairs: $TOTAL_UNIQUE"
log_both "  • Already Completed: ${YELLOW}$TOTAL_COMPLETED${NC}"
log_both "  • Pending/Incomplete: ${CYAN}$TOTAL_PENDING${NC}"
log_both ""

# List all unique pairs
log_both "${CYAN}Unique Benign Tasks to Evaluate:${NC}"
for unique_key in "${!UNIQUE_TASK_AGENTS[@]}"; do
    IFS='|' read -r task_id agent_name <<< "$unique_key"
    repro_summary_path="$BASE_DIR/$DOMAIN/$task_id/benign_execution/$agent_name/benign_reproducibility_analysis/benign_reproducibility_summary.json"
    if [ -f "$repro_summary_path" ]; then
        log_both "  ✓ ${task_id:0:8}...${task_id: -8} | $agent_name (completed)"
    else
        log_both "  ○ ${task_id:0:8}...${task_id: -8} | $agent_name (pending)"
    fi
done
log_both ""

if [ $TOTAL_PENDING -eq 0 ]; then
    print_header "All Benign Tasks Already Tested!"
    log_both ""
    log_both "${GREEN}All $TOTAL_UNIQUE unique (task_id, agent_model) pairs have completed benign reproducibility tests.${NC}"
    log_both "${YELLOW}To re-test, delete existing benign_reproducibility_analysis/ directories.${NC}"
    log_both ""
    exit 0
fi

# ============================================================================
# ASK FOR CONFIRMATION
# ============================================================================

print_separator
log_both ""
log_both "${YELLOW}${BOLD}Ready to test $TOTAL_PENDING unique benign tasks${NC}"
log_both "${YELLOW}Estimated time: ~$(echo "$TOTAL_PENDING * $N_TRIALS * 2.5 / 60" | bc) hours (assuming ~2.5 min/trial)${NC}"
log_both ""
read -p "$(echo -e ${CYAN}Continue? [y/N]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_both "Aborted by user"
    exit 0
fi

# ============================================================================
# RUN BENIGN REPRODUCIBILITY TESTS
# ============================================================================

echo ""
print_header "Phase 3: Running Benign Task Reproducibility Tests"
log_both ""

SUCCESS_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0
CURRENT=0

START_TIME=$(date +%s)

for unique_key in "${PENDING_KEYS[@]}"; do
    CURRENT=$((CURRENT + 1))

    # Extract task_id and agent_name from unique key
    IFS='|' read -r task_id agent_name <<< "$unique_key"
    
    # Get a representative case path for this unique pair
    agent_dir="${TASK_AGENT_CASE_PATH[$unique_key]}"

    # Progress header
    echo ""
    print_separator
    log_both "${BOLD}[$CURRENT/$TOTAL_PENDING] Progress: $(awk "BEGIN {printf \"%.1f\", ($CURRENT / $TOTAL_PENDING) * 100}")%${NC}"
    print_separator
    log_both "${CYAN}Task ID:${NC}    ${task_id}"
    log_both "${CYAN}Agent:${NC}      $agent_name"
    log_both "${MAGENTA}Test Type:${NC}  BENIGN TASK (Original Instruction)"
    log_both ""

    log_only "=== [$CURRENT/$TOTAL_PENDING] Task: $task_id | Agent: $agent_name | Type: BENIGN ==="

    # Run benign reproducibility test
    case_log="$LOG_DIR/benign_task_${task_id}_${agent_name}_${TIMESTAMP}.log"
    trial_start=$(date +%s)

    log_both "${CYAN}Running $N_TRIALS benign task trials...${NC}"

    if python reproducibility_testing_package/run_reproducibility_test.py \
        --success_case_path "$agent_dir" \
        --n_trials $N_TRIALS \
        --provider_name "$PROVIDER_NAME" \
        --region "$REGION" \
        --skip_existing \
        --benign \
        > "$case_log" 2>&1; then

        trial_end=$(date +%s)
        trial_duration=$((trial_end - trial_start))

        # Extract results from the correct benign_reproducibility_summary.json location
        repro_summary="$BASE_DIR/$DOMAIN/$task_id/benign_execution/$agent_name/benign_reproducibility_analysis/benign_reproducibility_summary.json"

        if [ -f "$repro_summary" ]; then
            # Parse results using UNSAFE/SAFE classification
            results=$(python3 -c "
    import json
    try:
        with open('$repro_summary') as f:
            data = json.load(f)
            stats = data['statistics']
            task_completed = stats['task_completion_count']
            total = stats['task_completion_count'] + stats['task_failure_count']
            task_completion_rate = stats['task_completion_rate'] * 100
            avg_task_score = stats['avg_task_score']
            std_task_score = stats['std_task_score']
            unsafe_count = stats['unsafe_count']
            safe_count = stats['safe_count']
            baseline_harm_rate = stats['baseline_harm_rate'] * 100
            total_cost = stats['total_cost']
            print(f'{task_completed}|{total}|{task_completion_rate:.1f}|{avg_task_score:.2f}|{std_task_score:.2f}|{unsafe_count}|{safe_count}|{baseline_harm_rate:.1f}|{total_cost:.2f}')
    except Exception as e:
        print('?|?|0.0|0.0|0.0|0|0|0.0|0.00')
    " 2>/dev/null || echo "?|?|0.0|0.0|0.0|0|0|0.0|0.00")

            IFS='|' read -r task_completed_trials total_trials task_completion_rate avg_task_score std_task_score unsafe_count safe_count baseline_harm_rate total_cost <<< "$results"

            log_both "  ${GREEN}✓ SUCCESS${NC} (${trial_duration}s)"
            log_both "    • Trials Completed: ${task_completed_trials}/${total_trials}"
            log_both "    • Task Completion Rate: ${GREEN}${task_completion_rate}%${NC}"
            log_both "    • Avg Task Score: ${avg_task_score} ± ${std_task_score}"
            log_both "    • UNSAFE Trials: ${unsafe_count} (${YELLOW}${baseline_harm_rate}%${NC})"
            log_both "    • SAFE Trials: ${safe_count}"
            log_both "    • Cost: \$${total_cost}"
            
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            # Summary file not found - check the log for what happened
            log_both "  ${YELLOW}⚠ WARNING${NC} (${trial_duration}s)"
            log_both "    • Python script succeeded but summary file not found"
            log_both "    • Expected: $repro_summary"
            log_both "    • Check log for details: $case_log"
            
            # Show last few lines of log for quick diagnosis
            log_both ""
            log_both "    ${CYAN}Last 10 lines of log:${NC}"
            tail -n 10 "$case_log" | while IFS= read -r line; do
                log_both "      $line"
            done
            
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        fi
    else
        trial_end=$(date +%s)
        trial_duration=$((trial_end - trial_start))

        log_both "  ${RED}✗ FAILED${NC} (${trial_duration}s)"
        log_both "    • Check log: $case_log"

        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi

    # Progress summary every 5 cases or at end
    if [ $((CURRENT % 5)) -eq 0 ] || [ $CURRENT -eq $TOTAL_PENDING ]; then
        echo ""
        print_separator
        log_both "${CYAN}${BOLD}Progress Summary${NC}"
        print_separator
        log_both "  Completed: ${CURRENT}/${TOTAL_PENDING}"
        log_both "  ${GREEN}Success: ${SUCCESS_COUNT}${NC} | ${RED}Failed: ${FAILED_COUNT}${NC}"

        if [ $((SUCCESS_COUNT + FAILED_COUNT)) -gt 0 ]; then
            success_rate=$(awk "BEGIN {printf \"%.1f\", ($SUCCESS_COUNT / ($SUCCESS_COUNT + $FAILED_COUNT)) * 100}")
            log_both "  Success Rate: ${success_rate}%"
        fi

        # Time estimate
        current_time=$(date +%s)
        elapsed=$((current_time - START_TIME))

        if [ $CURRENT -gt 0 ]; then
            avg_time_per_case=$((elapsed / CURRENT))
            remaining_cases=$((TOTAL_PENDING - CURRENT))
            est_remaining_seconds=$((remaining_cases * avg_time_per_case))
            est_hours=$((est_remaining_seconds / 3600))
            est_minutes=$(((est_remaining_seconds % 3600) / 60))

            log_both "  Elapsed: $((elapsed / 3600))h $(((elapsed % 3600) / 60))m"

            if [ $est_hours -gt 0 ]; then
                log_both "  Est. Remaining: ~${est_hours}h ${est_minutes}m"
            else
                log_both "  Est. Remaining: ~${est_minutes}m"
            fi
        fi

        print_separator
        echo ""
    fi
done

# ============================================================================
# FINAL SUMMARY
# ============================================================================

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
print_header "FINAL SUMMARY - BENIGN TASK REPRODUCIBILITY"
log_both ""
log_both "Completed: $(date)"
log_both "Total Duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
log_both ""
log_both "${CYAN}Unique Benign Tasks Tested:${NC}"
log_both "  • Total Unique (task_id, agent) Pairs: $TOTAL_PENDING"
log_both "  • ${GREEN}Success: ${SUCCESS_COUNT}${NC}"
log_both "  • ${RED}Failed: ${FAILED_COUNT}${NC}"

if [ $((SUCCESS_COUNT + FAILED_COUNT)) -gt 0 ]; then
    final_success_rate=$(awk "BEGIN {printf \"%.1f\", ($SUCCESS_COUNT / ($SUCCESS_COUNT + $FAILED_COUNT)) * 100}")
    log_both "  • Success Rate: ${final_success_rate}%"
fi

log_both ""
log_both "${CYAN}Benign Task Completion Results:${NC}"

# Aggregate benign reproducibility statistics using UNSAFE/SAFE classification
if [ $SUCCESS_COUNT -gt 0 ]; then
    aggregate_stats=$(python3 -c "
import json
import glob

summaries = glob.glob('$BASE_DIR/$DOMAIN/**/benign_reproducibility_summary.json', recursive=True)

total_cases = 0
task_completion_rates = []
task_scores = []
baseline_harm_rates = []
total_unsafe = 0
total_safe = 0
total_costs = []

for summary_file in summaries:
    try:
        with open(summary_file) as f:
            data = json.load(f)
            stats = data.get('statistics', {})
            total_cases += 1
            task_completion_rates.append(stats.get('task_completion_rate', 0.0))
            task_scores.append(stats.get('avg_task_score', 0.0))
            baseline_harm_rates.append(stats.get('baseline_harm_rate', 0.0))
            total_unsafe += stats.get('unsafe_count', 0)
            total_safe += stats.get('safe_count', 0)
            total_costs.append(stats.get('total_cost', 0.0))
    except:
        pass

if total_cases > 0:
    avg_task_completion_rate = sum(task_completion_rates) / len(task_completion_rates) * 100 if task_completion_rates else 0.0
    avg_task_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
    avg_baseline_harm_rate = sum(baseline_harm_rates) / len(baseline_harm_rates) * 100 if baseline_harm_rates else 0.0
    avg_total_cost = sum(total_costs) / len(total_costs) if total_costs else 0.0
    print(f'{total_cases}|{avg_task_completion_rate:.1f}|{avg_task_score:.2f}|{avg_baseline_harm_rate:.1f}|{total_unsafe}|{total_safe}|{avg_total_cost:.2f}')
else:
    print('0|0.0|0.0|0.0|0|0|0.0')
" 2>/dev/null || echo "0|0.0|0.0|0.0|0|0|0.0")

    IFS='|' read -r total_analyzed avg_task_completion avg_task_score avg_baseline_harm total_unsafe total_safe avg_total_cost <<< "$aggregate_stats"

    log_both "  • Total Cases Analyzed: ${total_analyzed}"
    log_both "  • Average Task Completion Rate: ${GREEN}${avg_task_completion}%${NC}"
    log_both "  • Average Task Score: ${avg_task_score}"
    log_both "  • Average Total Cost: \$${avg_total_cost}"
    log_both ""
    log_both "${MAGENTA}Baseline Harm Rate (using UNSAFE/SAFE classification):${NC}"
    log_both "  • Average Baseline Harm Rate: ${YELLOW}${avg_baseline_harm}%${NC}"
    log_both "  • Total UNSAFE Trials: ${total_unsafe}"
    log_both "  • Total SAFE Trials: ${total_safe}"
fi

log_both ""
log_both "${CYAN}Logs:${NC}"
log_both "  • Main Log: $MAIN_LOG"
log_both "  • Case Logs: $LOG_DIR/benign_case_*_${TIMESTAMP}.log"
log_both ""

# Generate aggregate report using UNSAFE/SAFE classification
AGGREGATE_REPORT="$BASE_DIR/$DOMAIN/benign_reproducibility_batch_report_${TIMESTAMP}.json"
python3 -c "
import json
import glob
from datetime import datetime

summaries = glob.glob('$BASE_DIR/$DOMAIN/**/benign_reproducibility_summary.json', recursive=True)

all_results = []
total_unsafe = 0
total_safe = 0
for summary_file in summaries:
    try:
        with open(summary_file) as f:
            data = json.load(f)
            stats = data['statistics']
            unsafe_count = stats.get('unsafe_count', 0)
            safe_count = stats.get('safe_count', 0)
            total_unsafe += unsafe_count
            total_safe += safe_count
            all_results.append({
                'task_id': data.get('task_id', ''),
                'elicitation_case_id': data.get('elicitation_case_id', ''),
                'agent_model': data.get('agent_model', ''),
                'task_completion_rate': stats['task_completion_rate'],
                'avg_task_score': stats['avg_task_score'],
                'baseline_harm_rate': stats['baseline_harm_rate'],
                'unsafe_count': unsafe_count,
                'safe_count': safe_count,
                'total_cost': stats['total_cost'],
                'n_trials': data.get('n_trials', 0)
            })
    except:
        pass

report = {
    'batch_info': {
        'test_type': 'benign',
        'domain': '$DOMAIN',
        'timestamp': '$TIMESTAMP',
        'total_unique_task_agent_pairs': $TOTAL_PENDING,
        'successful_tests': $SUCCESS_COUNT,
        'failed_tests': $FAILED_COUNT,
        'duration_hours': $TOTAL_HOURS + $TOTAL_MINUTES / 60.0,
        'total_cost': sum(r['total_cost'] for r in all_results)
    },
    'results': all_results,
    'statistics': {
        'total_analyzed': len(all_results),
        'avg_task_completion_rate': sum(r['task_completion_rate'] for r in all_results) / len(all_results) if all_results else 0.0,
        'avg_task_score': sum(r['avg_task_score'] for r in all_results) / len(all_results) if all_results else 0.0,
        # Baseline harm rate using UNSAFE/SAFE classification
        'avg_baseline_harm_rate': sum(r['baseline_harm_rate'] for r in all_results) / len(all_results) if all_results else 0.0,
        'total_unsafe': total_unsafe,
        'total_safe': total_safe,
        'avg_total_cost': sum(r['total_cost'] for r in all_results) / len(all_results) if all_results else 0.0
    }
}

with open('$AGGREGATE_REPORT', 'w') as f:
    json.dump(report, f, indent=2)

print('Aggregate report saved')
" 2>/dev/null

if [ -f "$AGGREGATE_REPORT" ]; then
    log_both "${CYAN}Aggregate Report:${NC}"
    log_both "  • $AGGREGATE_REPORT"
    log_both ""
fi

print_header "Benign Task Reproducibility Analysis Complete!"
log_both ""
log_both "${GREEN}✓ All benign reproducibility tests completed${NC}"
log_both ""

