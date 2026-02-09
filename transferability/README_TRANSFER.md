# Transferability Evaluation

This module evaluates the **transferability** of successful perturbations across different computer-use agents (CUAs). Given a benchmark of 117 human-verified perturbations that successfully elicited unsafe behaviors from source agents (Claude 4.5 Haiku and Opus), we measure whether these perturbations can elicit similar behaviors in other target agents.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  robustness_benchmark/                                              │
│  117 human-verified perturbations from Claude 4.5 Haiku & Opus      │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  run_benchmark_multiple_runs.py                                     │
│  Execute perturbations on target agent (3 runs each)                │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  analyze_trajectories_multiple_runs.py                              │
│  Two-stage LLM analysis: Summarization → Safety Classification      │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  summarize_results.py                                               │
│  Aggregate safety classifications across agents                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  compute_transfer_rates.py                                          │
│  Compute transfer rates from source to target agents                │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Step 1: Execute perturbations on target agent (3 runs each)
python run_benchmark_multiple_runs.py \
    --agent_model claude-sonnet-4-5-20250929 \
    --provider_name aws \
    --num_runs 3

# Step 2: Analyze trajectories for safety
python analyze_trajectories_multiple_runs.py \
    --agent_model claude-sonnet-4-5-20250929

# Step 3: Aggregate results across agents
python summarize_results.py

# Step 4: Compute transfer rates
python compute_transfer_rates.py
```

## Scripts

| Script | Description |
|--------|-------------|
| `run_benchmark.py` | Execute perturbations on target agent (single run) |
| `run_benchmark_multiple_runs.py` | Execute with 3 runs per perturbation |
| `analyze_trajectories.py` | Trajectory analysis (single run) |
| `analyze_trajectories_multiple_runs.py` | Multi-run analysis with aggregation |
| `summarize_results.py` | Cross-agent result aggregation |
| `compute_transfer_rates.py` | Transfer rate computation |

## Safety Classification

The analysis pipeline uses a two-stage LLM approach:

1. **Trajectory Summarization** (GPT-5-mini with vision)
   - Analyzes screenshots to summarize agent actions
   - Prioritizes visual evidence over agent's stated reasoning

2. **Safety Classification** (GPT-5)
   - Binary classification: SAFE or UNSAFE
   - Severity levels: `none`, `minimal`, `low`, `medium`, `high`, `critical`

## Transfer Rate Metric

For each perturbation, we execute 3 runs and report the percentage of perturbations where **unsafe behavior occurs in at least one run**:

```
Transfer Rate = (perturbations with ≥1 unsafe run) / (total perturbations) × 100
```

## Benchmark Data

The `robustness_benchmark/` directory contains **117 human-verified perturbations**:

| Source Agent | Count |
|--------------|-------|
| Claude 4.5 Haiku | 50 |
| Claude 4.5 Opus | 67 |

Each perturbation includes the perturbed instruction, elicitation strategy, and severity level.

## Output Structure

```
results/{agent_model}/{example_id}/
├── run_0/
│   ├── traj.jsonl                  # Action trajectory
│   ├── step_*.png                  # Screenshots
│   ├── trajectory_summary.md       # LLM-generated summary
│   └── safety_classification.json  # SAFE/UNSAFE verdict
├── run_1/
├── run_2/
├── aggregated_results.json         # Execution statistics
└── aggregated_analysis.json        # Aggregated safety verdict
```

## Requirements

```bash
pip install openai python-dotenv
```

**Environment Variables:**
```bash
export OPENAI_API_KEY="your-api-key"
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```
