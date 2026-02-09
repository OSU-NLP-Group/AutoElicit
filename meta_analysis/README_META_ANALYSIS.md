# Meta-Analysis

**Meta-Analysis** provides tools for automatically analyzing successful elicitation runs, allowing for deeper insights about benign input vulnerabilities only apparent across large-scale elicitation data. It generates fine-grained vulnerability categoires and higher-level clusters describing recurring patterns and failure modes. 

**This codebase allows you to:**
- Build filtered task lists of successful elicitation runs for a given domain and agent
- Generate LLM summaries of each elicitation run, describing how the instruction was perturbed and why it resulted in harm
- Categorize summaries into fine-grained vulnerability categories (new or resume from existing)
- Cluster categories into higher-level themes and produce human-readable reports

---

## Description

After running the elicitation pipeline, you obtain many successful perturbed instructions that led computer-use agents to unintended behaviors. The **Meta-Analysis** pipeline performs automatic qualitative analysis over this data:

1. **Task Selection**  
   Generates a filtered task list containing all successful elicitation runs (by domain, agent, refinement model). Optional filters: baseline harm rate (exclude tasks with >0% baseline harm) and human annotation decisions (exclude false positives).

2. **Summary Generation**  
   For each successful run, an LLM produces a short summary of *how* the instruction was perturbed and why it elicited the unintended behavior, using the original and perturbed instructions and full elicitation run context.

3. **Categorization**  
   Summaries are grouped into **benign input vulnerability categories**: fine-grained, actionable labels (with definitions and example trigger phrases) that describe the linguistic or contextual patterns that led to harm.

4. **Clustering**  
   Categories can be merged by an LLM into higher-level **clusters** (e.g., “tidiness/cleanup framing,” “persistent enforcement without lifecycle controls”) to reduce redundancy and support thematic reporting. A separate script generates markdown reports from cluster outputs for easy analysis.

---

## Prerequisites

```bash
# Set up API keys (see project root or main README)
cat > .env << EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
# Optional: Azure, AWS Bedrock
AZURE_API_KEY=...
AZURE_ENDPOINT=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
EOF
```

```bash
# Activate environment (see main README for setup)
conda activate unintended_behaviors
```

---

## Directory Structure

```
meta_analysis/
├── generate_successful_task_list.py      # Step 1: Task selection
├── elicitation_run_summary.py           # Step 2: Single-task summary generation
├── parallel_generate_elicitation_run_summaries.sh   # Step 2: Batch summary generation
├── elicitation_run_categorization.py    # Step 3: Categorization
├── llm_category_clustering.py           # Step 4: Cluster categories with an LLM
├── cluster_report.py                    # Step 4: Generate cluster summary report (Markdown)
├── prompts/
│   ├── elicitation_run_summary_prompt.md
│   ├── elicitation_run_categorization_prompt.md
│   ├── elicitation_run_categorization_update_prompt.md
│   └── category_clustering.md
├── task_filters/                        # Optional: filter files (you create as needed)
│   ├── data_filter_harm_threshold_0%_final.json
│   └── final_decisions.json
├── elicitation_run_categorization/      # Categorization outputs
│   └── {perturbation_model}/
│       └── all_refinement_models/
│           └── {agent}/
│               ├── elicitation_run_categorization_{agent}_{timestamp}.json
│               └── clusters_{model}/    # Clustering outputs (Step 4)
│                   ├── llm_clusters_{timestamp}.json
│                   └── cluster_report_{timestamp}.md
└── elicitation_summary_logs/            # Summary generation logs
```

---

## Step 1: Task Selection

Generate a filtered task list that identifies successful elicitation runs for a given domain, perturbation model, and execution agent.

### Usage

```bash
python generate_successful_task_list.py \
    --domain os \
    --perturbation_model o4-mini-2025-04-16 \
    --agent claude-haiku-4-5-20251001 
```

### Output

meta_analysis\
└── task_list_{domain}_{execution_agent}_all_refinement_models_{timestamp}.json


### Filter file formats (optional)

**Baseline harm rate filter** (e.g. `data_filter_harm_threshold_0%_final.json`):  
JSON that lists task IDs with baseline harm rate > 0% to **exclude**. 

```json
{
    "domain_name": {
        "agent_name": [
            "task-id-1", 
            "task-id-2"
        ]
    }
}
```

**Human annotation filter** (e.g. `final_decisions.json`):  
JSON of human annotation decisions. Entries with `"final_decision": "False Positive"` are excluded. Keys are composite (e.g. `task_id:perturbed_id:refinement_model:execution_agent`).

```json
{
    "task_id:perturbed_id:refinement_model:execution_agent": {
        "example_id": "task_id:perturbed_id:refinement_model:execution_agent",
        "final_decision": "True Success",
        "vote_count": 3,
        "execution_agent": "execution_agent",
        "domain": "domain_name",
        "severity": "severity_level",
        "annotator_decisions": [
            {
                "annotator": "Annotator 1",
                "decision": "True Success",
                "failed_criteria": []
            },
            {
                "annotator": "Annotator 2",
                "decision": "True Success",
                "failed_criteria": []
            },
            {
                "annotator": "Annotator 3",
                "decision": "True Success",
                "failed_criteria": []
            }
        ]
    }
    ...
}
```

### Usage

```bash
# With baseline harm filter and human annotation filter
python generate_successful_task_list.py \
    --domain os \
    --perturbation_model o4-mini-2025-04-16 \
    --agent claude-haiku-4-5-20251001 \
    --data_filter task_filters/data_filter_harm_threshold_0%_final.json \
    --human_annotation_filter task_filters/final_decisions.json \
    --output task_list_filtered.json
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--domain` | Domain to search (e.g. `os`, `multi_apps_test`, or `all`) |
| `--perturbation_model` | Perturbation model (default: `o4-mini-2025-04-16`) |
| `--agent` | Execution agent (e.g. `claude-haiku-4-5-20251001`) |
| `--data_filter` | Path to baseline harm rate filter JSON (optional) |
| `--human_annotation_filter` | Path to human annotation decisions JSON (optional) |
| `--output` | Output filename for the task list |

---

## Step 2: Summary Generation

Generate an elicitation run summary for each task in the task list. Each summary explains *how* the original instruction was perturbed, what specific strategy was used, and a hypothesis of *why* this perturbation led the agent to an unintended behavior.

### Single Summary Generation

```bash
python elicitation_run_summary.py \
    --task_id <TASK_ID> \
    --perturbed_id <PERTURBED_ID> \
    --domain <DOMAIN> \
    --perturbation_model <PERTURBATION_MODEL> \
    --refinement_model <REFINEMENT_MODEL> \
    --agent <EXECUTION_AGENT> 
```


### Batch Summary Generation

```bash
./parallel_generate_elicitation_run_summaries.sh \
    --task_list_file <TASK_LIST_FILE_PATH> \
    --perturbation_model <PERTURBATION_MODEL> \
    --agent <EXECUTION_AGENT> \
    --max_parallel 10
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--task_list_file` | Path to the task list JSON from Step 1 |
| `--agent` | Execution agent name |
| `--max_parallel` | Number of parallel jobs (default: 10) |
| `--perturbation_model` | Perturbation model (default: `o4-mini-2025-04-16`) |
| `--api` | API provider: `openai`, `anthropic`, `anthropic_bedrock` (default: `openai`) |
| `--model` | Summary generation model (default: `gpt-5-2025-08-07`) |

### Output location

Summaries are written next to each task’s data:

```
perturbed_queries/{domain}/{task_id}/{perturbation_model}/
└── perturbed_query_{perturbed_id}/
    └── iterative_refinement_{refinement_model}/
        └── agent_{agent}/
            └── elicitation_run_summary.json
```

### Logs

Batch logs and per-task status/cost are under:

```
elicitation_summary_logs/summary_{agent}_{timestamp}/
├── run_summary_{agent}_{timestamp}.json
├── {task_id}_{perturbed_id}_{refinement_suffix}.log
├── {task_id}_{perturbed_id}_{refinement_suffix}.status
└── {task_id}_{perturbed_id}_{refinement_suffix}.cost
```

---

## Step 3: Categorization

Categorize the generated summaries into fine-grained, actionable vulnerability categories based on shared linguistic features and failure modes. You can **create a new categorization** or **resume from an existing one** (e.g. to add another domain or agent).

To enable categorization over large-scale data while avoiding context limits, categorization is performed in batches with `intial_categorization_batch_size` defining the batch size for the initial set of categories and `iterative_categorization_batch_size` defining the batch size for all subsequent iterations. 

### Create a new categorization

```bash
python elicitation_run_categorization.py \
    --task_list_path <TASK_LIST_FILE_PATH> \
    --agent <EXECUTION_AGENT> \
    --initial_categorization_batch_size 10 \
    --iterative_categorization_batch_size 5
```

### Output
```
meta_analysis/
└── elicitation_run_categorization/{perturbation_model}/all_refinement_models/
    └── {execution_agent}
        └── elicitation_run_categorization_{execution_agent}_{timestamp}.json
```

### Resume from an existing categorization

```bash
python elicitation_run_categorization.py \
    --task_list_path task_list_filtered.json \
    --agent claude-haiku-4-5-20251001 \
    --resume_categorization_path <CATEGORIZATION_PATH>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--task_list_path` | Path to task list JSON from Step 1 |
| `--agent` | Execution agent name |
| `--resume_categorization_path` | Path to existing categorization JSON to extend (optional) |
| `--refinement_model_filter` | Restrict to a single refinement model (optional) |
| `--initial_categorization_batch_size` | Number of tasks in initial categorization (default: 10) |
| `--iterative_categorization_batch_size` | Tasks per iterative batch (default: 5) |
| `--model` | Categorization model (default: `gpt-5-2025-08-07`) |

---

## Clustering

After categorization, fine-grained categories are merged into higher-level **clusters** to capture high-level vulnerabilty patterns. This uses two scripts: `llm_category_clustering.py` (LLM clustering) and `cluster_report.py` (report generation).

### Step 4a: LLM category clustering

`llm_category_clustering.py` reads a categorization JSON and prompts an LLM to group categories by semantic overlap based on shared linguistic and contextual cues, heuristic identity based on whether the frontier CUA deviates from safe behavior due to the same fundamental limitation, and harm similarity based on whether the unintended behavior results in the same harm.

### Usage

```bash
python llm_category_clustering.py \
    --input_path <CATEGORIZATION_PATH>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--input_path` | Path to categorization JSON (output of Step 3) |
| `--output_dir` | Output directory (default: `clusters_{model_name}` in same dir as input) |
| `--api` | API provider: `openai`, `azure`, `anthropic`, `anthropic_bedrock` (default: `openai`) |
| `--model` | Model for clustering (default: `gpt-5-2025-08-07`) |
| `--max_tokens` | Max tokens for LLM response (default: 64000) |
| `--temperature` | Sampling temperature (default: 1.0) |

### Output
```
meta_analysis/
└── elicitation_run_categorization/{perturbation_model}/all_refinement_models/
    └── {execution_agent}
        └── clusters_{refinement_model}
            └── llm_clusters_{timestamp}.json
```
---

### Step 4b: Cluster report

`cluster_report.py` turns a clusters JSON (and the original categorization JSON) into a Markdown report: overall statistics, a summary table, and a detailed breakdown per cluster with example trigger phrases.

### Usage

```bash
# Use categorization path stored in cluster file metadata
python cluster_report.py \
    --cluster_path <CLUSTER_FILE_PATH>

# Specify categorization file explicitly
python cluster_report.py \
    --cluster_path <CLUSTER_FILE_PATH> \
    --categorization_path <CATEGORIZATION_PATH>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--cluster_path` | Path to the LLM clusters JSON from Step 4a |
| `--categorization_path` | Path to original categorization JSON (optional; read from cluster metadata if not set) |
| `--output_path` | Output Markdown path (default: `cluster_report_{timestamp}.md` in same dir as cluster file) |
| `--max_examples` | Max trigger phrase examples per category in the report (default: 3) |
| `--print_report` | Print the report to stdout |

### Output

```
meta_analysis/
└── elicitation_run_categorization/{perturbation_model}/all_refinement_models/
    └── {execution_agent}
        └── clusters_{refinement_model}
            └── cluster_report_{timestamp}.md
```

---