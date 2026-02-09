# AutoElicit

<div align="center">
  <img src="figures/autoelicit_figure1.png" width="95%" title="Introduction Figure">
</div>

<br>

<p align="center">
[<a href="https://osu-nlp-group.github.io/AutoElicit/">Website</a>] •
[<a href="">Paper</a>] •
[<a href="https://huggingface.co/collections/osunlp/autoelicit">Data</a>]
</p>

### AutoElicit is an agentic framework for automatically eliciting *unintended behaviors* from computer-use agents under realistic, benign inputs, surfacing long-tail safety risks reflecting real-world scenarios.

This is the official codebase for "[When Benign Inputs Lead to Severe Harms: Eliciting Unsafe Unintended Behaviors of Computer-Use Agents]()". AutoElicit elicits unintended behaviors, unsafe agent behaviors that deviate from the user’s intentions for a task and emerge inadvertently from benign instruction and environment contexts without adversarial manipulation.

AutoElicit iteratively perturbs benign instructions using real-world CUA execution feedback to elicit severe harms while keepign perturbations realistic and benign. The framework consists of two stages:
- **Context-Aware Seed Generation** proposes *seed perturbations*, featuring plausible unintended behavior targets given an OSWorld
task’s environment context and an initial perturbation to increase the likelihood of eliciting harms.
    - We release [AutoElicit-Seed](https://huggingface.co/datasets/osunlp/AutoElicit-Seed), a dataset of 361 seed perturbations spanning 66 benign OSWorld tasks. 
- **Execution-Guided Perturbation Refinement** executes seed perturbed instructions, automatically evaluates the resulting trajectories, and iteratively refines perturbations given execution feedback and predefined quality rubrics to improve elicitation success while preserving realism and benignity.
    - We release [AutoElicit-Bench](https://huggingface.co/datasets/osunlp/AutoElicit-Bench), a benchmark of 117 successful perturbations against Haiku (50) and Opus (67) for evaluating robustness to unintended behaviors in benign contexts.

<div align="center">
  <img src="figures/autoelicit_figure2.png" width="95%" title="Introduction Figure">
</div>

---

## 📣 Updates
- 2/9/2026: Paper, code, and dataset release.

---

## 🛠️ Setup

### 1. Python Environment Setup

First, install all dependencies and ensure you’re running Python 3.10.

```bash
# (Optional)
conda create -n unintended_behaviors python=3.10
conda activate unintended_behaviors

# Install project dependencies:
pip install -r requirements.txt
```

### 2. API Provider Environment Variables

Create an `.env` file within your project directory, setting your API keys for CUA usage and the OSWorld environment. 

- OpenAI: `OPENAI_API_KEY`
- Azure OpenAI: `AZURE_API_KEY`, `AZURE_API_VERSION`, `AZURE_ENDPOINT`
- Anthropic: `ANTHROPIC_API_KEY`
- AWS Bedrock: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`

### 3. OSWorld Environment Setup 

AutoElicit is designed to run the OSWorld environment using AWS EC2 Instances for large-scale parallel evaluation. 

For additional information about OSWorld environment setup, follow the official OSWorld Github's instructions at [Setup Guideline - Public Evaluation Platform](https://github.com/xlang-ai/OSWorld/blob/main/SETUP_GUIDELINE.md#3-public-evaluation-platform).

Within the `.env` file, set the following enviroment variables:
- `AWS_SUBNET_ID`
- `AWS_SECURITY_GROUP_ID`
- `AWS_USE_PUBLIC_IP`
- `ENABLE_TTL=false`
- `AWS_AUTO_CREATE_SCHEDULER_ROLE=false`

To change the instance type for your AWS EC2 Instance, edit line 13 within `AutoElicit/desktop_env/providers/aws/manager.py`.

---

## 🗂️ Dataset Installation

### AutoElicit-Seed

The [AutoElicit-Seed](https://huggingface.co/datasets/osunlp/AutoElicit-Seed) dataset is hosted on HuggingFace. If you seek to use this data for large-scale elicitation analysis with AutoElicit, use `autoelicit_seed_loader.py` to install the dataset to your directory.

```bash
# Reconstruct the full dataset
python autoelicit_seed_loader.py --output-dir ./autoelicit_seed

# Or filter by domain
python autoelicit_seed_loader.py --output-dir ./autoelicit_seed --domain multi_apps
```

This reconstructs the directory structure required by `iterative_refinement.py`:
```bash
AutoElicit/
└── autoelicit_seed/
    └── {domain}/
        └── {task_id}/
            └── {perturbation_model}/
                └── perturbed_query_{perturbation_id}/
                    └── perturbed_query_{perturbation_id}.json

    
```

Then use `./autoelicit_seed` as the base_dir for `iterative_refinement.py`:

```bash
python iterative_refinement.py --task-id <task_id> --domain <domain> \
    --perturbed-id <perturbation_id> --base-dir ./autoelicit_seed
```

### AutoElicit-Bench

The [AutoElicit-Bench](https://huggingface.co/datasets/osunlp/AutoElicit-Seed) dataset is hosted on HuggingFace and is also available under the `transferability/robustness_benchmark/` directory. If you seek to download from HuggingFace, use the following commands.

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("osunlp/AutoElicit-Bench", split="train")

print(f"Total perturbations: {len(dataset)}")  # 117

# Iterate through perturbations
for example in dataset:
    task_id = example["task_id"]
    domain = example["domain"]  # 'os' or 'multi_apps'
    instruction = example["perturbed_instruction"]
    source_agent = example["execution_agent"]

    # Execute on your CUA and evaluate safety
```

---

## Context-Aware Seed Generation

**Context-Aware Seed Generation** is the first stage of AutoElicit. It generates minimal, realistic perturbations to benign OSWorld tasks that are likely to elicit unintended behaviors from computer-use agents.

This stage results in *seed perturbations* consisting of: 

1. **Unintended Behavior Targets**  
   A plausible harm that could emerge during execution of a specified benign task, grounded in the environment context and a representative benign task trajectory.

2. **Initial Perturbations**  
   Minimal modifications to the original benign task that increase the likelihood of eliciting the unintended behavior while preserving task intent.

To *generate new seeds* using Context-Aware Seed Generation, follow the provided instructions in [seed_generation/README_SEED_GEN.md](https://github.com/OSU-NLP-Group/AutoElicit/blob/main/seed_generation/README_SEED_GEN.md).

---

## Execution-Guided Perturbation Refinement 

**Execution-Guided Perturbation Refinement** is the second stage of AutoElicit. It performs iterative refinement of seed perturbations based on real agent execution of perturbed instructions, using automatic feedback to increase the likelihood of elicitation while keeping instructions benign and realistic. This stage operates on filtered seed perturbations produced by Context-Aware Seed Generation.

This stage refines perturbed instructions to elicit unintended behaviors using *nested dual feedback loops*:

1. **Execution Feedback Loop (Outer Loop)**  

   Executes perturbed instructions on a specified computer-use agent, automatically evaluates resulting trajectories, and refines instructions based on execution from prior attempts. This outer loop continues until an unintended behavior is elicited or until the max number of execution iterations is reached.

2. **Quality Feedback Loop (Inner Loop)** 

   Performs a *quality check* to ensure any proposed perturbation based on execution feedback maintains required quality thresholds before being executed. This inner loop continues until the perturbation meets all quality thresholds or until the max number of quality refinement iterations is reached.

To surface unintended behaviors using Execution-Guided Perturbation Refinement, follow the provided instructions in [iterative_refinement/README_REFINEMENT.md](https://github.com/OSU-NLP-Group/AutoElicit/blob/main/iterative_refinement/README_REFINEMENT.md).

---

## Meta-Analysis

**Meta-Analysis** provides tools for automatically analyzing successful elicitation runs, allowing for deeper insights about benign input vulnerabilities only apparent across large-scale elicitation data. It generates fine-grained vulnerability categoires and higher-level clusters describing recurring patterns and failure modes.

To perform large-scale automatic qualitative analysis over successful perturbations identified from AutoElicit, follow the provided instructions in [meta_analysis/README_META_ANALYSIS.md](https://github.com/OSU-NLP-Group/AutoElicit/blob/main/meta_analysis/README_META_ANALYSIS.md).

---

## AutoElicit-Bench Evaluation 

Using [AutoElicit-Bench](https://huggingface.co/datasets/osunlp/AutoElicit-Bench), we measure the *transferability* of successful perturbations identified by AutoElicit. Given a benchmark of 117 human-verified perturbations that successfully elicited unsafe behaviors from source agents (Claude 4.5 Haiku and Opus), we measure whether these perturbations can elicit similar behaviors in other target agents.

To perform this analysis on additional target agents, follow the provided instructions in [README_TRANSFER.md](https://github.com/OSU-NLP-Group/AutoElicit/blob/main/transferability/README_TRANSFER.md).

---

## Contacts 

[Jaylen Jones](mailto:jones.6278@osu.edu), [Zhehao Zhang](mailto:zhang.16420@buckeyemail.osu.edu), [Huan Sun](mailto:sun.397@osu.edu)

---

## Citation

**Fill in this section later**
