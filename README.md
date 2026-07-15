# MLPlatAgent

> Collaborating with Specialized Software: A System-AI Collaborative Agent for Automated Machine Learning Workflow Construction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-Submitted_to_JSS-orange.svg)]()

MLPlatAgent is a system-AI collaborative agent that converts natural-language machine-learning requests into executable low-code ML platform workflows. Instead of generating a full raw script, MLPlatAgent retrieves platform widgets, assembles workflow DAGs, and lets the underlying platform execute the workflow.

## Core Features

- **NL2Workflow paradigm:** translates user requirements into platform-native workflow operations rather than standalone scripts.
- **Function Call Code mechanism:** represents node creation, parameter updates, and DAG edge construction as executable function calls.
- **Data-aware tool selection:** uses dataset summaries and task context to retrieve relevant widgets and cases.
- **Feedback-driven repair:** uses platform execution feedback and error messages to revise generated workflows.

![Comparison of NL2Code and NL2Workflow.](static/NL2Workflow_NL2Code.png)

## System Architecture

MLPlatAgent has three main stages:

1. **Intent identification and task decomposition:** classifies the request and decomposes it into workflow-level steps.
2. **Hierarchical tool retrieval:** combines user requirements, cases, widgets, and dataset summaries to select platform operations.
3. **Workflow assembly via function calls:** generates function-call code that adds nodes, updates parameters, and connects the workflow DAG.

![Overview of MLPlatAgent.](static/MLPlatAgent_overview.png)

## Repository Contents

The repository includes the source code, benchmark instruction files, evaluation requirements, SQL dumps, and the RCR checklist used to make the reproducibility package auditable.

| Path | Contents |
| --- | --- |
| `agents/`, `ml_platform/`, `prompts/`, `llm/`, `utils/` | MLPlatAgent implementation. |
| `data/benchmark/ml_benchmark.json` | ML-Benchmark instruction/evaluation file; 8 released instructions. |
| `data/benchmark/dseval_kaggle.json` | DSEval-Kaggle instruction/evaluation file; 10 released instructions. |
| `data/benchmark/DSEval-Kaggle-Ext/UCI.json` | User command intent subset for DSEval-Kaggle-Ext; 40 released instructions. |
| `data/benchmark/DSEval-Kaggle-Ext/CI.json` | Command intent subset for DSEval-Kaggle-Ext; 40 released instructions. |
| `data/benchmark/DSEval-Kaggle-Ext/MI.json` | Modification intent subset for DSEval-Kaggle-Ext; 21 released instructions. |
| `data/benchmark/datasets/ml_benchmark.sql` | SQL dump for ML-Benchmark datasets. |
| `data/benchmark/datasets/dseval_kaggle.sql` | SQL dump for DSEval-Kaggle datasets. |
| `data/benchmark/datasets/mlagent.sql` | SQL dump for the extended MLPlatAgent/DSEval-Kaggle-Ext datasets. |
| `RCR Checklist.pdf` | Reproducibility checklist used for evaluation. The highlighted red text in each instruction indicates the required elements that should be satisfied. |
| `data/ml_platform_data_example/` | Example platform widget metadata used by retrieval and workflow generation. |
| `data/cases_library/` | Case libraries used by planner/executor retrieval. |

Each benchmark JSON entry contains an `Instruction` field and a `requirements` object. The requirements object records both `step_requirements` and `dependency_requirements`, allowing generated workflows to be checked against required nodes, parameters, and DAG dependencies.

## What Is Included and What Requires External Access

Included in this repository:

- Source code for MLPlatAgent.
- Released benchmark instructions and machine-readable evaluation requirements.
- SQL dumps for the released benchmark datasets.
- RCR checklist PDF.
- Example platform widget metadata and retrieval cases.
- A clear open-source license.

Not included:

- Private credentials for the Uniplore AI Studio or any other low-code ML platform deployment.
- Private API keys for LLM providers.
- Locally downloaded embedding-model weights.
- Platform execution logs generated during a run.

Executing workflows end to end requires valid platform, database, and LLM credentials. The public files are sufficient to audit the released instructions, expected workflow requirements, dataset table dumps, and implementation logic.

## Environment

The experiments were configured with:

- Python: 3.10 or later; Python 3.11 is recommended.
- LLM client: `openai==0.27.0` with an OpenAI-compatible chat-completion endpoint.
- Default LLM setting in `config.py`: `MODEL_NAME = "qwen3.6-plus"`, `Model_PATH = "https://dashscope.aliyuncs.com/compatible-mode/v1"`.
- LLM decoding defaults in `llm/llm.py`: `temperature=0`, `top_p=1`, `max_tokens=4096`.
- Default retriever in `agents/mlagent.py`: `multilingual-e5-large`.
- Random seed in `config.py`: `RANDOM_SEED = 42`.

The low-code platform executes the final ML workflow. If a widget exposes its own seed parameter, set it in the platform widget parameters when exact platform-level repeatability is required.

## Installation

```bash
git clone https://github.com/acmisxuyutian/MLPlatAgent.git
cd MLPlatAgent

conda create -n mlagent python=3.11
conda activate mlagent
pip install -r requirements.txt
```

Download the embedding model weights before running retrieval:

```bash
mkdir -p embedding_models
# Download from Hugging Face and place the directory here:
# embedding_models/multilingual-e5-large
```

Default embedding model:

- `intfloat/multilingual-e5-large`: https://huggingface.co/intfloat/multilingual-e5-large

If you use another retriever supported in `embedding_models/embedding_model.py`, place it under `embedding_models/<retriever-name>`.

## Database Setup

Create a MySQL database and import the released benchmark dumps as needed:

```bash
mysql -u <user> -p -e "CREATE DATABASE IF NOT EXISTS mlagent DEFAULT CHARACTER SET utf8mb4;"
mysql -u <user> -p mlagent < data/benchmark/datasets/ml_benchmark.sql
mysql -u <user> -p mlagent < data/benchmark/datasets/dseval_kaggle.sql
mysql -u <user> -p mlagent < data/benchmark/datasets/mlagent.sql
```

The low-code platform must be able to access the same MySQL database. Update `config.py` accordingly:

```python
MySQL_Config = {
    "server": "<host reachable by the platform>",
    "port": "3306",
    "username": "<mysql-user>",
    "password": "<mysql-password>",
    "database": "mlagent",
}
```

## Credentials and Configuration

Edit `config.py` before running:

```python
Accesstoken = "<platform-access-token>"
Workflow_id = <platform-workflow-id>
AI_STUDIO_URL = "<platform-api-base-url>"

Model_PATH = "<openai-compatible-base-url>"
MODEL_NAME = "<chat-model-name>"
API_KEY = "<llm-api-key>"
RANDOM_SEED = 42
```

Credential assumptions:

- `Accesstoken` must authorize workflow creation, node updates, edge updates, execution, and log retrieval on the target low-code ML platform.
- `Workflow_id` must refer to a workflow that the token can modify.
- `API_KEY` must be valid for the configured OpenAI-compatible LLM endpoint.
- The MySQL account must allow reading benchmark tables and must be reachable by both MLPlatAgent and the platform service.

## Quick Start

After configuration, run the demo request:

```bash
python run.py
```

The default `run.py` request is:

```python
requirement = "Load the Iris example dataset and build a model."
```

The script prints the generated plan and execution result JSON. A short demo video is available at https://youtu.be/aN-5xPOluyU.

## Reproducing Benchmark Audits

1. Import the SQL dumps for the benchmark you want to inspect.
2. Configure `config.py` with platform, database, and LLM credentials.
3. Select an instruction from one of the JSON benchmark files under `data/benchmark/`.
4. Run MLPlatAgent with that instruction.
5. Compare the generated workflow against the entry's `requirements.step_requirements` and `requirements.dependency_requirements`.
6. Use `RCR Checklist.pdf` to audit whether all highlighted required elements are satisfied.

Example:

```python
from agents.mlagent import MLAgent

instruction = "Load the Iris example dataset and build a model."
agent = MLAgent()
result = agent.run(instruction, case_number=3, retriever="multilingual-e5-large")
print(result)
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
