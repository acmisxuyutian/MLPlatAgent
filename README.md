# MLPlatAgent 🤖🛠️

> **Collaborating with Specialized Software: A System-AI Collaborative Agent for Automated Machine Learning Workflow Construction**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-Submitted_to_JSS-orange.svg)]()

Welcome to the official repository for **MLPlatAgent**. 

We exploring the novel `System-AI Collaborative` software engineering path. MLPlatAgent orchestrates professional low-code Machine Learning platforms via Large Language Models (LLMs), pioneering the transition from traditional code generation (NL2Code) to workflow orchestration (**NL2Workflow**).

---

## ✨ Core Features & Paradigm Shift

Unlike existing coding assistants (e.g., Claude Code) or agents evaluated on MLE-Bench that generate raw scripts from scratch, MLPlatAgent acts as an intelligent orchestrator over mature enterprise ML platforms.

* 🔄 **NL2Workflow Paradigm (System-AI Collaboration):** Shields the LLM from highly error-prone low-level syntax generation, avoiding hallucinations and bugs by delegating execution to the native engines of low-code platforms.
* 🔗 **Function Call Code (FCC) Mechanism:** A novel architectural design that resolves Directed Acyclic Graph (DAG) dependency bottlenecks, successfully passing intermediate states and variables between sequential node operations.
* 🧠 **Data-Aware Tool Selection:** Dynamically incorporates dataset summary profiles (e.g., target feature distributions) into the context, enabling the agent to proactively retrieve specialized tools (such as resampling widgets for highly imbalanced datasets).
* 🛡️ **Feedback-Driven Fault Tolerance:** Leverages the robust error-handling of industrial ML platforms, capturing native error logs to prompt the LLM for re-evaluation and self-correction during workflow generation.

---
![Comparison of the NL2Code and NL2Workflow paradigms.](MLPlatAgent/static/NL2Workflow_NL2Code.png){width=50%}
## 🏗️ System Architecture

MLPlatAgent processes unstructured natural language instructions and translates them into executable workflow topologies through three main phases:
1. **Intent Identification & Task Decomposition:** Routes intents into specialized paths (Traditional ML, Deep Learning, Modification) utilizing standardized heuristic mappings to align with enterprise low-code templates.
2. **Hierarchical Tool Retrieval:** Synergizes user queries with dynamic data summaries to retrieve the most context-appropriate platform widgets.
3. **Workflow Assembly via FCC:** Constructs the explicit DAG topology, ensuring logical soundness and executable structural adherence prior to deployment.
---
![Overview of the MLPlatAgent framework.](MLPlatAgent/static/MLPlatAgent_overview.png){width=50%}

## 🚀 Quick Start

### Prerequisites
* Python 3.10 or higher
* Access credentials for the designated Low-code Machine Learning Platform API (e.g., Uniplore)
* OpenAI API Key (or equivalent LLM provider)

### Installation
Download dependencies
```bash
git clone https://github.com/acmisxuyutian/MLPlatAgent.git
cd MLPlatAgent
conda create -n mlagent python=3.11
conda activate mlagent
pip install -r requirements.txt
```

由于涉及数据集和示例检索需要获取embedding模型下载完成后将文件放到embedding_models目录下。：https://huggingface.co/intfloat/multilingual-e5-large

