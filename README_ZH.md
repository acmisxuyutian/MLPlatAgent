# MLPlatAgent

[English](README.md)

> 与专业软件协作：面向自动化机器学习工作流构建的系统–AI 协同智能体

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-Submitted_to_JSS-orange.svg)]()

MLPlatAgent 是一个系统–AI 协同智能体，能够将自然语言描述的机器学习需求转换为可执行的低代码机器学习平台工作流。MLPlatAgent 不直接生成完整的原始脚本，而是检索平台组件、组装工作流有向无环图（DAG），并由选定的平台执行工作流。

## 核心特性

- **NL2Workflow 范式：**将用户需求转换为平台原生工作流操作，而不是独立脚本。
- **Function Call Code 机制：**将节点创建、参数更新和 DAG 边构建表示为可执行的函数调用。
- **数据感知的工具选择：**结合数据集摘要和任务上下文检索相关组件与案例。
- **反馈驱动的修复：**利用平台执行反馈和错误信息修正生成的工作流。
- **统一平台 Action：**使规划和工作流构建逻辑不依赖具体平台的 API 与工作流格式。

![NL2Code 与 NL2Workflow 对比。](static/NL2Workflow_NL2Code.png)

## 系统架构

MLPlatAgent 包含三个主要阶段：

1. **意图识别与任务分解：**对用户请求进行分类，并将其分解为工作流级步骤。
2. **分层工具检索：**综合用户需求、案例、平台组件和数据集摘要，选择合适的操作。
3. **通过函数调用组装工作流：**生成函数调用代码，用于添加节点、更新参数和连接工作流 DAG。

Agent 核心始终导入同一个进程级平台适配器：

```python
from ml_platform.actions import action_agent
```

`action_agent` 根据根目录 `config.py` 中选择的平台完成实例化。

![MLPlatAgent 总体架构。](static/MLPlatAgent_overview.png)

## 支持的平台

| 平台 | 工作流对象 | SQL Table 使用的数据源 | 运行环境 | 平台配置 |
| --- | --- | --- | --- | --- |
| **Uniplore** | 通过平台 API 操作的远程工作流 | MySQL | 项目的 Conda 或 Python 环境 | `UNIPLORE_CONFIG` |
| **Orange3** | 本地 `.ows` 工作流 | PostgreSQL | Portable Orange Python | `ORANGE3_CONFIG` |

两个平台共享同一套 Planner、Executor、提示词、案例和根目录 `run.py` 入口，但平台边界不同：

- Uniplore 需要访问令牌、远程工作流 ID 和 API 地址。
- Orange3 的平台配置只需要当前 `.ows` 工作流的路径，不读取或校验 Uniplore 凭据。
- MySQL 和 PostgreSQL 属于数据源配置，不属于平台身份配置。
- Python 进程初始化时只读取一次平台选择。修改 `PLATFORM` 或对应配置后，需要重新启动进程。

如需增加其他平台，请参阅[平台扩展指南](ml_platform/README_ZH.md)。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| `agents/`、`ml_platform/`、`prompts/`、`llm/`、`utils/` | MLPlatAgent 实现。 |
| `data/benchmark/ml_benchmark.json` | ML-Benchmark 指令与评估文件；包含 8 条公开指令。 |
| `data/benchmark/dseval_kaggle.json` | DSEval-Kaggle 指令与评估文件；包含 10 条公开指令。 |
| `data/benchmark/DSEval-Kaggle-Ext/UCI.json` | DSEval-Kaggle-Ext 的模糊指令；包含 14 条公开指令。 |
| `data/benchmark/DSEval-Kaggle-Ext/CI.json` | DSEval-Kaggle-Ext 的明确指令；包含 14 条公开指令。 |
| `data/benchmark/DSEval-Kaggle-Ext/MI.json` | DSEval-Kaggle-Ext 的修改指令；包含 21 条公开指令。 |
| `data/benchmark/datasets/ml_benchmark.sql` | ML-Benchmark 数据集的 SQL 转储文件。 |
| `data/benchmark/datasets/dseval_kaggle.sql` | DSEval-Kaggle 数据集的 SQL 转储文件。 |
| `data/benchmark/datasets/mlagent.sql` | MLPlatAgent 扩展数据集的 SQL 转储文件。 |
| `data/ml_platform_data_example/` | 平台组件元数据示例。 |
| `data/cases_library/` | Planner 和 Executor 检索使用的案例库。 |
| `RCR Checklist.pdf` | 实验使用的需求检查清单。 |

benchmark JSON 条目包含 `Instruction` 字段和 `requirements` 对象。`requirements` 通过 `step_requirements` 和 `dependency_requirements` 记录预期的工作流节点、参数及 DAG 依赖关系。

## 前置条件

公共要求：

- 可访问的兼容 OpenAI 的聊天补全接口和 API 密钥；
- 检索器所需的嵌入模型；
- 当前任务所需数据库的访问权限；
- 使用本文所述 Portable Orange 运行环境和命令时，需要 Windows。

实验使用以下配置：

- Python 3.10 或更高版本；Uniplore 环境推荐使用 Python 3.11；
- `openai==0.27.0`；
- 使用 `Qwen2.5-72B-Instruct` 和 `Qwen2.5-14B-Coder` 作为 Agent 骨架模型；
- 使用 `multilingual-e5-large` 作为默认检索器；
- `llm/llm.py` 中使用 `temperature=0`、`top_p=1` 和 `max_tokens=4096`；
- `config.py` 中使用 `RANDOM_SEED = 42`。

仓库不包含私有平台凭据、LLM API 密钥、下载后的嵌入模型权重和平台执行日志。

## 安装

首先克隆仓库：

```bash
git clone https://github.com/acmisxuyutian/MLPlatAgent.git
cd MLPlatAgent
```

然后根据需要使用的平台准备运行环境。

### 方案 A：Uniplore

创建标准 Python 环境，并安装根目录依赖：

```bash
conda create -n mlagent python=3.11
conda activate mlagent
pip install -r requirements.txt
```

该环境用于启动 MLPlatAgent，Uniplore 工作流则通过配置的平台 API 在远程执行。

### 方案 B：Orange3

MLPlatAgent 使用 Windows 版 Portable Orange。该启动方式不使用 Conda 环境或系统中单独安装的 Orange。

1. 打开 [Orange 官方下载页面](https://orangedatamining.com/download/)。
2. 在 **Windows** 下载项中选择并下载 **Portable Orange**（`Orange3-3.40.0.zip`）。
3. 将压缩包解压到本项目的 `ml_platform/orange3/` 目录下。
4. 确认解释器最终位于：

```text
ml_platform/orange3/Orange3-3.40.0/Orange/python.exe
```

正确的目录结构应为：

```text
ml_platform/
└── orange3/
    └── Orange3-3.40.0/
        └── Orange/
            ├── python.exe
            └── pythonw.exe
```

在项目根目录执行以下命令，将根目录 `requirements.txt` 中的依赖安装到 Portable Orange：

```powershell
.\ml_platform\orange3\Orange3-3.40.0\Orange\python.exe `
  -m pip install -r .\requirements.txt
```

后续启动根目录 `run.py` 时必须继续使用同一个解释器。

### 嵌入模型

运行组件和案例检索前，请下载嵌入模型：

```bash
mkdir -p embedding_models
# 从 Hugging Face 下载模型并放置到：
# embedding_models/multilingual-e5-large
```

默认模型：[`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)。

如果使用 `embedding_models/embedding_model.py` 支持的其他检索器，请将其放置在 `embedding_models/<retriever-name>` 目录下。

## 配置

所有用户配置均放在根目录 `config.py` 中。MLPlatAgent 不使用命令行参数选择平台或传入凭据。

### 公共配置

选择平台并配置 LLM：

```python
PLATFORM = "uniplore"  # 或 "orange3"

Model_PATH = "<openai-compatible-base-url>"
MODEL_NAME = "<chat-model-name>"
API_KEY = "<llm-api-key>"
RANDOM_SEED = 42
```

如需复现实验，可部署开源的 [`Qwen2.5-72B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) 或实验所需骨架模型，并通过兼容 OpenAI 的接口提供服务。如只进行功能测试，可以通过相同字段配置其他兼容 LLM。

### Uniplore 配置

将 `PLATFORM` 设置为 `"uniplore"`，然后配置远程工作流和 MySQL 数据源：

```python
UNIPLORE_CONFIG = {
    "access_token": "<platform-access-token>",
    "workflow_id": <platform-workflow-id>,
    "api_url": "<platform-api-base-url>",
}

MySQL_Config = {
    "server": "<host reachable by Uniplore>",
    "port": "3306",
    "username": "<mysql-user>",
    "password": "<mysql-password>",
    "database": "mlagent",
}
```

访问令牌必须具有修改和执行指定工作流的权限。MySQL 数据库必须同时能够被 MLPlatAgent 和 Uniplore 服务访问。

如果任务需要 benchmark 数据表，可按需创建数据库并导入公开的 MySQL 转储文件：

```bash
mysql -u <user> -p -e "CREATE DATABASE IF NOT EXISTS mlagent DEFAULT CHARACTER SET utf8mb4;"
mysql -u <user> -p mlagent < data/benchmark/datasets/ml_benchmark.sql
mysql -u <user> -p mlagent < data/benchmark/datasets/dseval_kaggle.sql
mysql -u <user> -p mlagent < data/benchmark/datasets/mlagent.sql
```

### Orange3 配置

将 `PLATFORM` 设置为 `"orange3"`，然后配置本地工作流路径和 PostgreSQL 数据源：

```python
ORANGE3_CONFIG = {
    "workflow_path": r"D:\workflows\current.ows",
}

PG_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "mlagent",
    "schema": "public",
    "user": "postgres",
    "password": "<postgres-password>",
}
```

`workflow_path` 必须以 `.ows` 结尾。必要时程序会创建父目录和空工作流。Orange3 不读取或校验 `UNIPLORE_CONFIG`。只有工作流使用 SQL Table 时才需要 `PG_CONFIG`；使用示例数据集时无需数据库连接。

## 启动 MLPlatAgent

根目录 `run.py` 是项目唯一的正式入口。修改该文件中的 `requirement` 字符串，在 `config.py` 中配置当前平台，然后使用对应命令启动。

### 使用 Uniplore 启动

```powershell
python .\run.py
```

### 使用 Orange3 启动

```powershell
.\ml_platform\orange3\Orange3-3.40.0\Orange\python.exe .\run.py
```

Orange3 会将生成的工作流持久化到 `ORANGE3_CONFIG["workflow_path"]` 配置的 `.ows` 文件中。任务完成后，可以使用 Orange Canvas 打开该文件，查看工作流节点、参数和连接关系。请观看 [Orange3 工作流查看演示视频](static/orange3_demo.mp4)，了解如何打开并查看生成的 `.ows` 工作流。

不要使用 Conda 环境或系统 Python 启动 Orange3 路径。修改 `PLATFORM` 后，应结束当前 Python 进程并重新运行 `run.py`。

`run.py` 中的默认请求用于构建银行客户流失预测模型，脚本会以 JSON 输出 Agent 结果。简短演示视频位于 <https://youtu.be/aN-5xPOluyU>。

## 许可证

本项目采用 MIT 许可证发布。详情请参阅 [LICENSE](LICENSE)。
