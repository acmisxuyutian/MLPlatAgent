"""Run the Orange3 portability experiments for all supported benchmarks.

All experiment settings are constants in this file.  No command-line
arguments are used.  Results are checkpointed after every instruction and can
be resumed by running this script again.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "07_portability" / "orange3"
REPETITIONS = 5
CASE_NUMBER = 3
RETRIEVER = "multilingual-e5-large"

# The first value is sent to the configured OpenAI-compatible endpoint.  The
# second value is the stable directory name used for experiment artifacts.
# MODELS = (
#     ("qwen2_5-14b-coder", "qwen2_5-14b-coder"),
#     ("qwen2.5-72b", "qwen2_5-72b"),
# )
MODELS = (
    ("qwen2.5-72b", "qwen2_5-72b"),
)

CSV_FIELDS = (
    "instruction",
    "rcr",
    "input_tokens",
    "output_tokens",
    "times",
    "time_task_decomposition",
    "time_widget_retrieval",
    "time_operation_sequence_generation",
    "time_data_retrieval",
    "workflow",
    "requirements",
    "wgf1",
)


@dataclass(frozen=True)
class DatasetSpec:
    """One result file within a benchmark suite."""

    name: str
    source_path: Path
    is_modification: bool = False


@dataclass(frozen=True)
class BenchmarkSuite:
    """A benchmark family and the PostgreSQL database it uses."""

    name: str
    database: str
    datasets: tuple[DatasetSpec, ...]


BENCHMARK_SUITES = (
    BenchmarkSuite(
        name="dseval",
        database="dseval_kaggle",
        datasets=(
            DatasetSpec(
                name="dseval",
                source_path=PROJECT_ROOT
                / "data"
                / "benchmark"
                / "dseval_kaggle.json",
            ),
        ),
    ),
    BenchmarkSuite(
        name="mlb",
        database="ml_benchmark",
        datasets=(
            DatasetSpec(
                name="mlb",
                source_path=PROJECT_ROOT
                / "data"
                / "benchmark"
                / "ml_benchmark.json",
            ),
        ),
    ),
    BenchmarkSuite(
        name="nl2workflow",
        database="mlagent",
        datasets=(
            DatasetSpec(
                name="uci",
                source_path=PROJECT_ROOT
                / "data"
                / "benchmark"
                / "DSEval-Kaggle-Ext"
                / "UCI.json",
            ),
            DatasetSpec(
                name="ci",
                source_path=PROJECT_ROOT
                / "data"
                / "benchmark"
                / "DSEval-Kaggle-Ext"
                / "CI.json",
            ),
            DatasetSpec(
                name="mi",
                source_path=PROJECT_ROOT
                / "experiments"
                / "07_portability"
                / "orange3"
                / "benchmark"
                / "DSEval-Kaggle-Ext"
                / "MI.json",
                is_modification=True,
            ),
        ),
    ),
)


def _load_benchmark(
    path: Path,
    *,
    is_modification: bool = False,
) -> list[dict[str, Any]]:
    """Load and validate the fields consumed by the experiment runner."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Benchmark file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Benchmark file is not valid JSON: {path} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc

    if not isinstance(data, list):
        raise RuntimeError(f"Benchmark root must be a list: {path}")

    instructions: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"Benchmark item {index} in {path} must be an object"
            )
        instruction = item.get("Instruction")
        requirements = item.get("requirements")
        if not isinstance(instruction, str) or not instruction.strip():
            raise RuntimeError(
                f"Benchmark item {index} in {path} has no valid Instruction"
            )
        if not isinstance(requirements, dict):
            raise RuntimeError(
                f"Benchmark item {index} in {path} has no requirements object"
            )
        if is_modification:
            input_workflow = item.get("Input Workflow")
            if (
                not isinstance(input_workflow, str)
                or not input_workflow.strip()
            ):
                raise RuntimeError(
                    f"MI benchmark item {index} in {path} has no valid "
                    "Input Workflow"
                )
        if instruction in instructions:
            raise RuntimeError(
                f"Benchmark contains a duplicate Instruction: {path}"
            )
        instructions.add(instruction)

    return data


def _load_existing_rows(path: Path) -> list[dict[str, str]]:
    """Read a checkpoint and reject a schema that could corrupt new results."""
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        actual_fields = tuple(reader.fieldnames or ())
        if actual_fields != CSV_FIELDS:
            raise RuntimeError(
                f"Unexpected CSV schema in {path}. "
                f"Expected {list(CSV_FIELDS)}, got {list(actual_fields)}"
            )
        return list(reader)


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically checkpoint a complete result CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in CSV_FIELDS})
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _empty_workflow() -> dict[str, list[dict[str, Any]]]:
    return {"nodes": [], "edges": []}


def _safe_workflow(action_agent: Any) -> dict[str, Any]:
    """Return a normalized snapshot even after an unexpected Agent failure."""
    try:
        workflow = action_agent.get_workflow()
    except Exception:
        return _empty_workflow()
    if not isinstance(workflow, dict):
        return _empty_workflow()
    if not isinstance(workflow.get("nodes"), list):
        return _empty_workflow()
    if not isinstance(workflow.get("edges"), list):
        return _empty_workflow()
    return workflow


def _token_totals(agent: Any, action_agent: Any) -> tuple[int, int]:
    """Aggregate Planner, Executor and retrieval LLM accounting."""
    input_tokens = (
        int(getattr(agent.planner, "input_tokens", 0))
        + int(getattr(agent.executor, "input_tokens", 0))
        + int(getattr(action_agent, "input_tokens", 0))
    )
    output_tokens = (
        int(getattr(agent.planner, "output_tokens", 0))
        + int(getattr(agent.executor, "output_tokens", 0))
        + int(getattr(action_agent, "output_tokens", 0))
    )
    return input_tokens, output_tokens


def _time_totals(
    agent: Any,
    action_agent: Any,
    elapsed: float,
) -> dict[str, float]:
    """Adapt current Agent timing state to the historical CSV schema.

    Executor's operation timer includes execution of generated calls, including
    data retrieval.  Subtracting the separately measured retrieval duration
    prevents that duration from being counted twice.  The task-decomposition
    value is the remaining wall-clock duration, so the four fields add up
    exactly to ``times``.
    """
    executor_times = getattr(agent.executor, "time_cost", {})
    widget_retrieval = max(
        0.0,
        float(executor_times.get("widget_retrieval", 0.0)),
    )
    data_retrieval = max(
        0.0,
        float(getattr(action_agent, "data_retrieval_time", 0.0)),
    )
    raw_operation = max(
        0.0,
        float(
            executor_times.get(
                "operation_sequence_generation",
                0.0,
            )
        ),
    )
    operation_generation = max(0.0, raw_operation - data_retrieval)
    task_decomposition = max(
        0.0,
        elapsed
        - widget_retrieval
        - operation_generation
        - data_retrieval,
    )
    total = (
        task_decomposition
        + widget_retrieval
        + operation_generation
        + data_retrieval
    )
    return {
        "times": total,
        "time_task_decomposition": task_decomposition,
        "time_widget_retrieval": widget_retrieval,
        "time_operation_sequence_generation": operation_generation,
        "time_data_retrieval": data_retrieval,
    }


def _build_input_workflow(
    input_workflow: str,
    *,
    source_path: Path,
    action_agent: Any,
) -> None:
    """Reconstruct the trusted MI input graph before invoking the Agent."""
    namespace = {"__name__": "__mi_input_workflow__"}
    exec(compile(input_workflow, str(source_path), "exec"), namespace)

    # Input Workflow code must operate on the same process-wide Action instance
    # used by the Agent and by the experiment runner.
    if namespace.get("action_agent") is not action_agent:
        raise RuntimeError(
            f"Input Workflow in {source_path} did not use the global "
            "ml_platform.actions.action_agent"
        )
    workflow = _safe_workflow(action_agent)
    if not workflow["nodes"]:
        raise RuntimeError(
            f"Input Workflow in {source_path} reconstructed an empty workflow"
        )


def _run_instruction(
    *,
    item: Mapping[str, Any],
    source_path: Path,
    is_modification: bool,
    model_name: str,
    database: str,
    MLAgent: type,
    action_agent: Any,
    project_config: Any,
    llm_module: Any,
    RCR: Any,
    workflow_graph_f1: Any,
) -> dict[str, Any]:
    """Run one instruction and build one complete CSV record."""
    instruction = item["Instruction"]
    requirements = item["requirements"]

    project_config.MODEL_NAME = model_name
    llm_module.MODEL_NAME = model_name
    project_config.PG_CONFIG["database"] = database
    action_agent.pg_config["database"] = database

    action_agent.reset()
    action_agent.clear_workflow()
    if is_modification:
        # Setup is intentionally outside the Agent timing and failure handler:
        # a malformed input workflow invalidates the MI run and must not be
        # checkpointed as an Agent prediction.
        _build_input_workflow(
            item["Input Workflow"],
            source_path=source_path,
            action_agent=action_agent,
        )
    agent = MLAgent(tool_retrieve_type=0)

    started_at = time.perf_counter()
    try:
        result = agent.run(
            Instruction=instruction,
            case_number=CASE_NUMBER,
            retriever=RETRIEVER,
        )
        if not isinstance(result, dict):
            raise RuntimeError(
                "MLAgent.run() returned an unsupported result type: "
                f"{type(result).__name__}"
            )
        if not result.get("is_success", False):
            print(
                "  Agent reported an incomplete workflow: "
                f"{result.get('error', 'one or more tasks failed')}",
                flush=True,
            )
    except Exception as exc:
        print(
            f"  Agent raised {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
    elapsed = time.perf_counter() - started_at

    workflow = _safe_workflow(action_agent)
    input_tokens, output_tokens = _token_totals(agent, action_agent)
    time_values = _time_totals(agent, action_agent, elapsed)

    workflow_json = json.dumps(workflow, ensure_ascii=False)
    requirements_json = json.dumps(requirements, ensure_ascii=False)
    rcr = RCR(requirements=requirements, workflow=workflow)
    wgf1 = workflow_graph_f1(workflow_json, requirements_json)

    return {
        "instruction": instruction,
        "rcr": rcr,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        **time_values,
        "workflow": workflow_json,
        "requirements": requirements_json,
        "wgf1": wgf1,
    }


def _run_dataset(
    *,
    dataset: DatasetSpec,
    database: str,
    model_name: str,
    model_directory: Path,
    runtime: Mapping[str, Any],
) -> None:
    benchmark = _load_benchmark(
        dataset.source_path,
        is_modification=dataset.is_modification,
    )
    result_path = model_directory / f"{dataset.name}.csv"
    rows = _load_existing_rows(result_path)
    completed_counts = Counter(row["instruction"] for row in rows)
    expected_instructions = {
        item["Instruction"]
        for item in benchmark
    }
    unknown_instructions = sorted(
        set(completed_counts) - expected_instructions
    )
    if unknown_instructions:
        raise RuntimeError(
            f"Checkpoint {result_path} contains instructions that are not "
            f"present in {dataset.source_path}"
        )
    excessive_instructions = sorted(
        instruction
        for instruction, count in completed_counts.items()
        if count > REPETITIONS
    )
    if excessive_instructions:
        raise RuntimeError(
            f"Checkpoint {result_path} contains more than {REPETITIONS} "
            "rows for at least one instruction"
        )
    expected_rows = len(benchmark) * REPETITIONS

    print(
        f"\n[{dataset.name}] database={database}, "
        f"checkpoint={len(rows)}/{expected_rows}",
        flush=True,
    )
    for repetition in range(REPETITIONS):
        for item_index, item in enumerate(benchmark, start=1):
            instruction = item["Instruction"]
            if completed_counts[instruction] > repetition:
                continue

            print(
                f"  repetition {repetition + 1}/{REPETITIONS}, "
                f"item {item_index}/{len(benchmark)}",
                flush=True,
            )
            row = _run_instruction(
                item=item,
                source_path=dataset.source_path,
                is_modification=dataset.is_modification,
                model_name=model_name,
                database=database,
                **runtime,
            )
            rows.append(row)
            completed_counts[instruction] += 1
            _write_rows(result_path, rows)

    incomplete_counts = {
        instruction: completed_counts[instruction]
        for instruction in expected_instructions
        if completed_counts[instruction] != REPETITIONS
    }
    if len(rows) != expected_rows or incomplete_counts:
        raise RuntimeError(
            f"Result row count mismatch for {dataset.name}: "
            f"expected {expected_rows}, got {len(rows)}; "
            f"incomplete instructions: {len(incomplete_counts)}"
        )
    print(f"  completed: {result_path}", flush=True)


def _validate_runtime_config(project_config: Any, action_agent: Any) -> None:
    if getattr(project_config, "PLATFORM", None) != "orange3":
        raise RuntimeError(
            "experiments/run.py only runs Orange3 experiments. "
            "Set PLATFORM = 'orange3' in the root config.py and restart Python."
        )
    pg_config = getattr(project_config, "PG_CONFIG", None)
    if not isinstance(pg_config, dict):
        raise RuntimeError("config.PG_CONFIG must be a dictionary")
    required_pg_fields = {
        "host",
        "port",
        "user",
        "password",
        "database",
        "schema",
    }
    missing_fields = sorted(required_pg_fields - set(pg_config))
    if missing_fields:
        raise RuntimeError(
            "config.PG_CONFIG is missing fields: " + ", ".join(missing_fields)
        )
    if getattr(action_agent, "platform_name", None) != "orange3":
        raise RuntimeError("The global action_agent is not an Orange3 Action")
    if not isinstance(getattr(action_agent, "pg_config", None), dict):
        raise RuntimeError("Orange3 action_agent has no PostgreSQL configuration")


def main() -> None:
    """Run DSEval, MLB and NL2Workflow (UCI/CI/MI) on Orange3."""
    # Configure headless Qt before importing the Orange3 platform adapter.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import config as project_config

    if getattr(project_config, "PLATFORM", None) != "orange3":
        raise RuntimeError(
            "experiments/run.py only supports Orange3. "
            "Set PLATFORM = 'orange3' in config.py."
        )

    import llm.llm as llm_module
    from agents.mlagent import MLAgent
    from utils.evaluate_rcr import RCR
    from utils.evaluate_wgf1 import workflow_graph_f1
    from ml_platform.actions import action_agent

    _validate_runtime_config(project_config, action_agent)
    runtime = {
        "MLAgent": MLAgent,
        "action_agent": action_agent,
        "project_config": project_config,
        "llm_module": llm_module,
        "RCR": RCR,
        "workflow_graph_f1": workflow_graph_f1,
    }

    print(
        "Starting Orange3 portability experiments: "
        "DSEval, MLB, NL2Workflow (UCI, CI and MI).",
        flush=True,
    )
    for model_name, directory_name in MODELS:
        model_directory = OUTPUT_ROOT / directory_name
        print(f"\n=== model: {model_name} ===", flush=True)
        for suite in BENCHMARK_SUITES:
            print(
                f"\n--- benchmark: {suite.name} ---",
                flush=True,
            )
            for dataset in suite.datasets:
                _run_dataset(
                    dataset=dataset,
                    database=suite.database,
                    model_name=model_name,
                    model_directory=model_directory,
                    runtime=runtime,
                )

    print(
        f"\nAll portability experiments completed. Results: {OUTPUT_ROOT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
