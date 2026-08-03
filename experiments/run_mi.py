"""Run every Uniplore Modification Instructions (MI) method in one process.

All settings are defined in this file and the root ``config.py``.  The script
uses no command-line arguments.  It runs MLPlatAgent and all six baselines
sequentially, checkpoints one row after every task, and resumes incomplete
five-run experiments without duplicating completed rows.
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

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "DSEval-Kaggle-Ext"
    / "MI.json"
)
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "08_mi"
EXPECTED_TASKS = 21
REPETITIONS = 5
CASE_NUMBER = 3
RETRIEVER = "multilingual-e5-large"

# Every method writes an independent checkpoint below
# ``<model>/<method>/mi.csv``.  Display names follow the paper; directory
# names avoid characters such as "/" that are invalid in paths.
METHOD_LAYOUT = (
    # ("MLPlatAgent", "MLPlatAgent"),
    ("w/o planning", "wo_planning"),
    ("w/o tool retrieval", "wo_tool_retrieval"),
    ("workflow construction", "workflow_construction"),
    # ("ReAct", "ReAct"),
    # ("Function Call", "FunctionCall"),
    # ("DFSDT", "DFSDT"),
)

# The first value is sent to the configured OpenAI-compatible endpoint.  The
# second value is the stable directory name used for experiment artifacts.
MODELS = (
    # ("qwen2_5-14b-coder", "qwen2_5-14b-coder"),
    ("qwen2.5-72b", "qwen2_5-72b"),
)

# Preserve every field of experiments/06_tool_retrieve_type/Default/mi.csv,
# then append the three MI metrics and the raw edit sets needed to recalculate
# those metrics without rerunning the Agent.
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
    "wep",
    "wer",
    "wef1",
    "predicted_workflow_edit_sets",
    "ground_truth_workflow_edit_sets",
)


@dataclass(frozen=True)
class MethodSpec:
    """One MI method and its isolated result directory."""

    name: str
    directory_name: str
    agent_class: type


def _load_method_specs(mlplatagent_class: type) -> tuple[MethodSpec, ...]:
    """Load all baseline modules only after platform validation."""
    from baselines import get_agent_classes

    baseline_classes = get_agent_classes()
    classes = {
        "MLPlatAgent": mlplatagent_class,
        **baseline_classes,
    }
    specs = tuple(
        MethodSpec(
            name=name,
            directory_name=directory_name,
            agent_class=classes[directory_name],
        )
        for name, directory_name in METHOD_LAYOUT
    )
    if len({spec.directory_name for spec in specs}) != len(specs):
        raise RuntimeError("MI method result directories must be unique")
    return specs


def _load_benchmark(
    path: Path = BENCHMARK_PATH,
) -> list[dict[str, Any]]:
    """Load and validate all fields needed by the MI experiment."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"MI benchmark does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"MI benchmark is not valid JSON: {path} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc

    if not isinstance(data, list):
        raise RuntimeError(f"MI benchmark root must be a list: {path}")
    if len(data) != EXPECTED_TASKS:
        raise RuntimeError(
            f"MI benchmark must contain {EXPECTED_TASKS} tasks, got {len(data)}"
        )

    instructions: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise RuntimeError(f"MI item {index} must be an object")
        instruction = item.get("Instruction")
        input_workflow = item.get("Input Workflow")
        requirements = item.get("requirements")
        edit_set = item.get("workflow edit sets")
        if not isinstance(instruction, str) or not instruction.strip():
            raise RuntimeError(f"MI item {index} has no valid Instruction")
        if instruction in instructions:
            raise RuntimeError(
                f"MI benchmark contains duplicate Instruction: {instruction}"
            )
        if not isinstance(input_workflow, str) or not input_workflow.strip():
            raise RuntimeError(
                f"MI item {index} has no valid Input Workflow"
            )
        if not isinstance(requirements, dict):
            raise RuntimeError(f"MI item {index} has no requirements object")
        if (
            not isinstance(edit_set, list)
            or not edit_set
            or not all(
                isinstance(operation, str) and operation.strip()
                for operation in edit_set
            )
        ):
            raise RuntimeError(
                f"MI item {index} has no valid workflow edit sets"
            )
        instructions.add(instruction)

    return data


def _load_existing_rows(path: Path) -> list[dict[str, str]]:
    """Load a checkpoint only when its schema is exactly the expected schema."""
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


def _write_rows(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Atomically checkpoint a complete MI result CSV."""
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
    """Aggregate Planner, Executor, and Action LLM accounting."""
    planner = getattr(agent, "planner", None)
    executor = getattr(agent, "executor", None)
    input_tokens = (
        int(getattr(planner, "input_tokens", 0))
        + int(getattr(executor, "input_tokens", 0))
        + int(getattr(action_agent, "input_tokens", 0))
    )
    output_tokens = (
        int(getattr(planner, "output_tokens", 0))
        + int(getattr(executor, "output_tokens", 0))
        + int(getattr(action_agent, "output_tokens", 0))
    )
    return input_tokens, output_tokens


def _time_totals(
    agent: Any,
    action_agent: Any,
    elapsed: float,
) -> dict[str, float]:
    """Adapt current timing state to the historical MI CSV schema."""
    executor = getattr(agent, "executor", None)
    executor_times = getattr(executor, "time_cost", {})
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


def _build_input_workflow(input_workflow: str) -> None:
    """Execute the trusted benchmark setup outside the Agent edit recorder."""
    namespace = {"__name__": "__mi_input_workflow__"}
    exec(compile(input_workflow, str(BENCHMARK_PATH), "exec"), namespace)


def _run_instruction(
    *,
    item: Mapping[str, Any],
    model_name: str,
    MLAgent: type,
    action_agent: Any,
    project_config: Any,
    llm_module: Any,
    RCR: Any,
    workflow_edit_metrics: Any,
) -> dict[str, Any]:
    """Build one input workflow, modify it, and return one complete CSV row."""
    instruction = item["Instruction"]
    requirements = item["requirements"]
    ground_truth_edit_sets = item["workflow edit sets"]

    project_config.MODEL_NAME = model_name
    llm_module.MODEL_NAME = model_name

    action_agent.reset()
    action_agent.clear_workflow()
    # A setup failure invalidates the task run, so let it stop the experiment
    # instead of silently checkpointing a result for a malformed input graph.
    _build_input_workflow(item["Input Workflow"])

    agent = MLAgent(tool_retrieve_type=0)
    predicted_edit_sets: list[str] = []
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
        predicted_edit_sets = result.get("workflow_edit_sets", [])
        if not isinstance(predicted_edit_sets, list):
            raise RuntimeError(
                "MLAgent.run() returned invalid workflow_edit_sets"
            )
        if not result.get("is_success", False):
            print(
                "  Agent reported an incomplete modification: "
                f"{result.get('error', 'one or more tasks failed')}",
                flush=True,
            )
    except Exception as exc:
        predicted_edit_sets = (
            agent.executor.get_workflow_edit_sets()
            if hasattr(agent.executor, "get_workflow_edit_sets")
            else []
        )
        print(
            f"  Agent raised {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
    elapsed = time.perf_counter() - started_at

    workflow = _safe_workflow(action_agent)
    input_tokens, output_tokens = _token_totals(agent, action_agent)
    time_values = _time_totals(agent, action_agent, elapsed)
    rcr = RCR(requirements=requirements, workflow=workflow)
    wep, wer, wef1 = workflow_edit_metrics(
        predicted_edit_sets,
        ground_truth_edit_sets,
    )

    return {
        "instruction": instruction,
        "rcr": rcr,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        **time_values,
        "workflow": json.dumps(workflow, ensure_ascii=False),
        "requirements": json.dumps(requirements, ensure_ascii=False),
        "wep": wep,
        "wer": wer,
        "wef1": wef1,
        "predicted_workflow_edit_sets": json.dumps(
            predicted_edit_sets,
            ensure_ascii=False,
        ),
        "ground_truth_workflow_edit_sets": json.dumps(
            ground_truth_edit_sets,
            ensure_ascii=False,
        ),
    }


def _validate_checkpoint(
    rows: list[dict[str, str]],
    benchmark: list[dict[str, Any]],
    result_path: Path,
) -> Counter:
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
            f"present in {BENCHMARK_PATH}"
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
    return completed_counts


def _result_path(
    model_directory: str,
    method_directory: str,
) -> Path:
    """Return the isolated checkpoint path for one model and method."""
    return (
        OUTPUT_ROOT
        / model_directory
        / method_directory
        / "mi.csv"
    )


def _prepare_result_path(
    method: MethodSpec,
    model_directory: str,
    benchmark: list[dict[str, Any]],
) -> Path:
    """Create the canonical path and preserve a legacy full-method checkpoint."""
    model_root = OUTPUT_ROOT / model_directory
    result_path = _result_path(
        model_directory,
        method.directory_name,
    )
    legacy_path = model_root / "mi.csv"
    if (
        method.directory_name == "MLPlatAgent"
        and not result_path.exists()
        and legacy_path.exists()
    ):
        legacy_rows = _load_existing_rows(legacy_path)
        _validate_checkpoint(legacy_rows, benchmark, legacy_path)
        _write_rows(result_path, legacy_rows)
        print(
            "  migrated legacy MLPlatAgent checkpoint: "
            f"{legacy_path} -> {result_path}",
            flush=True,
        )
    return result_path


def _run_model(
    *,
    method: MethodSpec,
    model_name: str,
    directory_name: str,
    benchmark: list[dict[str, Any]],
    runtime: Mapping[str, Any],
) -> None:
    result_path = _prepare_result_path(
        method,
        directory_name,
        benchmark,
    )
    rows = _load_existing_rows(result_path)
    completed_counts = _validate_checkpoint(
        rows,
        benchmark,
        result_path,
    )
    expected_rows = len(benchmark) * REPETITIONS

    print(
        f"\n=== method: {method.name}, model: {model_name}, "
        f"checkpoint={len(rows)}/{expected_rows} ===",
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
                model_name=model_name,
                MLAgent=method.agent_class,
                **runtime,
            )
            rows.append(row)
            completed_counts[instruction] += 1
            _write_rows(result_path, rows)

    incomplete_counts = {
        instruction: completed_counts[instruction]
        for instruction in (
            item["Instruction"] for item in benchmark
        )
        if completed_counts[instruction] != REPETITIONS
    }
    if len(rows) != expected_rows or incomplete_counts:
        raise RuntimeError(
            f"MI result row count mismatch: expected {expected_rows}, "
            f"got {len(rows)}; incomplete instructions: "
            f"{len(incomplete_counts)}"
        )

    mean_wep = sum(float(row["wep"]) for row in rows) / len(rows)
    mean_wer = sum(float(row["wer"]) for row in rows) / len(rows)
    mean_wef1 = sum(float(row["wef1"]) for row in rows) / len(rows)
    print(
        f"  completed: {result_path}\n"
        f"  WEP={mean_wep:.4f}, WER={mean_wer:.4f}, "
        f"WEF1={mean_wef1:.4f}",
        flush=True,
    )


def _validate_runtime_config(
    project_config: Any,
    action_agent: Any,
) -> None:
    """Validate only Uniplore platform identity, not any database config."""
    if getattr(project_config, "PLATFORM", None) != "uniplore":
        raise RuntimeError(
            "experiments/run_mi.py only runs Uniplore experiments. "
            "Set PLATFORM = 'uniplore' in the root config.py and restart "
            "Python."
        )
    if getattr(action_agent, "platform_name", None) != "uniplore":
        raise RuntimeError("The global action_agent is not a Uniplore Action")

    platform_config = getattr(project_config, "UNIPLORE_CONFIG", None)
    if not isinstance(platform_config, dict):
        raise RuntimeError("config.UNIPLORE_CONFIG must be a dictionary")
    required_fields = {"access_token", "workflow_id", "api_url"}
    missing_fields = sorted(required_fields - set(platform_config))
    if missing_fields:
        raise RuntimeError(
            "config.UNIPLORE_CONFIG is missing fields: "
            + ", ".join(missing_fields)
        )
    empty_fields = sorted(
        field
        for field in required_fields
        if platform_config.get(field) in (None, "")
    )
    if empty_fields:
        raise RuntimeError(
            "config.UNIPLORE_CONFIG has empty fields: "
            + ", ".join(empty_fields)
        )


def main() -> None:
    """Run all 21 MI tasks five times for each configured model."""
    import config as project_config

    # Validate the selected platform before importing the dynamic Action
    # factory, so a mismatched config fails without initializing Orange3.
    if getattr(project_config, "PLATFORM", None) != "uniplore":
        raise RuntimeError(
            "experiments/run_mi.py only supports Uniplore. "
            "Set PLATFORM = 'uniplore' in config.py."
        )

    import llm.llm as llm_module
    from agents.mlagent import MLAgent
    from utils.evaluate_rcr import RCR
    from utils.evaluate_we import workflow_edit_metrics
    from ml_platform.actions import action_agent

    _validate_runtime_config(project_config, action_agent)
    benchmark = _load_benchmark()

    # Validate annotation syntax before making any remote workflow changes.
    for item in benchmark:
        perfect_score = workflow_edit_metrics(
            item["workflow edit sets"],
            item["workflow edit sets"],
        )
        if perfect_score != (1.0, 1.0, 1.0):
            raise RuntimeError(
                "Invalid workflow edit sets annotation for instruction: "
                f"{item['Instruction']}"
            )

    runtime = {
        "action_agent": action_agent,
        "project_config": project_config,
        "llm_module": llm_module,
        "RCR": RCR,
        "workflow_edit_metrics": workflow_edit_metrics,
    }

    print(
        "Starting Uniplore Modification Instructions experiments: "
        f"{len(METHOD_LAYOUT)} methods x {len(benchmark)} tasks x "
        f"{REPETITIONS} repetitions.",
        flush=True,
    )
    methods = _load_method_specs(MLAgent)
    # Methods run sequentially because every run targets the same configured
    # Uniplore workflow ID.  Parallel execution would corrupt shared state.
    for model_name, directory_name in MODELS:
        for method in methods:
            _run_model(
                method=method,
                model_name=model_name,
                directory_name=directory_name,
                benchmark=benchmark,
                runtime=runtime,
            )

    print(
        f"\nAll MI experiments completed. Results: {OUTPUT_ROOT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
