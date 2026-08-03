"""Interactively review and update workflow edit sets in MI.json.

Run from the project root:

    python data/benchmark/DSEval-Kaggle-Ext/review_workflow_edit_sets.py

The benchmark's Input Workflow definitions use Uniplore widgets. Set
``PLATFORM = "uniplore"`` in the root ``config.py`` before running this
program. The configured Uniplore workflow is cleared and rebuilt for every
review item.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_PATH = Path(__file__).with_name("MI.json")
EXPECTED_TASKS = 21
WORKFLOW_OPERATIONS = {
    "add_node",
    "delete_node",
    "update_node_params",
    "add_edge",
    "delete_edge",
}


def _load_benchmark(path: Path = BENCHMARK_PATH) -> list[dict[str, Any]]:
    try:
        benchmark = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"找不到 MI 数据文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"MI.json 不是有效 JSON：第 {exc.lineno} 行，"
            f"第 {exc.colno} 列"
        ) from exc

    if not isinstance(benchmark, list):
        raise RuntimeError("MI.json 的顶层结构必须是列表")
    if len(benchmark) != EXPECTED_TASKS:
        raise RuntimeError(
            f"MI.json 应包含 {EXPECTED_TASKS} 个任务，"
            f"实际为 {len(benchmark)} 个"
        )

    for index, item in enumerate(benchmark, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"第 {index} 个任务不是 JSON 对象")
        for field in ("Instruction", "Input Workflow", "requirements"):
            if field not in item:
                raise RuntimeError(f"第 {index} 个任务缺少字段：{field}")
        _validate_edit_sets(
            item.get("workflow edit sets"),
            label=f"第 {index} 个任务的 workflow edit sets",
        )
    return benchmark


def _validate_edit_sets(value: Any, *, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(
            isinstance(operation, str) and operation.strip()
            for operation in value
        )
    ):
        raise ValueError(f"{label}必须是非空字符串列表")

    for index, operation in enumerate(value, start=1):
        try:
            syntax_tree = ast.parse(operation)
        except SyntaxError as exc:
            raise ValueError(
                f"{label}中的第 {index} 项不是有效 Python 代码：{exc.msg}"
            ) from exc

        call_names = [
            node.func.id
            for node in ast.walk(syntax_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        ]
        if len(call_names) != 1 or call_names[0] not in WORKFLOW_OPERATIONS:
            raise ValueError(
                f"{label}中的第 {index} 项必须只包含一次受支持的工作流"
                f"操作：{', '.join(sorted(WORKFLOW_OPERATIONS))}"
            )

    return [operation.strip() for operation in value]


def _save_benchmark(
    benchmark: list[dict[str, Any]],
    path: Path = BENCHMARK_PATH,
) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(benchmark, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_action_agent() -> tuple[Any, int | str | None]:
    import config as project_config

    if getattr(project_config, "PLATFORM", None) != "uniplore":
        raise RuntimeError(
            "该 MI.json 使用 Uniplore 组件。请先在根目录 config.py 中"
            "设置 PLATFORM = \"uniplore\"，然后重新启动脚本。"
        )

    from ml_platform.actions import action_agent

    if getattr(action_agent, "platform_name", None) != "uniplore":
        raise RuntimeError("全局 action_agent 不是 Uniplore PlatformAction")

    platform_config = getattr(project_config, "UNIPLORE_CONFIG", {})
    workflow_id = (
        platform_config.get("workflow_id")
        if isinstance(platform_config, dict)
        else None
    )
    return action_agent, workflow_id


def _execute_input_workflow(
    input_workflow: str,
    *,
    task_index: int,
    action_agent: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    action_agent.reset()
    action_agent.clear_workflow()

    namespace = {"__name__": f"__mi_review_task_{task_index}__"}
    try:
        exec(
            compile(
                input_workflow,
                f"{BENCHMARK_PATH}::Input Workflow[{task_index}]",
                "exec",
            ),
            namespace,
        )
    except Exception as exc:
        raise RuntimeError(
            f"第 {task_index} 个任务的 Input Workflow 执行失败："
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if namespace.get("action_agent") is not action_agent:
        raise RuntimeError(
            f"第 {task_index} 个 Input Workflow 未使用全局 action_agent"
        )

    workflow = action_agent.get_workflow()
    if (
        not isinstance(workflow, dict)
        or not isinstance(workflow.get("nodes"), list)
        or not isinstance(workflow.get("edges"), list)
        or not workflow["nodes"]
    ):
        raise RuntimeError(
            f"第 {task_index} 个 Input Workflow 未复原出有效工作流"
        )

    identifiers = {
        name: value
        for name, value in namespace.items()
        if name.endswith(("_node_id", "_edge_id"))
    }
    return workflow, identifiers


def _print_review_item(
    *,
    index: int,
    total: int,
    item: dict[str, Any],
    workflow: dict[str, Any],
    identifiers: dict[str, Any],
) -> None:
    print("\n" + "=" * 78)
    print(f"任务 {index}/{total}")
    print("=" * 78)
    print(f"Instruction：{item['Instruction']}")

    print("\nInput Workflow 变量：")
    for name, value in identifiers.items():
        print(f"  {name} = {value}")

    print("\n复原后的节点：")
    for node in workflow["nodes"]:
        node_id = node.get("node_id", "")
        widget_name = node.get("widget_name", "")
        node_name = node.get("node_name", "")
        node_params = node.get("node_params", {})
        print(f"  [{node_id}] {widget_name} | {node_name}")
        if node_params:
            print(
                "      params = "
                + json.dumps(node_params, ensure_ascii=False)
            )

    print("\n复原后的边：")
    if workflow["edges"]:
        for edge in workflow["edges"]:
            edge_id = edge.get("edge_id", "")
            source = edge.get("source_node_id", "")
            target = edge.get("target_node_id", "")
            print(f"  [{edge_id}] {source} -> {target}")
    else:
        print("  （无）")

    print("\n当前 workflow edit sets：")
    for operation_index, operation in enumerate(
        item["workflow edit sets"],
        start=1,
    ):
        print(f"  {operation_index}. {operation}")


def _read_replacement_edit_sets() -> list[str] | None:
    print(
        "\n请逐行输入新的工作流编辑操作。输入 END 完成，"
        "输入 CANCEL 取消本次修改。"
    )
    operations: list[str] = []
    while True:
        value = input(f"edit[{len(operations) + 1}]> ").strip()
        command = value.upper()
        if command == "CANCEL":
            return None
        if command == "END":
            try:
                return _validate_edit_sets(
                    operations,
                    label="新的 workflow edit sets",
                )
            except ValueError as exc:
                print(f"输入无效：{exc}")
                print("请重新输入整个 workflow edit sets。")
                operations = []
                continue
        if value:
            operations.append(value)
        else:
            print("操作不能为空；请输入工作流操作、END 或 CANCEL。")


def _review_decision() -> str:
    while True:
        decision = input(
            "\n审核结果（1=通过，0=修改，q=保存当前结果并退出）："
        ).strip().lower()
        if decision in {"1", "0", "q"}:
            return decision
        print("请输入 1、0 或 q。")


def main() -> None:
    benchmark = _load_benchmark()
    action_agent, workflow_id = _load_action_agent()

    print(f"将审核 {BENCHMARK_PATH}")
    print(f"共 {len(benchmark)} 个任务。")
    print(
        "执行期间会反复清空并复原配置指向的 Uniplore 工作流"
        f"（workflow_id={workflow_id}）。"
    )

    try:
        for index, item in enumerate(benchmark, start=1):
            workflow, identifiers = _execute_input_workflow(
                item["Input Workflow"],
                task_index=index,
                action_agent=action_agent,
            )
            _print_review_item(
                index=index,
                total=len(benchmark),
                item=item,
                workflow=workflow,
                identifiers=identifiers,
            )

            while True:
                decision = _review_decision()
                if decision == "q":
                    print("审核已提前结束；已经写入的修改均已保存。")
                    return
                if decision == "1":
                    break

                replacement = _read_replacement_edit_sets()
                if replacement is None:
                    print("已取消本次修改，请重新选择审核结果。")
                    continue
                item["workflow edit sets"] = replacement
                _save_benchmark(benchmark)
                print("新的 workflow edit sets 已写入 MI.json。")
                break
    finally:
        action_agent.clear_workflow()

    print("\n21 个任务均已审核完成。")


if __name__ == "__main__":
    main()
