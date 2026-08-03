"""计算工作流编辑操作的 Precision、Recall 和 F1。"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _Identifier:
    """不关心具体取值的节点或边 ID。"""

    namespace: str
    value: str


@dataclass(frozen=True)
class _Edit:
    """解析后的单次工作流编辑操作。"""

    operation: str
    arguments: tuple[Any, ...]
    result_identifier: _Identifier | None = None


# workflow edit sets 只允许使用 Agent 暴露的五种编辑操作。
_OPERATION_ARGUMENTS = {
    "add_node": ("widget_name", "node_name"),
    "delete_node": ("node_id",),
    "update_node_params": ("node_id", "widget_name", "node_params"),
    "add_edge": ("source_node_id", "target_node_id"),
    "delete_edge": ("edge_id",),
}


def _freeze_value(value: Any) -> Any:
    """将参数转换为类型稳定、可直接比较的不可变结构。"""

    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (
                        _freeze_value(key),
                        _freeze_value(item_value),
                    )
                    for key, item_value in value.items()
                )
            ),
        )
    if isinstance(value, list):
        return (
            "list",
            tuple(_freeze_value(item) for item in value),
        )
    if isinstance(value, tuple):
        return (
            "tuple",
            tuple(_freeze_value(item) for item in value),
        )
    raise ValueError(
        "Workflow edit arguments must contain only JSON-compatible values, "
        f"got {type(value).__name__}"
    )


def _literal_value(expression: ast.expr) -> Any:
    """安全读取组件名、节点名和参数，不执行预测代码。"""

    try:
        value = ast.literal_eval(expression)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Workflow edit values must be Python literals"
        ) from exc
    return _freeze_value(value)


def _identifier(
    expression: ast.expr,
    namespace: str,
) -> _Identifier:
    """统一表示运行时 ID 和标注中的符号 ID。"""

    if isinstance(expression, ast.Name):
        value = f"name:{expression.id}"
    elif (
        isinstance(expression, ast.Constant)
        and isinstance(expression.value, (str, int))
        and not isinstance(expression.value, bool)
    ):
        # 平台快照可能将同一个数字 ID 表示成整数或数字字符串。
        value = f"literal:{expression.value}"
    else:
        raise ValueError(
            "Workflow node and edge IDs must be names, strings, or integers"
        )
    return _Identifier(namespace=namespace, value=value)


def _bind_arguments(
    call: ast.Call,
    operation: str,
) -> dict[str, ast.expr]:
    """按照操作签名绑定 Agent 生成的位置参数。"""

    parameter_names = _OPERATION_ARGUMENTS[operation]
    if call.keywords:
        raise ValueError(
            "Workflow edit operations must use positional arguments"
        )
    if len(call.args) != len(parameter_names):
        raise ValueError(
            f"{operation} expects {len(parameter_names)} arguments, "
            f"got {len(call.args)}"
        )
    return dict(zip(parameter_names, call.args))


def _extract_call(
    source: str,
) -> tuple[ast.Call, str | None]:
    """读取“单次调用”或“变量 = 单次调用”形式的编辑字符串。"""

    module = ast.parse(source)
    if len(module.body) != 1:
        raise ValueError(
            "Each workflow edit set item must contain exactly one operation"
        )

    statement = module.body[0]
    if isinstance(statement, ast.Expr) and isinstance(
        statement.value,
        ast.Call,
    ):
        return statement.value, None
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Call)
    ):
        return statement.value, statement.targets[0].id
    raise ValueError(
        "Workflow edit must be a function call or a variable assignment "
        "whose value is a function call"
    )


def _parse_edit(source: str) -> _Edit:
    """将一条编辑字符串解析为与平台 ID 无关的结构。"""

    call, assignment_target = _extract_call(source)
    if not isinstance(call.func, ast.Name):
        raise ValueError("Workflow edit operation must be a direct call")

    operation = call.func.id
    if operation not in _OPERATION_ARGUMENTS:
        raise ValueError(
            f"Unsupported workflow edit operation: {operation}"
        )
    arguments = _bind_arguments(call, operation)

    if assignment_target is not None and operation not in {
        "add_node",
        "add_edge",
    }:
        raise ValueError(
            "Only add_node and add_edge may assign a returned identifier"
        )

    if operation == "add_node":
        # node_name 只是展示名称，不参与工作流编辑语义比较；
        # add_node 仅要求实际组件 widget_name 一致。
        parsed_arguments = (
            _literal_value(arguments["widget_name"]),
        )
        result_identifier = (
            _Identifier("node", f"name:{assignment_target}")
            if assignment_target is not None
            else None
        )
    elif operation == "delete_node":
        parsed_arguments = (
            _identifier(arguments["node_id"], "node"),
        )
        result_identifier = None
    elif operation == "update_node_params":
        parsed_arguments = (
            _identifier(arguments["node_id"], "node"),
            _literal_value(arguments["widget_name"]),
            _literal_value(arguments["node_params"]),
        )
        result_identifier = None
    elif operation == "add_edge":
        parsed_arguments = (
            _identifier(arguments["source_node_id"], "node"),
            _identifier(arguments["target_node_id"], "node"),
        )
        result_identifier = (
            _Identifier("edge", f"name:{assignment_target}")
            if assignment_target is not None
            else None
        )
    else:
        parsed_arguments = (
            _identifier(arguments["edge_id"], "edge"),
        )
        result_identifier = None

    return _Edit(
        operation=operation,
        arguments=parsed_arguments,
        result_identifier=result_identifier,
    )


def _parse_edit_set(
    edit_set: Any,
    *,
    label: str,
) -> list[_Edit]:
    """解析字符串列表，或 CSV 中保存的 JSON 字符串列表。"""

    if isinstance(edit_set, str):
        try:
            decoded = json.loads(edit_set)
        except json.JSONDecodeError:
            decoded = None
        edit_set = decoded if isinstance(decoded, list) else [edit_set]

    if not isinstance(edit_set, (list, tuple)):
        raise TypeError(
            f"{label} must be a list of operation strings or a JSON list"
        )

    parsed: list[_Edit] = []
    for index, source in enumerate(edit_set):
        if not isinstance(source, str):
            raise TypeError(
                f"{label}[{index}] must be a string, got "
                f"{type(source).__name__}"
            )
        try:
            parsed.append(_parse_edit(source))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Unable to parse {label}[{index}]: {source!r}"
            ) from exc
    return parsed


def _copy_identifier_maps(
    identifier_maps: dict[
        str,
        tuple[dict[str, str], dict[str, str]],
    ],
) -> dict[str, tuple[dict[str, str], dict[str, str]]]:
    """复制双射映射，供一次候选匹配安全试探。"""

    return {
        namespace: (dict(forward), dict(reverse))
        for namespace, (forward, reverse) in identifier_maps.items()
    }


def _unify(
    predicted: Any,
    ground_truth: Any,
    identifier_maps: dict[
        str,
        tuple[dict[str, str], dict[str, str]],
    ],
) -> bool:
    """在节点和边 ID 的全局双射约束下比较两个值。"""

    if isinstance(predicted, _Identifier) or isinstance(
        ground_truth,
        _Identifier,
    ):
        if not isinstance(predicted, _Identifier) or not isinstance(
            ground_truth,
            _Identifier,
        ):
            return False
        if predicted.namespace != ground_truth.namespace:
            return False

        forward, reverse = identifier_maps[predicted.namespace]
        if predicted.value in forward:
            return forward[predicted.value] == ground_truth.value
        if ground_truth.value in reverse:
            return reverse[ground_truth.value] == predicted.value

        forward[predicted.value] = ground_truth.value
        reverse[ground_truth.value] = predicted.value
        return True

    if isinstance(predicted, tuple) or isinstance(ground_truth, tuple):
        if not isinstance(predicted, tuple) or not isinstance(
            ground_truth,
            tuple,
        ):
            return False
        if len(predicted) != len(ground_truth):
            return False
        return all(
            _unify(
                predicted_item,
                ground_truth_item,
                identifier_maps,
            )
            for predicted_item, ground_truth_item in zip(
                predicted,
                ground_truth,
            )
        )

    return predicted == ground_truth


def _match_edits(
    predicted: _Edit,
    ground_truth: _Edit,
    identifier_maps: dict[
        str,
        tuple[dict[str, str], dict[str, str]],
    ],
) -> dict[str, tuple[dict[str, str], dict[str, str]]] | None:
    """匹配两次操作，并返回加入本次 ID 对应关系后的映射。"""

    if predicted.operation != ground_truth.operation:
        return None

    updated_maps = _copy_identifier_maps(identifier_maps)
    if not _unify(
        predicted.arguments,
        ground_truth.arguments,
        updated_maps,
    ):
        return None
    if (
        predicted.result_identifier is not None
        and ground_truth.result_identifier is not None
        and not _unify(
            predicted.result_identifier,
            ground_truth.result_identifier,
            updated_maps,
        )
    ):
        return None
    return updated_maps


def _maximum_intersection(
    predicted_edits: list[_Edit],
    ground_truth_edits: list[_Edit],
) -> int:
    """求满足同一套 ID 映射时可匹配的最大操作数。"""

    if not predicted_edits or not ground_truth_edits:
        return 0

    empty_maps: dict[
        str,
        tuple[dict[str, str], dict[str, str]],
    ] = {
        "node": ({}, {}),
        "edge": ({}, {}),
    }

    # 优先处理候选较少的操作，可明显减少回溯分支。
    candidate_counts = [
        (
            sum(
                _match_edits(
                    predicted,
                    ground_truth,
                    empty_maps,
                )
                is not None
                for ground_truth in ground_truth_edits
            ),
            index,
        )
        for index, predicted in enumerate(predicted_edits)
    ]
    ordered_predictions = [
        predicted_edits[index]
        for _, index in sorted(candidate_counts)
    ]

    maximum = 0

    def backtrack(
        predicted_index: int,
        used_ground_truth: set[int],
        identifier_maps: dict[
            str,
            tuple[dict[str, str], dict[str, str]],
        ],
        matches: int,
    ) -> None:
        nonlocal maximum
        maximum = max(maximum, matches)
        if predicted_index >= len(ordered_predictions):
            return

        remaining_predictions = (
            len(ordered_predictions) - predicted_index
        )
        remaining_ground_truth = (
            len(ground_truth_edits) - len(used_ground_truth)
        )
        if (
            matches
            + min(remaining_predictions, remaining_ground_truth)
            <= maximum
        ):
            return

        predicted = ordered_predictions[predicted_index]
        for ground_truth_index, ground_truth in enumerate(
            ground_truth_edits
        ):
            if ground_truth_index in used_ground_truth:
                continue
            updated_maps = _match_edits(
                predicted,
                ground_truth,
                identifier_maps,
            )
            if updated_maps is None:
                continue

            used_ground_truth.add(ground_truth_index)
            backtrack(
                predicted_index + 1,
                used_ground_truth,
                updated_maps,
                matches + 1,
            )
            used_ground_truth.remove(ground_truth_index)

        # 当前预测操作也可以不参与匹配。
        backtrack(
            predicted_index + 1,
            used_ground_truth,
            identifier_maps,
            matches,
        )

    backtrack(
        predicted_index=0,
        used_ground_truth=set(),
        identifier_maps=empty_maps,
        matches=0,
    )
    return maximum


def workflow_edit_metrics(
    predicted_edit_set: Any,
    ground_truth_edit_set: Any,
) -> tuple[float, float, float]:
    """返回 Workflow Edit Precision、Recall 和 F1。

    输入可以是编辑操作字符串列表，也可以是 CSV 中保存的 JSON 字符串
    列表。节点和边的具体 ID 会通过全局双射映射脱敏，操作顺序不影响
    匹配结果。
    """

    predicted = _parse_edit_set(
        predicted_edit_set,
        label="predicted_edit_set",
    )
    ground_truth = _parse_edit_set(
        ground_truth_edit_set,
        label="ground_truth_edit_set",
    )

    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0
    if not predicted or not ground_truth:
        return 0.0, 0.0, 0.0

    correct = _maximum_intersection(predicted, ground_truth)
    precision = correct / len(predicted)
    recall = correct / len(ground_truth)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return (
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
    )


__all__ = ["workflow_edit_metrics"]
