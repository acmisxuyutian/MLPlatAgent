"""Adapt the portability reference graphs to Orange3 workflow semantics.

The original annotations were written for Uniplore.  Orange3 requires a
``Select Columns`` widget immediately after every data-loading widget, and
the Orange3 adapter does not expose native XGBoost or LightGBM widgets.

This script updates the four Orange3 benchmark files in place.  It is
idempotent: running it again does not add another Select Columns requirement
or otherwise change an already adapted graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = ROOT / "orange3" / "benchmark"
BENCHMARK_PATHS = (
    BENCHMARK_ROOT / "ml_benchmark.json",
    BENCHMARK_ROOT / "dseval_kaggle.json",
    BENCHMARK_ROOT / "DSEval-Kaggle-Ext" / "UCI.json",
    BENCHMARK_ROOT / "DSEval-Kaggle-Ext" / "CI.json",
)

DATA_LOADING_STEP = "Tabular Dataset Loading"
SELECT_COLUMNS_STEP = "Select Columns"
UNSUPPORTED_STEPS = frozenset({"XGBoost", "LightGBM"})


def _deduplicate_edges(
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the first edge for every directed endpoint pair."""

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        source = str(edge["source_node_id"])
        target = str(edge["target_node_id"])
        if source == target or (source, target) in seen:
            continue
        seen.add((source, target))
        edge["source_node_id"] = source
        edge["target_node_id"] = target
        result.append(edge)
    return result


def _remove_unsupported_steps(
    steps: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    next_edge_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    """Remove unsupported nodes while preserving dependencies by bypassing them."""

    unsupported_ids = {
        str(step["node_id"])
        for step in steps
        if step.get("algorithm_step_name") in UNSUPPORTED_STEPS
    }
    removed_count = len(unsupported_ids)

    # Eliminate one unsupported node at a time.  Connecting every predecessor
    # to every successor retains the reachability relation among requirements
    # that remain in the platform-specific reference graph.
    for node_id in unsupported_ids:
        predecessors = {
            str(edge["source_node_id"])
            for edge in edges
            if str(edge["target_node_id"]) == node_id
            and str(edge["source_node_id"]) != node_id
        }
        successors = {
            str(edge["target_node_id"])
            for edge in edges
            if str(edge["source_node_id"]) == node_id
            and str(edge["target_node_id"]) != node_id
        }
        edges = [
            edge
            for edge in edges
            if str(edge["source_node_id"]) != node_id
            and str(edge["target_node_id"]) != node_id
        ]

        existing_pairs = {
            (
                str(edge["source_node_id"]),
                str(edge["target_node_id"]),
            )
            for edge in edges
        }
        for source in sorted(predecessors):
            for target in sorted(successors):
                if (
                    source == target
                    or (source, target) in existing_pairs
                ):
                    continue
                edges.append(
                    {
                        "edge_id": str(next_edge_id),
                        "source_node_id": source,
                        "target_node_id": target,
                    }
                )
                next_edge_id += 1
                existing_pairs.add((source, target))

    steps = [
        step
        for step in steps
        if str(step["node_id"]) not in unsupported_ids
    ]
    return steps, _deduplicate_edges(edges), next_edge_id, removed_count


def _repair_dangling_dependencies(
    steps: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> int:
    """Repair the one legacy loader-to-prediction annotation typo.

    The original CI benchmark contains a test-loader edge whose target ID does
    not exist.  When a task has exactly one prediction requirement, that
    prediction is the unambiguous intended target.  Other dangling endpoints
    remain errors and are rejected by validation.
    """

    step_by_id = {
        str(step["node_id"]): step
        for step in steps
    }
    prediction_ids = [
        str(step["node_id"])
        for step in steps
        if step.get("algorithm_step_name") == "Test Dataset Predictions"
    ]
    repaired = 0
    for edge in edges:
        source_id = str(edge["source_node_id"])
        target_id = str(edge["target_node_id"])
        if source_id not in step_by_id:
            continue
        if target_id in step_by_id:
            continue
        if (
            step_by_id[source_id].get("algorithm_step_name")
            == DATA_LOADING_STEP
            and len(prediction_ids) == 1
        ):
            edge["target_node_id"] = prediction_ids[0]
            repaired += 1
    return repaired


def _insert_select_columns(
    steps: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    next_node_id: int,
    next_edge_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int, int]:
    """Make Select Columns the immediate successor of every data loader."""

    step_by_id = {
        str(step["node_id"]): step
        for step in steps
    }
    new_steps: list[dict[str, Any]] = []
    inserted_count = 0
    redirected_count = 0

    for step in steps:
        new_steps.append(step)
        if step.get("algorithm_step_name") != DATA_LOADING_STEP:
            continue

        loader_id = str(step["node_id"])
        direct_select_ids = [
            str(edge["target_node_id"])
            for edge in edges
            if str(edge["source_node_id"]) == loader_id
            and step_by_id.get(
                str(edge["target_node_id"]), {}
            ).get("algorithm_step_name") == SELECT_COLUMNS_STEP
        ]

        if direct_select_ids:
            select_id = direct_select_ids[0]
        else:
            select_id = str(next_node_id)
            next_node_id += 1
            select_step = {
                "node_id": select_id,
                "algorithm_step_name": SELECT_COLUMNS_STEP,
                "node_name": f"Select Columns after loader {loader_id}",
                "node_params": {},
                "contrast_level": 1,
                "node_type": "preprocess",
            }
            new_steps.append(select_step)
            step_by_id[select_id] = select_step
            inserted_count += 1

        # All non-Select successors must receive data from Select Columns, not
        # directly from the loader.  This makes the required ordering explicit.
        for edge in edges:
            if (
                str(edge["source_node_id"]) == loader_id
                and str(edge["target_node_id"]) != select_id
                and step_by_id.get(
                    str(edge["target_node_id"]), {}
                ).get("algorithm_step_name") != SELECT_COLUMNS_STEP
            ):
                edge["source_node_id"] = select_id
                redirected_count += 1

        if not any(
            str(edge["source_node_id"]) == loader_id
            and str(edge["target_node_id"]) == select_id
            for edge in edges
        ):
            edges.append(
                {
                    "edge_id": str(next_edge_id),
                    "source_node_id": loader_id,
                    "target_node_id": select_id,
                }
            )
            next_edge_id += 1

    return (
        new_steps,
        _deduplicate_edges(edges),
        next_node_id,
        next_edge_id,
        inserted_count,
    )


def _split_shared_loader_selects(
    steps: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    next_node_id: int,
    next_edge_id: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    int,
    int,
]:
    """Give each loader its own Select Columns node.

    This also repairs data produced by the first version of this script, where
    a legacy dangling endpoint happened to equal a newly allocated node ID.
    """

    step_by_id = {
        str(step["node_id"]): step
        for step in steps
    }
    loader_ids = {
        node_id
        for node_id, step in step_by_id.items()
        if step.get("algorithm_step_name") == DATA_LOADING_STEP
    }
    cloned_count = 0

    for select_step in list(steps):
        if select_step.get("algorithm_step_name") != SELECT_COLUMNS_STEP:
            continue
        select_id = str(select_step["node_id"])
        incoming_loader_edges = [
            edge
            for edge in edges
            if str(edge["target_node_id"]) == select_id
            and str(edge["source_node_id"]) in loader_ids
        ]
        if len(incoming_loader_edges) <= 1:
            continue

        named_owner = str(select_step.get("node_name", "")).removeprefix(
            "Select Columns after loader "
        )
        owner_id = (
            named_owner
            if named_owner
            in {
                str(edge["source_node_id"])
                for edge in incoming_loader_edges
            }
            else str(incoming_loader_edges[0]["source_node_id"])
        )
        outgoing_targets = [
            str(edge["target_node_id"])
            for edge in edges
            if str(edge["source_node_id"]) == select_id
        ]

        for incoming_edge in incoming_loader_edges:
            loader_id = str(incoming_edge["source_node_id"])
            if loader_id == owner_id:
                continue
            clone_id = str(next_node_id)
            next_node_id += 1
            clone = dict(select_step)
            clone["node_id"] = clone_id
            clone["node_name"] = f"Select Columns after loader {loader_id}"
            steps.append(clone)
            step_by_id[clone_id] = clone
            incoming_edge["target_node_id"] = clone_id

            existing_pairs = {
                (
                    str(edge["source_node_id"]),
                    str(edge["target_node_id"]),
                )
                for edge in edges
            }
            for target_id in outgoing_targets:
                if (clone_id, target_id) in existing_pairs:
                    continue
                edges.append(
                    {
                        "edge_id": str(next_edge_id),
                        "source_node_id": clone_id,
                        "target_node_id": target_id,
                    }
                )
                next_edge_id += 1
            cloned_count += 1

    return (
        steps,
        _deduplicate_edges(edges),
        next_node_id,
        next_edge_id,
        cloned_count,
    )


def _validate_requirements(requirements: dict[str, Any]) -> None:
    """Validate IDs, endpoints, unsupported steps, and loader sequencing."""

    steps = requirements["step_requirements"]
    edges = requirements["dependency_requirements"]
    node_ids = [str(step["node_id"]) for step in steps]
    edge_ids = [str(edge["edge_id"]) for edge in edges]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Duplicate node IDs in adapted requirements")
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Duplicate edge IDs in adapted requirements")

    node_id_set = set(node_ids)
    step_by_id = {
        str(step["node_id"]): step
        for step in steps
    }
    for edge in edges:
        if (
            str(edge["source_node_id"]) not in node_id_set
            or str(edge["target_node_id"]) not in node_id_set
        ):
            raise ValueError(f"Dangling dependency: {edge}")

    remaining_unsupported = [
        step["algorithm_step_name"]
        for step in steps
        if step.get("algorithm_step_name") in UNSUPPORTED_STEPS
    ]
    if remaining_unsupported:
        raise ValueError(
            f"Unsupported Orange3 steps remain: {remaining_unsupported}"
        )

    for step in steps:
        if step.get("algorithm_step_name") != DATA_LOADING_STEP:
            continue
        loader_id = str(step["node_id"])
        successors = [
            step_by_id[str(edge["target_node_id"])]
            for edge in edges
            if str(edge["source_node_id"]) == loader_id
        ]
        if len(successors) != 1 or any(
            successor.get("algorithm_step_name") != SELECT_COLUMNS_STEP
            for successor in successors
        ):
            raise ValueError(
                f"Loader {loader_id} must have exactly one immediate "
                "Select Columns successor"
            )

    for step in steps:
        if step.get("algorithm_step_name") != SELECT_COLUMNS_STEP:
            continue
        select_id = str(step["node_id"])
        loader_predecessors = [
            edge
            for edge in edges
            if str(edge["target_node_id"]) == select_id
            and step_by_id[str(edge["source_node_id"])].get(
                "algorithm_step_name"
            )
            == DATA_LOADING_STEP
        ]
        if len(loader_predecessors) > 1:
            raise ValueError(
                f"Select Columns {select_id} is shared by multiple loaders"
            )


def _adapt_task(
    task: dict[str, Any],
    next_node_id: int,
    next_edge_id: int,
) -> tuple[int, int, int, int, int, int]:
    requirements = task["requirements"]
    steps = requirements["step_requirements"]
    edges = requirements["dependency_requirements"]
    repaired_count = _repair_dangling_dependencies(steps, edges)

    steps, edges, next_edge_id, removed_count = (
        _remove_unsupported_steps(
            steps,
            edges,
            next_edge_id,
        )
    )
    (
        steps,
        edges,
        next_node_id,
        next_edge_id,
        inserted_count,
    ) = _insert_select_columns(
        steps,
        edges,
        next_node_id,
        next_edge_id,
    )
    (
        steps,
        edges,
        next_node_id,
        next_edge_id,
        cloned_count,
    ) = _split_shared_loader_selects(
        steps,
        edges,
        next_node_id,
        next_edge_id,
    )
    inserted_count += cloned_count
    requirements["step_requirements"] = steps
    requirements["dependency_requirements"] = edges
    _validate_requirements(requirements)
    loader_count = sum(
        step.get("algorithm_step_name") == DATA_LOADING_STEP
        for step in steps
    )
    return (
        removed_count,
        inserted_count,
        loader_count,
        repaired_count,
        next_node_id,
        next_edge_id,
    )


def _make_ids_globally_unique(
    loaded_benchmarks: list[tuple[Path, list[dict[str, Any]]]],
    next_node_id: int,
    next_edge_id: int,
) -> tuple[int, int, int, int]:
    """Remove cross-task ID collisions introduced by platform adaptation."""

    node_occurrences: dict[
        str,
        list[tuple[dict[str, Any], dict[str, Any]]],
    ] = {}
    for _, tasks in loaded_benchmarks:
        for task in tasks:
            requirements = task["requirements"]
            for step in requirements["step_requirements"]:
                node_occurrences.setdefault(
                    str(step["node_id"]),
                    [],
                ).append((step, requirements))

    renamed_nodes = 0
    for occurrences in node_occurrences.values():
        if len(occurrences) <= 1:
            continue
        # Preserve an original annotation ID when it collides with a generated
        # Select Columns node.
        occurrences.sort(
            key=lambda item: str(item[0].get("node_name", "")).startswith(
                "Select Columns after loader "
            )
        )
        for step, requirements in occurrences[1:]:
            old_id = str(step["node_id"])
            new_id = str(next_node_id)
            next_node_id += 1
            step["node_id"] = new_id
            for edge in requirements["dependency_requirements"]:
                if str(edge["source_node_id"]) == old_id:
                    edge["source_node_id"] = new_id
                if str(edge["target_node_id"]) == old_id:
                    edge["target_node_id"] = new_id
            renamed_nodes += 1

    seen_edge_ids: set[str] = set()
    renamed_edges = 0
    for _, tasks in loaded_benchmarks:
        for task in tasks:
            for edge in task["requirements"]["dependency_requirements"]:
                edge_id = str(edge["edge_id"])
                if edge_id in seen_edge_ids:
                    edge_id = str(next_edge_id)
                    next_edge_id += 1
                    edge["edge_id"] = edge_id
                    renamed_edges += 1
                seen_edge_ids.add(edge_id)

    return next_node_id, next_edge_id, renamed_nodes, renamed_edges


def main() -> None:
    total_tasks = 0
    total_removed = 0
    total_inserted = 0
    total_loaders = 0
    total_repaired = 0
    loaded_benchmarks: list[tuple[Path, list[dict[str, Any]]]] = []

    for path in BENCHMARK_PATHS:
        tasks = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(tasks, list):
            raise TypeError(f"{path} must contain a JSON list")
        loaded_benchmarks.append((path, tasks))

    numeric_node_ids = [
        int(identifier)
        for _, tasks in loaded_benchmarks
        for task in tasks
        for identifier in (
            [
                str(step["node_id"])
                for step in task["requirements"]["step_requirements"]
            ]
            + [
                str(edge["source_node_id"])
                for edge in task["requirements"]["dependency_requirements"]
            ]
            + [
                str(edge["target_node_id"])
                for edge in task["requirements"]["dependency_requirements"]
            ]
        )
        if identifier.isdigit()
    ]
    numeric_edge_ids = [
        int(str(edge["edge_id"]))
        for _, tasks in loaded_benchmarks
        for task in tasks
        for edge in task["requirements"]["dependency_requirements"]
        if str(edge["edge_id"]).isdigit()
    ]
    next_node_id = max(numeric_node_ids, default=0) + 1
    next_edge_id = max(numeric_edge_ids, default=0) + 1

    for path, tasks in loaded_benchmarks:
        file_removed = 0
        file_inserted = 0
        file_loaders = 0
        file_repaired = 0
        for task in tasks:
            (
                removed,
                inserted,
                loaders,
                repaired,
                next_node_id,
                next_edge_id,
            ) = _adapt_task(
                task,
                next_node_id,
                next_edge_id,
            )
            file_removed += removed
            file_inserted += inserted
            file_loaders += loaders
            file_repaired += repaired

        total_tasks += len(tasks)
        total_removed += file_removed
        total_inserted += file_inserted
        total_loaders += file_loaders
        total_repaired += file_repaired
        print(
            f"{path.relative_to(ROOT)}: tasks={len(tasks)}, "
            f"removed={file_removed}, inserted_select={file_inserted}, "
            f"loaders={file_loaders}, repaired_edges={file_repaired}"
        )

    (
        _,
        _,
        renamed_nodes,
        renamed_edges,
    ) = _make_ids_globally_unique(
        loaded_benchmarks,
        next_node_id,
        next_edge_id,
    )
    for path, tasks in loaded_benchmarks:
        for task in tasks:
            _validate_requirements(task["requirements"])
        path.write_text(
            json.dumps(tasks, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        f"Total: tasks={total_tasks}, removed={total_removed}, "
        f"inserted_select={total_inserted}, loaders={total_loaders}, "
        f"repaired_edges={total_repaired}, "
        f"renamed_nodes={renamed_nodes}, renamed_edges={renamed_edges}"
    )


if __name__ == "__main__":
    main()
