"""Summarize the Modification Instructions (MI) experiment results.

For every model and method below this directory, the script calculates the
arithmetic mean and sample variance of RCR, WGF1, WEP, WER, and WEF1. Results
are formatted as ``mean±variance`` (three and two decimal places,
respectively), printed to the terminal, and saved as ``analysis_result.csv``.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "analysis_result.csv"

METRICS = ("wgf1", "wep", "wer", "wef1")
REQUIRED_COLUMNS = {"instruction", *METRICS}
METHOD_LABELS = OrderedDict(
    (
        ("MLPlatAgent", "MLPlatAgent"),
        ("wo_planning", "w/o planning"),
        ("wo_tool_retrieval", "w/o tool retrieval"),
        ("workflow_construction", "workflow construction"),
        ("ReAct", "ReAct"),
        ("FunctionCall", "Function Call"),
        ("DFSDT", "DFSDT"),
    )
)


def _result_paths() -> list[Path]:
    """Find result files in a stable model/method order."""

    paths = list(ROOT.glob("*/*/mi.csv"))
    if not paths:
        raise FileNotFoundError(f"No MI result files found below {ROOT}")

    method_order = {
        directory_name: index
        for index, directory_name in enumerate(METHOD_LABELS)
    }
    return sorted(
        paths,
        key=lambda path: (
            path.parent.parent.name,
            method_order.get(path.parent.name, len(method_order)),
            path.parent.name,
        ),
    )


def _load_result(path: Path) -> list[dict[str, str | float]]:
    """Load one method result and validate the aggregation inputs."""

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = sorted(
            REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        )
        if missing_columns:
            raise ValueError(
                f"{path} is missing columns: {', '.join(missing_columns)}"
            )

        data: list[dict[str, str | float]] = []
        for line_number, row in enumerate(reader, start=2):
            instruction = row["instruction"]
            if instruction is None or not instruction.strip():
                raise ValueError(
                    f"{path}:{line_number} has an empty instruction"
                )

            parsed_row: dict[str, str | float] = {"instruction": instruction}
            for metric in METRICS:
                try:
                    value = float(row[metric])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}:{line_number} has invalid {metric} value "
                        f"{row[metric]!r}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"{path}:{line_number} has non-finite {metric}"
                    )
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"{path}:{line_number} has {metric} outside [0, 1]"
                    )
                parsed_row[metric] = value
            data.append(parsed_row)

    if not data:
        raise ValueError(f"MI result is empty: {path}")

    repetitions = Counter(str(row["instruction"]) for row in data)
    repetition_counts = set(repetitions.values())
    if len(repetition_counts) != 1:
        raise ValueError(
            f"{path} has non-uniform task repetitions: "
            f"{sorted(repetition_counts)}"
        )
    if len(data) < 2:
        raise ValueError(f"Sample variance requires at least two rows: {path}")

    return data


def build_summary() -> list[dict[str, object]]:
    """Build one row for every model/method result file."""

    rows: list[dict[str, object]] = []
    reference_instructions: set[str] | None = None

    for path in _result_paths():
        data = _load_result(path)
        instructions = {str(row["instruction"]) for row in data}
        if reference_instructions is None:
            reference_instructions = instructions
        elif instructions != reference_instructions:
            raise ValueError(
                f"{path} does not contain the same task set as other results"
            )

        method_directory = path.parent.name
        row: dict[str, object] = {
            "model": path.parent.parent.name,
            "method": METHOD_LABELS.get(
                method_directory,
                method_directory,
            ),
        }
        for metric in METRICS:
            values = [float(result[metric]) for result in data]
            mean = statistics.fmean(values)
            variance = statistics.variance(values)
            row[metric] = f"{mean:.3f}±{variance:.2f}"
        rows.append(row)

    return rows


def _save_summary(summary: list[dict[str, object]]) -> None:
    fieldnames = ["model", "method", *METRICS]

    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def _print_summary(summary: list[dict[str, object]]) -> None:
    headers = ["Model", "Method", *(metric.upper() for metric in METRICS)]
    keys = ["model", "method", *METRICS]

    rows = [
        [str(row[key]) for key in keys]
        for row in summary
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    template = "  ".join(f"{{:<{width}}}" for width in widths)

    print(template.format(*headers))
    print(template.format(*("-" * width for width in widths)))
    for row in rows:
        print(template.format(*row))


def main() -> None:
    summary = build_summary()
    _save_summary(summary)
    _print_summary(summary)
    print("\nVariance: sample variance (ddof=1)")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
