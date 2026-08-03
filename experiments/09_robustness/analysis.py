"""Summarize refusal behavior in the robustness experiment.

The script discovers every ``*/robustness.csv`` result below this directory
and reports refusal counts and rates for each model/task-category pair.
The same summary is also saved to ``analysis_result.csv``.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "analysis_result.csv"
REQUIRED_COLUMNS = {"model", "category", "is_refuse"}


def _load_counts() -> OrderedDict[tuple[str, str], list[int]]:
    """Return ``(refusal count, total count)`` for each model and category."""

    result_paths = sorted(ROOT.glob("*/robustness.csv"))
    if not result_paths:
        raise FileNotFoundError(
            f"No robustness result files found below {ROOT}"
        )

    counts: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    for path in result_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path} is missing columns: {', '.join(sorted(missing))}"
                )

            row_count = 0
            for line_number, row in enumerate(reader, start=2):
                row_count += 1
                model = row["model"].strip()
                category = row["category"].strip()
                is_refuse_text = row["is_refuse"].strip()

                if not model or not category:
                    raise ValueError(
                        f"{path}:{line_number} has an empty model or category"
                    )
                if is_refuse_text not in {"0", "1"}:
                    raise ValueError(
                        f"{path}:{line_number} has invalid is_refuse value "
                        f"{is_refuse_text!r}; expected 0 or 1"
                    )

                key = (model, category)
                if key not in counts:
                    counts[key] = [0, 0]
                counts[key][0] += int(is_refuse_text)
                counts[key][1] += 1

            if row_count == 0:
                raise ValueError(f"Robustness result is empty: {path}")

    return counts


def build_summary() -> list[dict[str, object]]:
    """Build one summary row per model/task-category pair."""

    summary: list[dict[str, object]] = []
    for (model, category), (refuse_count, total_count) in _load_counts().items():
        summary.append(
            {
                "model": model,
                "category": category,
                "refuse_count": refuse_count,
                "total_count": total_count,
                "refuse_rate": refuse_count / total_count,
            }
        )
    return summary


def _save_summary(summary: list[dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "category",
        "refuse_count",
        "total_count",
        "refuse_rate",
    ]
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            output_row = dict(row)
            output_row["refuse_rate"] = f"{row['refuse_rate']:.4f}"
            writer.writerow(output_row)


def _print_summary(summary: list[dict[str, object]]) -> None:
    headers = ("Model", "Task category", "Refused", "Total", "Refusal rate")
    rows = [
        (
            str(row["model"]),
            str(row["category"]),
            str(row["refuse_count"]),
            str(row["total_count"]),
            f"{row['refuse_rate']:.2%}",
        )
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
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
