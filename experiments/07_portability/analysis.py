"""Build the compact summary used by the portability table.

Only the two task groups reported in the paper are retained. The
DSEval-Kaggle-Ext group contains UCI, CI, and MI. RCR and WGF1 are shown as
``mean±sample variance`` over all runs in a group.

Efficiency follows the conventions used by
``experiments/06_tool_retrieve_type/analysis.py``:

* time is the mean wall-clock time per run, in seconds; and
* cost uses the same token prices (CNY 0.004/1K input tokens and CNY
  0.012/1K output tokens, converted at CNY 7.2/USD).  The cost of a task
  group is the arithmetic mean of the total costs of its constituent
  benchmark files, matching the referenced analysis script.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent

MODEL_LABELS = OrderedDict(
    (
        ("qwen2_5-72b", "Qwen2.5-72B-Instruct"),
        ("qwen2_5-14b-coder", "Qwen2.5-14B-Coder"),
    )
)
PLATFORM_LABELS = OrderedDict(
    (
        ("uniplore", "Uniplore"),
        ("orange3", "Orange3"),
    )
)
TASK_GROUPS = OrderedDict(
    (
        ("MLBenchmark_DSEval-Kaggle", ("mlb", "dseval")),
        ("DSEval-Kaggle-Ext", ("uci", "ci", "mi")),
    )
)

REQUIRED_COLUMNS = {
    "instruction",
    "rcr",
    "wgf1",
    "times",
    "input_tokens",
    "output_tokens",
}
NUMERIC_COLUMNS = REQUIRED_COLUMNS.difference({"instruction"})

INPUT_CNY_PER_1K_TOKENS = 0.004
OUTPUT_CNY_PER_1K_TOKENS = 0.012
CNY_PER_USD = 7.2


def _load_result(platform: str, llm: str, benchmark: str) -> pd.DataFrame:
    """Load one raw result file and validate the columns used in aggregation."""

    path = ROOT / platform / llm / f"{benchmark}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing portability result: {path}")

    data = pd.read_csv(path)
    if data.empty:
        raise ValueError(f"Empty portability result: {path}")
    missing = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing:
        raise ValueError(
            f"{path} is missing columns: {', '.join(missing)}"
        )

    data = data.copy()
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if data[column].isna().any():
            raise ValueError(f"{path} contains missing {column} values")

    for metric in ("rcr", "wgf1"):
        invalid = ~data[metric].between(0.0, 1.0, inclusive="both")
        if invalid.any():
            raise ValueError(
                f"{path} contains {metric.upper()} outside [0, 1]"
            )

    repetitions = data.groupby("instruction", sort=False).size()
    if len(repetitions.unique()) != 1:
        raise ValueError(
            f"{path} has non-uniform task repetitions: "
            f"{sorted(repetitions.unique())}"
        )
    return data


def _mean_variance_display(values: pd.Series) -> str:
    """Format mean and sample variance using the paper's convention."""

    if len(values) < 2:
        raise ValueError("Sample variance requires at least two runs")
    return f"{values.mean():.3f}±{values.var(ddof=1):.2f}"


def _total_cost_usd(data: pd.DataFrame) -> float:
    """Compute one benchmark file's total token cost in USD."""

    input_cost_cny = (
        data["input_tokens"].sum()
        * INPUT_CNY_PER_1K_TOKENS
        / 1000
    )
    output_cost_cny = (
        data["output_tokens"].sum()
        * OUTPUT_CNY_PER_1K_TOKENS
        / 1000
    )
    return float((input_cost_cny + output_cost_cny) / CNY_PER_USD)


def _summarize_group(
    frames: dict[str, pd.DataFrame],
    benchmarks: tuple[str, ...],
) -> dict[str, object]:
    """Return the four values reported for one paper-level task group."""

    grouped = pd.concat(
        [frames[benchmark] for benchmark in benchmarks],
        ignore_index=True,
    )
    benchmark_costs = [
        _total_cost_usd(frames[benchmark])
        for benchmark in benchmarks
    ]
    return {
        "RCR": _mean_variance_display(grouped["rcr"]),
        "WGF1": _mean_variance_display(grouped["wgf1"]),
        "Time_s": round(float(grouped["times"].mean()), 3),
        "Cost_USD": round(
            float(sum(benchmark_costs) / len(benchmark_costs)),
            3,
        ),
    }


def build_summary() -> pd.DataFrame:
    """Build four rows in the same order as the paper table."""

    rows: list[dict[str, object]] = []
    benchmarks = {
        benchmark
        for group in TASK_GROUPS.values()
        for benchmark in group
    }

    for llm, model_label in MODEL_LABELS.items():
        for platform, platform_label in PLATFORM_LABELS.items():
            frames = {
                benchmark: _load_result(platform, llm, benchmark)
                for benchmark in benchmarks
            }
            row: dict[str, object] = {
                "Base_LLM": model_label,
                "Platform": platform_label,
            }
            for group_name, group_benchmarks in TASK_GROUPS.items():
                statistics = _summarize_group(
                    frames,
                    group_benchmarks,
                )
                for metric_name, value in statistics.items():
                    row[f"{group_name}_{metric_name}"] = value
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    summary = build_summary()
    output_path = ROOT / "analysis_result.csv"
    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(summary.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
