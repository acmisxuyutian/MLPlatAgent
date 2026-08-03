"""The ``w/o tool retrieval`` baseline from MLPlatAgent Table 13."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.runtime import (
    AllWidgetsExecutor,
    PlannerBackedAgent,
    run_standalone,
)


METHOD_NAME = "w/o tool retrieval"


class Agent(PlannerBackedAgent):
    """Retain the MLPlatAgent Planner and expose every platform widget."""

    executor_class = AllWidgetsExecutor


def main() -> None:
    run_standalone(Agent)


if __name__ == "__main__":
    main()
