"""The DFSDT workflow-construction baseline from MLPlatAgent Table 11."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.runtime import (
    DFSDTExecutor,
    PlannerBackedAgent,
    run_standalone,
)


METHOD_NAME = "DFSDT"


class Agent(PlannerBackedAgent):
    """Use MLPlatAgent planning/retrieval and DFSDT tree search."""

    executor_class = DFSDTExecutor


def main() -> None:
    run_standalone(Agent)


if __name__ == "__main__":
    main()
