"""The ``w/o planning`` baseline from MLPlatAgent Table 13."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.runtime import DirectWorkflowAgent, run_standalone


METHOD_NAME = "w/o planning"


class Agent(DirectWorkflowAgent):
    """Remove planning and retain semantic-similarity tool retrieval."""

    retrieval_mode = "semantic"
    catch_llm_errors = False


def main() -> None:
    run_standalone(Agent)


if __name__ == "__main__":
    main()
