"""The workflow-construction-only baseline from MLPlatAgent Table 13."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.runtime import DirectWorkflowAgent, run_standalone


METHOD_NAME = "workflow construction"


class Agent(DirectWorkflowAgent):
    """Remove both planning and tool retrieval."""

    retrieval_mode = "all"
    # The historical wo_TDandWR implementation stopped after an LLM error.
    catch_llm_errors = True


def main() -> None:
    run_standalone(Agent)


if __name__ == "__main__":
    main()
