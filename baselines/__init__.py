"""Independently runnable baseline implementations for MLPlatAgent."""

from __future__ import annotations

from typing import Any


def get_agent_classes() -> dict[str, type[Any]]:
    """Return baseline display names and their Agent implementations."""
    from baselines.dfsdt import Agent as DFSDTAgent
    from baselines.function_call import Agent as FunctionCallAgent
    from baselines.react import Agent as ReActAgent
    from baselines.wo_planning import Agent as WithoutPlanningAgent
    from baselines.wo_tool_retrieval import (
        Agent as WithoutToolRetrievalAgent,
    )
    from baselines.workflow_construction import (
        Agent as WorkflowConstructionAgent,
    )

    return {
        "wo_planning": WithoutPlanningAgent,
        "wo_tool_retrieval": WithoutToolRetrievalAgent,
        "workflow_construction": WorkflowConstructionAgent,
        "ReAct": ReActAgent,
        "FunctionCall": FunctionCallAgent,
        "DFSDT": DFSDTAgent,
    }
