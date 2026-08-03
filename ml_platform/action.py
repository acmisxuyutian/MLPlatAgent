"""Common workflow action contract for all supported ML platforms.

Concrete platforms only need to implement the workflow operations declared by
``Action``.  Widget metadata and per-run accounting state are initialized here
so the planner and executor can use every platform through the same interface.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping


class ActionError(RuntimeError):
    """Base exception for errors raised by the platform action layer."""


class ActionConfigurationError(ActionError):
    """Raised when a platform action or its widget metadata is misconfigured."""


class Action(ABC):
    """Abstract interface implemented by a machine-learning platform adapter.

    Args:
        platform_config: Configuration mapping for the selected platform.
        platform_name: Directory name below ``ml_platform``.  It is inferred
            from ``ml_platform.<name>.actions`` when omitted.
        widgets_path: Optional widget metadata path.  By default,
            ``ml_platform/<platform_name>/widgets.json`` is used.
        relevant_widgets_names: Optional initial allow-list used by Executor.

    Platform mutation methods accept one dictionary because this is the
    function-call shape used by the existing prompts and generated Python.
    """

    _ARG_COMMANDS = frozenset(
        {
            "add_node",
            "delete_node",
            "update_node_params",
            "add_edge",
            "delete_edge",
        }
    )
    _NO_ARG_COMMANDS = frozenset({"get_workflow", "clear_workflow", "reset"})

    def __init__(
        self,
        platform_config: Mapping[str, Any],
        *,
        platform_name: str | None = None,
        widgets_path: str | Path | None = None,
        relevant_widgets_names: list[str] | None = None,
    ) -> None:
        if not isinstance(platform_config, Mapping):
            raise ActionConfigurationError(
                "platform_config must be a mapping, "
                f"got {type(platform_config).__name__}"
            )

        self.platform_name = platform_name or self._infer_platform_name()
        if not self.platform_name:
            raise ActionConfigurationError(
                "platform_name could not be inferred; pass it explicitly"
            )

        self.platform_config = dict(platform_config)
        default_widgets_path = (
            Path(__file__).resolve().parent
            / self.platform_name
            / "widgets.json"
        )
        self.widgets_path = Path(
            widgets_path if widgets_path is not None else default_widgets_path
        ).expanduser().resolve()
        self.widgets = self._load_widgets(self.widgets_path)
        self.widgets_name = [widget["widget_name"] for widget in self.widgets]
        self.widgets_by_name = {
            widget["widget_name"]: widget for widget in self.widgets
        }

        self.relevant_widgets_names = list(
            dict.fromkeys(relevant_widgets_names or [])
        )

        # Shared execution/layout state used by the existing Agent pipeline.
        self.x = 30
        self.y = 300
        self.data_info: Any = None
        self.ori_data_info: Any = None
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.data_retrieval_time = 0.0

    def _infer_platform_name(self) -> str:
        """Infer the platform directory from the concrete class module."""
        module_parts = type(self).__module__.split(".")
        if (
            len(module_parts) >= 3
            and module_parts[0] == "ml_platform"
            and module_parts[-1] == "actions"
        ):
            return module_parts[-2]
        return ""

    @staticmethod
    def _load_widgets(widgets_path: Path) -> list[dict[str, Any]]:
        """Load and minimally validate a platform ``widgets.json`` file."""
        try:
            raw_widgets = json.loads(widgets_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ActionConfigurationError(
                f"Widget definition file does not exist: {widgets_path}"
            ) from exc
        except OSError as exc:
            raise ActionConfigurationError(
                f"Unable to read widget definition file {widgets_path}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ActionConfigurationError(
                "Widget definition file is not valid JSON "
                f"({widgets_path}, line {exc.lineno}, column {exc.colno})"
            ) from exc

        if not isinstance(raw_widgets, list):
            raise ActionConfigurationError(
                f"Widget definition root must be a list: {widgets_path}"
            )

        widgets: list[dict[str, Any]] = []
        widget_names: set[str] = set()
        for index, widget in enumerate(raw_widgets):
            if not isinstance(widget, dict):
                raise ActionConfigurationError(
                    f"Widget definition at index {index} must be an object"
                )

            widget_name = widget.get("widget_name")
            if not isinstance(widget_name, str) or not widget_name.strip():
                raise ActionConfigurationError(
                    f"Widget definition at index {index} must contain a "
                    "non-empty widget_name"
                )
            if widget_name in widget_names:
                raise ActionConfigurationError(
                    f"Duplicate widget_name in {widgets_path}: {widget_name}"
                )

            params = widget.get("params", [])
            if not isinstance(params, list):
                raise ActionConfigurationError(
                    f"Widget {widget_name!r} has a non-list params value"
                )
            description = widget.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ActionConfigurationError(
                    f"Widget {widget_name!r} must contain a non-empty description"
                )
            widget_type = widget.get("type")
            if not isinstance(widget_type, str) or not widget_type.strip():
                raise ActionConfigurationError(
                    f"Widget {widget_name!r} must contain a non-empty type"
                )

            parameter_names: set[str] = set()
            for parameter_index, parameter in enumerate(params):
                if not isinstance(parameter, dict):
                    raise ActionConfigurationError(
                        f"Parameter {parameter_index} of widget "
                        f"{widget_name!r} must be an object"
                    )
                parameter_name = parameter.get("name")
                if (
                    not isinstance(parameter_name, str)
                    or not parameter_name.strip()
                ):
                    raise ActionConfigurationError(
                        f"Parameter {parameter_index} of widget "
                        f"{widget_name!r} must contain a non-empty name"
                    )
                if parameter_name in parameter_names:
                    raise ActionConfigurationError(
                        f"Widget {widget_name!r} contains duplicate parameter "
                        f"{parameter_name!r}"
                    )
                parameter_type = parameter.get("type")
                if (
                    not isinstance(parameter_type, str)
                    or not parameter_type.strip()
                ):
                    raise ActionConfigurationError(
                        f"Parameter {parameter_name!r} of widget "
                        f"{widget_name!r} must contain a non-empty type"
                    )
                if "default" not in parameter:
                    raise ActionConfigurationError(
                        f"Parameter {parameter_name!r} of widget "
                        f"{widget_name!r} must define a default value"
                    )
                parameter_description = parameter.get("description")
                if (
                    not isinstance(parameter_description, str)
                    or not parameter_description.strip()
                ):
                    raise ActionConfigurationError(
                        f"Parameter {parameter_name!r} of widget "
                        f"{widget_name!r} must contain a non-empty description"
                    )
                parameter_names.add(parameter_name)

            widget_names.add(widget_name)
            widgets.append(widget)

        return widgets

    def reset(self) -> None:
        """Reset per-run counters and automatic node placement state.

        Dataset metadata and the current platform workflow intentionally remain
        intact so a subsequent ``Modify`` request can continue from them.
        """
        self.reset_XY()
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.data_retrieval_time = 0.0

    def reset_XY(self) -> None:
        """Reset the shared automatic node placement cursor."""
        self.x = 30
        self.y = 300

    def validate_node_params(
        self,
        widget_name: str,
        node_params: Mapping[str, Any],
    ) -> None:
        """Validate Agent-visible parameters against the current widget catalog."""
        widget = self.widgets_by_name.get(widget_name)
        if widget is None:
            raise ActionError(
                f"Widget {widget_name!r} is not available on "
                f"platform {self.platform_name!r}"
            )
        if not isinstance(node_params, Mapping):
            raise ActionError("node_params must be a mapping")

        definitions = {
            parameter["name"]: parameter for parameter in widget["params"]
        }
        unknown = sorted(set(node_params) - set(definitions))
        if unknown:
            raise ActionError(
                f"Widget {widget_name!r} does not define parameters: "
                + ", ".join(unknown)
            )

        validators = {
            "str": lambda value: isinstance(value, str),
            "int": lambda value: (
                isinstance(value, int) and not isinstance(value, bool)
            ),
            "bool": lambda value: isinstance(value, bool),
            "float": lambda value: (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ),
            "list": lambda value: isinstance(value, list),
            "dict": lambda value: isinstance(value, dict),
        }
        for parameter_name, value in node_params.items():
            expected_type = definitions[parameter_name].get("type")
            validator = validators.get(expected_type)
            if validator is not None and not validator(value):
                raise ActionError(
                    f"Parameter {parameter_name!r} of widget "
                    f"{widget_name!r} must have type {expected_type}"
                )

    def execute_command(
        self,
        command_name: str,
        args: Mapping[str, Any] | None = None,
    ) -> Any:
        """Dispatch a supported workflow action using the legacy call shape.

        Direct platform methods raise their native exceptions.  This dispatcher
        preserves the existing Executor contract by returning ``Error: ...`` for
        failed generated calls.
        """
        if command_name in self._ARG_COMMANDS:
            if not isinstance(args, Mapping):
                return (
                    f"Error: action {command_name!r} requires an argument mapping"
                )
            try:
                return getattr(self, command_name)(dict(args))
            except Exception as exc:
                return f"Error: {exc}"

        if command_name in self._NO_ARG_COMMANDS:
            try:
                return getattr(self, command_name)()
            except Exception as exc:
                return f"Error: {exc}"

        return f'Action "{command_name}" does not exist!'

    @abstractmethod
    def add_node(self, args: dict[str, Any]) -> Any:
        """Add a platform widget node and return its platform node ID."""

    @abstractmethod
    def delete_node(self, args: dict[str, Any]) -> Any:
        """Delete a node from the current workflow."""

    @abstractmethod
    def update_node_params(self, args: dict[str, Any]) -> Any:
        """Update a node using Agent-visible widget parameters."""

    @abstractmethod
    def add_edge(self, args: dict[str, Any]) -> Any:
        """Connect two nodes and return the platform edge ID when available."""

    @abstractmethod
    def delete_edge(self, args: dict[str, Any]) -> Any:
        """Delete an edge from the current workflow."""

    @abstractmethod
    def get_workflow(self) -> dict[str, list[dict[str, Any]]]:
        """Return a normalized ``{"nodes": [...], "edges": [...]}`` snapshot."""

    @abstractmethod
    def clear_workflow(self) -> Any:
        """Remove all nodes and edges from the current workflow."""
