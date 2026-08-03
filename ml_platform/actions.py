"""Dynamic platform action factory and process-wide action singleton."""

from __future__ import annotations

import importlib
import re
from types import ModuleType
from typing import Any, Mapping

import config

from ml_platform.action import Action, ActionConfigurationError


_PLATFORM_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _configured_platform(config_module: ModuleType) -> str:
    """Read and validate the selected platform directory name."""
    platform = getattr(config_module, "PLATFORM", None)
    if not isinstance(platform, str) or not platform:
        raise ActionConfigurationError(
            "config.PLATFORM must be a non-empty string"
        )
    if platform != platform.strip() or not _PLATFORM_NAME_PATTERN.fullmatch(
        platform
    ):
        raise ActionConfigurationError(
            "config.PLATFORM must match ^[a-z][a-z0-9_]*$ "
            f"(received {platform!r})"
        )
    return platform


def _platform_config(
    config_module: ModuleType,
    platform: str,
) -> Mapping[str, Any]:
    """Read the selected platform's ``<PLATFORM>_CONFIG`` mapping."""
    config_name = f"{platform.upper()}_CONFIG"
    platform_config = getattr(config_module, config_name, None)
    if platform_config is None:
        raise ActionConfigurationError(
            f"config.{config_name} is required when PLATFORM={platform!r}"
        )
    if not isinstance(platform_config, Mapping):
        raise ActionConfigurationError(
            f"config.{config_name} must be a mapping, "
            f"got {type(platform_config).__name__}"
        )
    return platform_config


def create_action(*, config_module: ModuleType = config) -> Action:
    """Instantiate the platform adapter selected by the root configuration.

    A platform module must be importable as
    ``ml_platform.<platform>.actions`` and export a concrete
    ``PlatformAction(Action)`` class whose constructor accepts the selected
    platform configuration mapping as its sole positional argument.
    """
    platform = _configured_platform(config_module)
    platform_config = _platform_config(config_module, platform)
    module_name = f"ml_platform.{platform}.actions"

    try:
        platform_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name.startswith(f"{exc.name}."):
            raise ActionConfigurationError(
                f"Platform module does not exist: {module_name}"
            ) from exc
        raise ActionConfigurationError(
            f"Platform {platform!r} is missing dependency {exc.name!r}"
        ) from exc
    except Exception as exc:
        raise ActionConfigurationError(
            f"Unable to import platform {platform!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    platform_class = getattr(platform_module, "PlatformAction", None)
    if not isinstance(platform_class, type):
        raise ActionConfigurationError(
            f"{module_name} must export a PlatformAction class"
        )
    if not issubclass(platform_class, Action):
        raise ActionConfigurationError(
            f"{module_name}.PlatformAction must inherit ml_platform.action.Action"
        )

    try:
        instance = platform_class(dict(platform_config))
    except ActionConfigurationError:
        raise
    except Exception as exc:
        raise ActionConfigurationError(
            f"Unable to initialize platform {platform!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(instance, Action):
        # The class check above should make this impossible, but retaining an
        # instance-level guard gives startup errors a deterministic location.
        raise ActionConfigurationError(
            f"{module_name}.PlatformAction did not create an Action instance"
        )
    if getattr(instance, "platform_name", None) != platform:
        raise ActionConfigurationError(
            f"{module_name}.PlatformAction must initialize Action with "
            f"platform_name={platform!r}"
        )
    return instance


action_agent = create_action()


__all__ = ["action_agent", "create_action"]
