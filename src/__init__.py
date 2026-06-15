"""LocalTune core package.

Keep package initialization lightweight. Dashboard startup imports small modules
such as ``src.constants`` and should not eagerly import training dependencies.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "load_config": ("src.config", "load_config"),
    "resolve_branch": ("src.config", "resolve_branch"),
    "RuntimeConfig": ("src.config", "RuntimeConfig"),
    "QuantBranchConfig": ("src.config", "QuantBranchConfig"),
    "detect_environment": ("src.env_detect", "detect_environment"),
    "EnvInfo": ("src.env_detect", "EnvInfo"),
    "check_branch_compatibility": ("src.env_detect", "check_branch_compatibility"),
    "load_model_and_tokenizer": ("src.model_loader", "load_model_and_tokenizer"),
    "format_chatml": ("src.data_utils", "format_chatml"),
    "load_and_prepare_dataset": ("src.data_utils", "load_and_prepare_dataset"),
    "setup_logging": ("src.logging_utils", "setup_logging"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
