from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional


def optional_import(name: str) -> Optional[ModuleType]:
    """Try import a module, return None if not available."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def require_module(name: str) -> ModuleType:
    """Import a module, raise a friendly error if missing."""
    mod = optional_import(name)
    if mod is None:
        raise ImportError(
            f"缺少依赖: {name}\n"
            "你可以按 README 的方式安装依赖，或运行：\n"
            f"  python -m pip install {name}\n"
        )
    return mod
