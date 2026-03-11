from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "MeshGenerationResult",
    "MetroMeshResult",
    "generate_humanoid_mesh_from_image",
    "generate_human_mesh_with_metro",
]


def __getattr__(name: str) -> Any:
    if name in {"MeshGenerationResult", "generate_humanoid_mesh_from_image"}:
        module = import_module(".procedural_human_mesh", __name__)
        return getattr(module, name)
    if name in {"MetroMeshResult", "generate_human_mesh_with_metro"}:
        module = import_module(".metro_mesh", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
