from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import shutil


@dataclass(slots=True)
class EnvironmentCheck:
    name: str
    available: bool
    required: bool
    detail: str

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "available": self.available,
            "required": self.required,
            "detail": self.detail,
        }


def _python_module_check(module_name: str, required: bool) -> EnvironmentCheck:
    spec = importlib.util.find_spec(module_name)
    return EnvironmentCheck(
        name=module_name,
        available=spec is not None,
        required=required,
        detail="python module",
    )


def _executable_check(command_name: str, required: bool) -> EnvironmentCheck:
    resolved = shutil.which(command_name)
    return EnvironmentCheck(
        name=command_name,
        available=resolved is not None,
        required=required,
        detail=resolved or "not found in PATH",
    )


def collect_environment_checks(
    *,
    blender_executable: str,
    require_motion: bool,
    require_render: bool,
    require_blender: bool,
) -> list[EnvironmentCheck]:
    checks = [
        _python_module_check("numpy", required=require_motion),
        _python_module_check("cv2", required=require_motion),
        _python_module_check("mediapipe", required=require_motion),
        _executable_check(blender_executable, required=require_blender),
        _executable_check("ffmpeg", required=False),
        _executable_check("ffprobe", required=False),
    ]
    return checks


def environment_warnings_and_errors(checks: list[EnvironmentCheck]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    for check in checks:
        message = f"{check.name}: {check.detail}"
        if check.available:
            continue
        if check.name == "mediapipe" and check.required:
            errors.append(message + " (install with `python3 -m pip install mediapipe`)")
            continue
        if check.required and check.detail == "not found in PATH":
            errors.append(message + f" (install it or pass the correct executable path for {check.name})")
            continue
        if check.name == "ffmpeg":
            warnings.append(message + " (Blender movie render may still work, but PNG fallback encoding will fail)")
            continue
        if check.required:
            errors.append(message)
        else:
            warnings.append(message)
    return warnings, errors
