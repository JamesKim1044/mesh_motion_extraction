from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Iterable


class BlenderInvocationError(RuntimeError):
    pass


def run_blender_script(
    *,
    blender_executable: str,
    script_path: Path,
    script_args: Iterable[str],
    cwd: Path,
) -> None:
    if shutil.which(blender_executable) is None:
        raise BlenderInvocationError(
            f"Blender executable was not found: {blender_executable}. "
            "Install Blender or pass --blender-exe /absolute/path/to/blender."
        )
    if not script_path.exists():
        raise BlenderInvocationError(f"Blender script not found: {script_path}")
    command = [
        blender_executable,
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(script_path),
        "--",
        *[str(argument) for argument in script_args],
    ]
    try:
        completed = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    except OSError as exc:
        raise BlenderInvocationError(
            f"Failed to launch Blender with command: {' '.join(command)}"
        ) from exc
    if completed.returncode != 0:
        raise BlenderInvocationError(
            "\n".join(
                [
                    f"Blender command failed: {' '.join(command)}",
                    completed.stdout.strip(),
                    completed.stderr.strip(),
                ]
            ).strip()
        )
