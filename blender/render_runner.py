from __future__ import annotations

from pathlib import Path

from blender.runner import run_blender_script


def run_render_stage(
    *,
    blender_executable: str,
    input_blend_path: Path,
    metadata_json_path: Path,
    output_video_path: Path,
    project_root: Path,
    resolution_x: int = 1920,
    resolution_y: int = 1080,
) -> Path:
    script_path = project_root / "blender" / "render_animation.py"
    run_blender_script(
        blender_executable=blender_executable,
        script_path=script_path,
        script_args=[
            "--blend-in",
            str(input_blend_path),
            "--metadata-json",
            str(metadata_json_path),
            "--output-video",
            str(output_video_path),
            "--resolution-x",
            str(resolution_x),
            "--resolution-y",
            str(resolution_y),
        ],
        cwd=project_root,
    )
    return output_video_path
