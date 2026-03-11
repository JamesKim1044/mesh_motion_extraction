from __future__ import annotations

from pathlib import Path

from blender.runner import run_blender_script


def run_rigging_stage(
    *,
    blender_executable: str,
    mesh_obj_path: Path,
    metadata_json_path: Path,
    blend_output_path: Path,
    report_output_path: Path | None,
    project_root: Path,
) -> Path:
    script_path = project_root / "rigging" / "blender_rigging.py"
    script_args = [
        "--mesh-obj",
        str(mesh_obj_path),
        "--metadata-json",
        str(metadata_json_path),
        "--blend-out",
        str(blend_output_path),
    ]
    if report_output_path is not None:
        script_args.extend(["--report-json", str(report_output_path)])
    run_blender_script(
        blender_executable=blender_executable,
        script_path=script_path,
        script_args=script_args,
        cwd=project_root,
    )
    return blend_output_path
