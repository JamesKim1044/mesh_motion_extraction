from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from common import DEFAULT_BODY_HEIGHT_M, ensure_directory, iso_timestamp, write_json
from pipeline_environment import collect_environment_checks, environment_warnings_and_errors
from pipeline_validation import (
    ValidationResult,
    validate_blend_file,
    validate_bvh,
    validate_mesh_metadata_json,
    validate_mp4,
    validate_obj,
    validate_path_exists,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_ROOT / "taekwondo-1.mp4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "demo"


@dataclass(slots=True)
class PipelineArtifacts:
    input_image: Path
    motion_bvh: Path
    motion_overlay: Path
    mesh_obj: Path
    mesh_metadata: Path
    rig_blend: Path
    rig_report: Path
    animated_blend: Path
    retarget_report: Path
    render_mp4: Path
    manifest: Path


@dataclass(slots=True)
class StageDefinition:
    name: str
    inputs: dict[str, Path]
    outputs: dict[str, Path]
    optional_outputs: dict[str, Path] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restartable end-to-end motion retargeting pipeline using MediaPipe and Blender.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="input video for motion extraction")
    parser.add_argument("--image", type=Path, default=None, help="single RGB image used for mesh generation")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for all generated outputs")
    parser.add_argument("--blender-exe", type=str, default="blender", help="Blender executable to use")
    parser.add_argument("--target-height-m", type=float, default=DEFAULT_BODY_HEIGHT_M, help="generated character height in meters")
    parser.add_argument("--frame-step", type=int, default=1, help="process every Nth video frame during pose extraction")
    parser.add_argument("--resolution-x", type=int, default=1920, help="render width")
    parser.add_argument("--resolution-y", type=int, default=1080, help="render height")
    parser.add_argument("--skip-motion", action="store_true", help="skip motion extraction and reuse an existing BVH")
    parser.add_argument("--skip-mesh", action="store_true", help="skip mesh generation and reuse existing mesh artifacts")
    parser.add_argument("--skip-rig", action="store_true", help="skip Blender rig generation")
    parser.add_argument("--skip-retarget", action="store_true", help="skip BVH retargeting")
    parser.add_argument("--skip-render", action="store_true", help="skip final MP4 rendering")
    parser.add_argument("--resume", action="store_true", help="reuse valid up-to-date artifacts when available")
    parser.add_argument("--force", action="store_true", help="recompute all non-skipped stages even if outputs already exist")
    parser.add_argument("--validate-only", action="store_true", help="validate stage outputs and write a manifest without running stages")
    return parser.parse_args()


def extract_first_frame(video_path: Path, image_path: Path) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to extract a frame from video: {video_path}")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"failed to save extracted frame to {image_path}")
    return image_path


def build_artifacts(output_dir: Path, input_image: Path) -> PipelineArtifacts:
    return PipelineArtifacts(
        input_image=input_image,
        motion_bvh=output_dir / "motion.bvh",
        motion_overlay=output_dir / "motion_overlay.mp4",
        mesh_obj=output_dir / "generated_human.obj",
        mesh_metadata=output_dir / "generated_human.json",
        rig_blend=output_dir / "rigged_scene.blend",
        rig_report=output_dir / "rig_report.json",
        animated_blend=output_dir / "animated_scene.blend",
        retarget_report=output_dir / "retarget_report.json",
        render_mp4=output_dir / "retargeted_animation.mp4",
        manifest=output_dir / "pipeline_manifest.json",
    )


def build_stage_definitions(args: argparse.Namespace, artifacts: PipelineArtifacts) -> list[StageDefinition]:
    return [
        StageDefinition(
            name="motion",
            inputs={"video": args.video},
            outputs={"motion_bvh": artifacts.motion_bvh},
            optional_outputs={"motion_overlay": artifacts.motion_overlay},
        ),
        StageDefinition(
            name="mesh",
            inputs={"image": artifacts.input_image},
            outputs={
                "mesh_obj": artifacts.mesh_obj,
                "mesh_metadata": artifacts.mesh_metadata,
            },
        ),
        StageDefinition(
            name="rig",
            inputs={
                "mesh_obj": artifacts.mesh_obj,
                "mesh_metadata": artifacts.mesh_metadata,
            },
            outputs={"rig_blend": artifacts.rig_blend},
            optional_outputs={"rig_report": artifacts.rig_report},
        ),
        StageDefinition(
            name="retarget",
            inputs={
                "rig_blend": artifacts.rig_blend,
                "mesh_metadata": artifacts.mesh_metadata,
                "motion_bvh": artifacts.motion_bvh,
            },
            outputs={"animated_blend": artifacts.animated_blend},
            optional_outputs={"retarget_report": artifacts.retarget_report},
        ),
        StageDefinition(
            name="render",
            inputs={
                "animated_blend": artifacts.animated_blend,
                "mesh_metadata": artifacts.mesh_metadata,
            },
            outputs={"render_mp4": artifacts.render_mp4},
        ),
    ]


def combine_validation(results: dict[str, ValidationResult]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, object] = {}
    for label, result in results.items():
        details[label] = result.to_dict()
        errors.extend(result.errors)
        warnings.extend(result.warnings)
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, details=details)


def validate_stage(stage: StageDefinition) -> ValidationResult:
    if stage.name == "motion":
        results = {
            "motion_bvh": validate_bvh(stage.outputs["motion_bvh"]),
        }
        if stage.optional_outputs["motion_overlay"].exists():
            results["motion_overlay"] = validate_path_exists(stage.optional_outputs["motion_overlay"], "Motion overlay")
        return combine_validation(results)

    if stage.name == "mesh":
        return combine_validation(
            {
                "mesh_obj": validate_obj(stage.outputs["mesh_obj"]),
                "mesh_metadata": validate_mesh_metadata_json(stage.outputs["mesh_metadata"]),
            }
        )

    if stage.name == "rig":
        results = {
            "rig_blend": validate_blend_file(stage.outputs["rig_blend"]),
        }
        if stage.optional_outputs["rig_report"].exists():
            results["rig_report"] = validate_path_exists(stage.optional_outputs["rig_report"], "Rig report")
        return combine_validation(results)

    if stage.name == "retarget":
        results = {
            "animated_blend": validate_blend_file(stage.outputs["animated_blend"]),
        }
        if stage.optional_outputs["retarget_report"].exists():
            results["retarget_report"] = validate_path_exists(
                stage.optional_outputs["retarget_report"],
                "Retarget report",
            )
        return combine_validation(results)

    if stage.name == "render":
        return combine_validation({"render_mp4": validate_mp4(stage.outputs["render_mp4"])})

    raise ValueError(f"unknown stage: {stage.name}")


def outputs_are_fresh(stage: StageDefinition) -> bool:
    required_outputs = list(stage.outputs.values())
    if not required_outputs or any(not path.exists() for path in required_outputs):
        return False

    existing_inputs = [path for path in stage.inputs.values() if path.exists()]
    if not existing_inputs:
        return True

    newest_input = max(path.stat().st_mtime for path in existing_inputs)
    oldest_output = min(path.stat().st_mtime for path in required_outputs)
    return oldest_output >= newest_input


def stage_status_key(stage_name: str) -> str:
    return f"skip_{stage_name}"


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    manifest["updated_at"] = iso_timestamp()
    write_json(path, manifest)


def initialize_manifest(
    args: argparse.Namespace,
    artifacts: PipelineArtifacts,
    stages: list[StageDefinition],
) -> dict[str, object]:
    return {
        "created_at": iso_timestamp(),
        "updated_at": iso_timestamp(),
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(args.output_dir),
        "artifacts": {
            "input_image": str(artifacts.input_image),
            "motion_bvh": str(artifacts.motion_bvh),
            "motion_overlay": str(artifacts.motion_overlay),
            "mesh_obj": str(artifacts.mesh_obj),
            "mesh_metadata": str(artifacts.mesh_metadata),
            "rig_blend": str(artifacts.rig_blend),
            "rig_report": str(artifacts.rig_report),
            "animated_blend": str(artifacts.animated_blend),
            "retarget_report": str(artifacts.retarget_report),
            "render_mp4": str(artifacts.render_mp4),
            "manifest": str(artifacts.manifest),
        },
        "environment": [],
        "warnings": [],
        "stages": {
            stage.name: {
                "status": "pending",
                "inputs": {key: str(value) for key, value in stage.inputs.items()},
                "outputs": {key: str(value) for key, value in stage.outputs.items()},
                "optional_outputs": {key: str(value) for key, value in stage.optional_outputs.items()},
                "timestamps": {},
                "warnings": [],
                "errors": [],
                "validation": {},
            }
            for stage in stages
        },
    }


def record_stage(
    manifest: dict[str, object],
    stage_name: str,
    *,
    status: str,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    validation: ValidationResult | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> None:
    stage_payload = manifest["stages"][stage_name]
    stage_payload["status"] = status
    stage_payload["warnings"] = warnings or []
    stage_payload["errors"] = errors or []
    stage_payload["validation"] = validation.to_dict() if validation is not None else {}
    timestamps = {}
    if started_at is not None:
        timestamps["started_at"] = started_at
    if finished_at is not None:
        timestamps["finished_at"] = finished_at
    stage_payload["timestamps"] = timestamps


def resolve_mesh_input(
    *,
    args: argparse.Namespace,
    artifacts: PipelineArtifacts,
    mesh_stage_will_run: bool,
) -> tuple[Path, list[str]]:
    warnings: list[str] = []
    if args.image is not None:
        return args.image, warnings
    if artifacts.input_image.exists() and (args.resume or args.validate_only or not mesh_stage_will_run):
        return artifacts.input_image, warnings
    if not mesh_stage_will_run:
        warnings.append(
            f"mesh input image is not present at {artifacts.input_image}; existing mesh artifacts must be reused"
        )
        return artifacts.input_image, warnings
    extracted = extract_first_frame(args.video, artifacts.input_image)
    warnings.append(f"generated mesh input image from first video frame: {extracted}")
    return extracted, warnings


def run_motion_stage(args: argparse.Namespace, stage: StageDefinition) -> list[str]:
    from motion_capture import capture_motion_to_bvh

    result = capture_motion_to_bvh(
        video_path=stage.inputs["video"],
        bvh_path=stage.outputs["motion_bvh"],
        overlay_path=stage.optional_outputs["motion_overlay"],
        scale=1.0,
        frame_step=args.frame_step,
    )
    return result.warnings + [
        f"bvh_path={result.bvh_path}",
        f"frame_count={result.frame_count}",
        f"fps={result.fps:.3f}",
    ]


def run_mesh_stage(args: argparse.Namespace, stage: StageDefinition) -> list[str]:
    from mesh_generation import generate_humanoid_mesh_from_image

    result = generate_humanoid_mesh_from_image(
        image_path=stage.inputs["image"],
        output_obj_path=stage.outputs["mesh_obj"],
        metadata_path=stage.outputs["mesh_metadata"],
        target_height_m=args.target_height_m,
    )
    return [f"estimated_height_m={result.estimated_height_m:.4f}"]


def run_rig_stage(args: argparse.Namespace, stage: StageDefinition) -> list[str]:
    from rigging import run_rigging_stage

    run_rigging_stage(
        blender_executable=args.blender_exe,
        mesh_obj_path=stage.inputs["mesh_obj"],
        metadata_json_path=stage.inputs["mesh_metadata"],
        blend_output_path=stage.outputs["rig_blend"],
        report_output_path=stage.optional_outputs["rig_report"],
        project_root=PROJECT_ROOT,
    )
    return [f"rigged_blend={stage.outputs['rig_blend']}"]


def run_retarget_stage(args: argparse.Namespace, stage: StageDefinition) -> list[str]:
    from retargeting import run_retargeting_stage

    run_retargeting_stage(
        blender_executable=args.blender_exe,
        input_blend_path=stage.inputs["rig_blend"],
        metadata_json_path=stage.inputs["mesh_metadata"],
        bvh_path=stage.inputs["motion_bvh"],
        blend_output_path=stage.outputs["animated_blend"],
        report_output_path=stage.optional_outputs["retarget_report"],
        project_root=PROJECT_ROOT,
    )
    return [f"animated_blend={stage.outputs['animated_blend']}"]


def run_render_stage_wrapper(args: argparse.Namespace, stage: StageDefinition) -> list[str]:
    from blender.render_runner import run_render_stage

    run_render_stage(
        blender_executable=args.blender_exe,
        input_blend_path=stage.inputs["animated_blend"],
        metadata_json_path=stage.inputs["mesh_metadata"],
        output_video_path=stage.outputs["render_mp4"],
        project_root=PROJECT_ROOT,
        resolution_x=args.resolution_x,
        resolution_y=args.resolution_y,
    )
    return [f"render_mp4={stage.outputs['render_mp4']}"]


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    initial_input_image = args.image if args.image is not None else output_dir / "mesh_input.png"
    artifacts = build_artifacts(output_dir, initial_input_image)
    stages = build_stage_definitions(args, artifacts)

    reusable_now = {
        stage.name: args.resume and not args.force and validate_stage(stage).ok and outputs_are_fresh(stage)
        for stage in stages
    }
    stage_execution_plan = {stage.name: False for stage in stages}
    if not args.validate_only:
        motion_requested = not args.skip_motion
        mesh_requested = not args.skip_mesh
        rig_requested = not args.skip_rig
        retarget_requested = not args.skip_retarget
        render_requested = not args.skip_render

        stage_execution_plan["motion"] = motion_requested and (args.force or not reusable_now["motion"])
        stage_execution_plan["mesh"] = mesh_requested and (args.force or not reusable_now["mesh"])
        stage_execution_plan["rig"] = rig_requested and (
            args.force or stage_execution_plan["mesh"] or not reusable_now["rig"]
        )
        stage_execution_plan["retarget"] = retarget_requested and (
            args.force
            or stage_execution_plan["motion"]
            or stage_execution_plan["mesh"]
            or stage_execution_plan["rig"]
            or not reusable_now["retarget"]
        )
        stage_execution_plan["render"] = render_requested and (
            args.force
            or stage_execution_plan["mesh"]
            or stage_execution_plan["rig"]
            or stage_execution_plan["retarget"]
            or not reusable_now["render"]
        )

    mesh_input_path, mesh_input_warnings = resolve_mesh_input(
        args=args,
        artifacts=artifacts,
        mesh_stage_will_run=stage_execution_plan["mesh"],
    )
    artifacts.input_image = mesh_input_path
    stages = build_stage_definitions(args, artifacts)
    manifest = initialize_manifest(args, artifacts, stages)
    manifest["warnings"].extend(mesh_input_warnings)

    reusable_now = {
        stage.name: args.resume and not args.force and validate_stage(stage).ok and outputs_are_fresh(stage)
        for stage in stages
    }
    stage_execution_plan = {stage.name: False for stage in stages}
    if not args.validate_only:
        motion_requested = not args.skip_motion
        mesh_requested = not args.skip_mesh
        rig_requested = not args.skip_rig
        retarget_requested = not args.skip_retarget
        render_requested = not args.skip_render

        stage_execution_plan["motion"] = motion_requested and (args.force or not reusable_now["motion"])
        stage_execution_plan["mesh"] = mesh_requested and (args.force or not reusable_now["mesh"])
        stage_execution_plan["rig"] = rig_requested and (
            args.force or stage_execution_plan["mesh"] or not reusable_now["rig"]
        )
        stage_execution_plan["retarget"] = retarget_requested and (
            args.force
            or stage_execution_plan["motion"]
            or stage_execution_plan["mesh"]
            or stage_execution_plan["rig"]
            or not reusable_now["retarget"]
        )
        stage_execution_plan["render"] = render_requested and (
            args.force
            or stage_execution_plan["mesh"]
            or stage_execution_plan["rig"]
            or stage_execution_plan["retarget"]
            or not reusable_now["render"]
        )

    checks = collect_environment_checks(
        blender_executable=args.blender_exe,
        require_motion=stage_execution_plan["motion"],
        require_render=stage_execution_plan["render"],
        require_blender=stage_execution_plan["rig"] or stage_execution_plan["retarget"] or stage_execution_plan["render"],
    )
    env_warnings, env_errors = environment_warnings_and_errors(checks)
    manifest["environment"] = [check.to_dict() for check in checks]
    manifest["warnings"].extend(env_warnings)
    write_manifest(artifacts.manifest, manifest)
    if env_errors:
        raise RuntimeError("environment checks failed:\n" + "\n".join(env_errors))

    stage_runners = {
        "motion": run_motion_stage,
        "mesh": run_mesh_stage,
        "rig": run_rig_stage,
        "retarget": run_retarget_stage,
        "render": run_render_stage_wrapper,
    }

    validation_failures: list[str] = []
    for stage in stages:
        skip = bool(getattr(args, stage_status_key(stage.name)))
        if skip:
            record_stage(
                manifest,
                stage.name,
                status="skipped",
                warnings=[f"stage skipped via --skip-{stage.name}"],
            )
            write_manifest(artifacts.manifest, manifest)
            continue

        if args.validate_only:
            validation = validate_stage(stage)
            record_stage(
                manifest,
                stage.name,
                status="validated" if validation.ok else "failed",
                warnings=validation.warnings,
                errors=validation.errors,
                validation=validation,
            )
            write_manifest(artifacts.manifest, manifest)
            if not validation.ok:
                validation_failures.append(f"{stage.name}: {'; '.join(validation.errors)}")
            continue

        if args.resume and not stage_execution_plan[stage.name]:
            validation = validate_stage(stage)
            warnings = validation.warnings
            if not outputs_are_fresh(stage):
                warnings = warnings + ["stage outputs are stale relative to inputs"]
            record_stage(
                manifest,
                stage.name,
                status="reused",
                warnings=warnings,
                validation=validation,
            )
            write_manifest(artifacts.manifest, manifest)
            continue

        missing_inputs = [f"{label}: {path}" for label, path in stage.inputs.items() if not path.exists()]
        if missing_inputs:
            error_message = (
                f"cannot run stage {stage.name}; missing inputs: {', '.join(missing_inputs)}. "
                f"Use --resume to reuse prior artifacts, regenerate upstream stages, or skip {stage.name} explicitly."
            )
            record_stage(
                manifest,
                stage.name,
                status="failed",
                errors=[error_message],
            )
            write_manifest(artifacts.manifest, manifest)
            raise RuntimeError(error_message)

        started_at = iso_timestamp()
        try:
            warnings = stage_runners[stage.name](args, stage)
            validation = validate_stage(stage)
            if not validation.ok:
                raise RuntimeError("; ".join(validation.errors))
            record_stage(
                manifest,
                stage.name,
                status="completed",
                warnings=warnings + validation.warnings,
                validation=validation,
                started_at=started_at,
                finished_at=iso_timestamp(),
            )
            write_manifest(artifacts.manifest, manifest)
        except Exception as exc:
            record_stage(
                manifest,
                stage.name,
                status="failed",
                errors=[str(exc)],
                started_at=started_at,
                finished_at=iso_timestamp(),
            )
            write_manifest(artifacts.manifest, manifest)
            raise

    if validation_failures:
        raise RuntimeError("validation failed:\n" + "\n".join(validation_failures))


if __name__ == "__main__":
    main()
