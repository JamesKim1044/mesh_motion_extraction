from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np

from common import COORDINATE_SYSTEM, JOINT_ORDER, JOINT_SCHEMA_NAME, PARENT_MAP, SCALE_METADATA, read_json
from retargeting.bvh_parser import load_bvh_animation


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


def _failure(message: str, *, details: dict[str, object] | None = None) -> ValidationResult:
    return ValidationResult(ok=False, errors=[message], details=details or {})


def validate_bvh(path: Path) -> ValidationResult:
    if not path.exists():
        return _failure(f"BVH file not found: {path}")
    if path.stat().st_size <= 0:
        return _failure(f"BVH file is empty: {path}")

    try:
        animation = load_bvh_animation(path)
    except Exception as exc:
        return _failure(f"BVH parse failed: {exc}")

    if not animation.joint_order:
        return _failure("BVH hierarchy is empty")
    if not animation.frames:
        return _failure("BVH motion block contains zero frames")
    expected_channel_count = sum(len(animation.joints[joint_name].channels) for joint_name in animation.joint_order)
    for frame_index, frame in enumerate(animation.frames, start=1):
        if len(frame) != expected_channel_count:
            return _failure(
                f"BVH frame {frame_index} has {len(frame)} values, expected {expected_channel_count}"
            )

    values = np.asarray(animation.frames, dtype=np.float64)
    if not np.isfinite(values).all():
        return _failure("BVH contains NaN or infinite motion values")

    return ValidationResult(
        ok=True,
        details={
            "frame_count": len(animation.frames),
            "joint_count": len(animation.joint_order),
            "frame_time": animation.frame_time,
            "root_name": animation.root_name,
        },
    )


def validate_obj(path: Path) -> ValidationResult:
    if not path.exists():
        return _failure(f"OBJ file not found: {path}")
    vertex_count = 0
    face_count = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            vertex_count += 1
        elif line.startswith("f "):
            face_count += 1
    if vertex_count <= 0:
        return _failure("OBJ contains no vertices")
    if face_count <= 0:
        return _failure("OBJ contains no faces")
    return ValidationResult(ok=True, details={"vertex_count": vertex_count, "face_count": face_count})


def validate_mesh_metadata_json(path: Path) -> ValidationResult:
    if not path.exists():
        return _failure(f"JSON file not found: {path}")

    try:
        payload = read_json(path)
    except Exception as exc:
        return _failure(f"JSON parse failed: {exc}")

    required_keys = {
        "image_path",
        "mesh_path",
        "estimated_height_m",
        "base_color",
        "measurement_backend",
        "joint_schema",
        "coordinate_system",
        "scale",
        "rest_pose",
        "bone_mapping_hints",
        "joint_positions",
        "parent_map",
        "vertex_groups",
        "measurements_m",
    }
    missing_keys = sorted(required_keys.difference(payload))
    if missing_keys:
        return _failure(f"mesh metadata is missing required keys: {', '.join(missing_keys)}")

    errors: list[str] = []
    warnings: list[str] = []

    joint_schema = payload["joint_schema"]
    if not isinstance(joint_schema, dict):
        errors.append("joint_schema must be an object")
    else:
        if joint_schema.get("name") != JOINT_SCHEMA_NAME:
            warnings.append(f"unexpected joint schema name: {joint_schema.get('name')}")
        if joint_schema.get("joints") != JOINT_ORDER:
            errors.append("joint_schema.joints does not match expected joint order")
        if joint_schema.get("root") != "Hips":
            errors.append("joint_schema.root must be Hips")

    coordinate_system = payload["coordinate_system"]
    if not isinstance(coordinate_system, dict):
        errors.append("coordinate_system must be an object")
    else:
        for key, expected_value in COORDINATE_SYSTEM.items():
            if coordinate_system.get(key) != expected_value:
                warnings.append(f"coordinate_system.{key}={coordinate_system.get(key)!r}, expected {expected_value!r}")

    scale = payload["scale"]
    if not isinstance(scale, dict):
        errors.append("scale must be an object")
    else:
        meters_per_unit = scale.get("meters_per_unit")
        if not isinstance(meters_per_unit, (int, float)) or not math.isfinite(float(meters_per_unit)):
            errors.append("scale.meters_per_unit must be a finite number")
        elif float(meters_per_unit) <= 0:
            errors.append("scale.meters_per_unit must be > 0")
        if scale.get("bvh_unit_scale") != SCALE_METADATA["bvh_unit_scale"]:
            warnings.append(f"unexpected bvh_unit_scale: {scale.get('bvh_unit_scale')}")

    joint_positions = payload["joint_positions"]
    if not isinstance(joint_positions, dict):
        errors.append("joint_positions must be an object")
    else:
        for joint_name in JOINT_ORDER:
            point = joint_positions.get(joint_name)
            if not isinstance(point, list) or len(point) != 3:
                errors.append(f"joint_positions.{joint_name} must be a 3-element list")

    rest_pose = payload["rest_pose"]
    if not isinstance(rest_pose, dict):
        errors.append("rest_pose must be an object")
    else:
        if rest_pose.get("joint_positions") != joint_positions:
            warnings.append("rest_pose.joint_positions differs from joint_positions")

    bone_mapping_hints = payload["bone_mapping_hints"]
    if not isinstance(bone_mapping_hints, dict):
        errors.append("bone_mapping_hints must be an object")
    else:
        for joint_name in JOINT_ORDER:
            aliases = bone_mapping_hints.get(joint_name)
            if not isinstance(aliases, list) or not aliases:
                errors.append(f"bone_mapping_hints.{joint_name} must be a non-empty list")

    parent_map = payload["parent_map"]
    if parent_map != PARENT_MAP:
        warnings.append("parent_map differs from the expected humanoid parent map")

    vertex_groups = payload["vertex_groups"]
    if not isinstance(vertex_groups, dict) or not vertex_groups:
        errors.append("vertex_groups must be a non-empty object")

    ok = not errors
    return ValidationResult(
        ok=ok,
        errors=errors,
        warnings=warnings,
        details={
            "measurement_backend": payload.get("measurement_backend"),
            "estimated_height_m": payload.get("estimated_height_m"),
        },
    )


def validate_blend_file(path: Path) -> ValidationResult:
    if not path.exists():
        return _failure(f"Blend file not found: {path}")
    if path.stat().st_size <= 0:
        return _failure(f"Blend file is empty: {path}")
    header = path.read_bytes()[:32]
    if b"BLENDER" not in header:
        return _failure(f"Blend file does not have a Blender header: {path}")
    return ValidationResult(ok=True, details={"size_bytes": path.stat().st_size})


def validate_mp4(path: Path) -> ValidationResult:
    if not path.exists():
        return _failure(f"MP4 file not found: {path}")
    if path.stat().st_size <= 0:
        return _failure(f"MP4 file is empty: {path}")

    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        return ValidationResult(
            ok=True,
            warnings=["ffprobe not found; only existence and file size were validated"],
            details={"size_bytes": path.stat().st_size},
        )

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_name,width,height",
        "-of",
        "default=noprint_wrappers=1",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        return _failure(f"ffprobe failed for MP4: {completed.stderr.strip() or completed.stdout.strip()}")
    details: dict[str, object] = {"size_bytes": path.stat().st_size}
    warnings: list[str] = []
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            details[key] = value
    if details.get("codec_name") not in {None, "h264"}:
        warnings.append(f"unexpected MP4 codec: {details.get('codec_name')}")
    if "width" in details and "height" in details:
        try:
            width = int(str(details["width"]))
            height = int(str(details["height"]))
            if width <= 0 or height <= 0:
                warnings.append("MP4 stream resolution is invalid")
        except ValueError:
            warnings.append("MP4 width/height are not numeric")
    return ValidationResult(ok=True, warnings=warnings, details=details)


def validate_path_exists(path: Path, label: str) -> ValidationResult:
    if not path.exists():
        return _failure(f"{label} not found: {path}")
    return ValidationResult(ok=True, details={"path": str(path)})
