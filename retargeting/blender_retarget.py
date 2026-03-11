from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import (  # noqa: E402
    ARMATURE_OBJECT_NAME,
    JOINT_ORDER,
    LEAF_DIRECTIONS,
    PRIMARY_CHILD_MAP,
    bvh_vector_to_blender,
    read_json,
)
from retargeting.bvh_parser import (  # noqa: E402
    all_frame_positions,
    apply_axis_multipliers,
    build_joint_mapping,
    estimate_height,
    load_bvh_animation,
    map_positions_to_targets,
    rest_positions,
    translation_channel_warnings,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget BVH joint motion onto the generated Blender armature.")
    parser.add_argument("--blend-in", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--blend-out", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args(argv)


def write_report(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def require_payload_keys(payload: dict[str, object], required_keys: list[str]) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise RuntimeError(f"metadata is missing required keys: {', '.join(missing)}")


def validate_target_armature(armature_object: bpy.types.Object) -> list[str]:
    missing_bones = [joint_name for joint_name in JOINT_ORDER if armature_object.data.bones.get(joint_name) is None]
    if missing_bones:
        raise RuntimeError(
            "target armature is missing required schema bones: "
            + ", ".join(missing_bones)
            + ". Rebuild the rigged scene from the current generated_human.json metadata."
        )
    return []


def create_motion_targets() -> tuple[dict[str, bpy.types.Object], dict[str, bpy.types.Object]]:
    collection = bpy.data.collections.new("MotionTargets")
    bpy.context.scene.collection.children.link(collection)
    collection.hide_viewport = True
    collection.hide_render = True

    joint_targets: dict[str, bpy.types.Object] = {}
    tip_targets: dict[str, bpy.types.Object] = {}

    for joint_name in JOINT_ORDER:
        empty = bpy.data.objects.new(f"Target_{joint_name}", None)
        empty.empty_display_type = "PLAIN_AXES"
        empty.empty_display_size = 0.04
        collection.objects.link(empty)
        joint_targets[joint_name] = empty

    for joint_name in LEAF_DIRECTIONS:
        empty = bpy.data.objects.new(f"Target_{joint_name}_Tip", None)
        empty.empty_display_type = "PLAIN_AXES"
        empty.empty_display_size = 0.03
        collection.objects.link(empty)
        tip_targets[joint_name] = empty

    return joint_targets, tip_targets


def add_track_constraints(
    armature_object: bpy.types.Object,
    joint_targets: dict[str, bpy.types.Object],
    tip_targets: dict[str, bpy.types.Object],
) -> None:
    for pose_bone in armature_object.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
        for constraint in list(pose_bone.constraints):
            pose_bone.constraints.remove(constraint)

        primary_child = PRIMARY_CHILD_MAP.get(pose_bone.name)
        target_object = joint_targets.get(primary_child) if primary_child else tip_targets.get(pose_bone.name)
        if target_object is None:
            continue

        constraint = pose_bone.constraints.new(type="DAMPED_TRACK")
        constraint.name = f"Track_{pose_bone.name}"
        constraint.target = target_object
        constraint.track_axis = "TRACK_Y"


def source_vector_to_blender(vector: tuple[float, float, float], scale: float) -> Vector:
    scaled = (vector[0] * scale, vector[1] * scale, vector[2] * scale)
    return Vector(bvh_vector_to_blender(scaled))


def set_linear_interpolation(action: bpy.types.Action | None) -> None:
    if action is None or not hasattr(action, "fcurves"):
        return
    for fcurve in action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"


def target_rest_relatives(target_rest: dict[str, Vector]) -> dict[str, Vector]:
    root = target_rest["Hips"]
    return {joint_name: position - root for joint_name, position in target_rest.items()}


def fallback_leaf_parent(joint_name: str) -> str:
    if joint_name == "Head":
        return "Neck"
    if joint_name == "LeftHand":
        return "LeftWrist"
    if joint_name == "RightHand":
        return "RightWrist"
    if joint_name == "LeftFoot":
        return "LeftAnkle"
    return "RightAnkle"


def main() -> None:
    args = parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    payload = read_json(args.metadata_json)
    require_payload_keys(payload, ["joint_positions", "estimated_height_m", "bone_mapping_hints"])

    bpy.ops.wm.open_mainfile(filepath=str(args.blend_in))

    armature_object = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    if armature_object is None or armature_object.type != "ARMATURE":
        raise RuntimeError(f"rigged blend file does not contain an {ARMATURE_OBJECT_NAME} armature")
    warnings: list[str] = validate_target_armature(armature_object)

    animation = load_bvh_animation(args.bvh)
    source_frames_all = all_frame_positions(animation)
    if not source_frames_all:
        raise RuntimeError(f"BVH file does not contain any frames: {args.bvh}")

    axis_multipliers = (1.0, 1.0, 1.0)
    mapping_result = build_joint_mapping(
        animation,
        target_joint_names=JOINT_ORDER,
        bone_mapping_hints=payload.get("bone_mapping_hints"),
    )
    warnings.extend(mapping_result.warnings)
    warnings.extend(translation_channel_warnings(animation, mapping_result.mapping))
    if "Hips" not in mapping_result.mapping:
        raise RuntimeError(
            "BVH retargeting requires a mapped Hips/root joint. "
            "Update generated_human.json bone_mapping_hints or provide a compatible BVH."
        )
    source_rest_raw = apply_axis_multipliers(rest_positions(animation), axis_multipliers)
    source_rest = map_positions_to_targets(source_rest_raw, mapping_result.mapping)
    source_frames = [
        map_positions_to_targets(apply_axis_multipliers(frame_positions, axis_multipliers), mapping_result.mapping)
        for frame_positions in source_frames_all
    ]
    if not any(source_frame for source_frame in source_frames):
        raise RuntimeError("no source BVH joints could be mapped onto the target armature")

    target_rest = {
        joint_name: Vector(payload["joint_positions"][joint_name]) for joint_name in JOINT_ORDER
    }
    target_rest_relative = target_rest_relatives(target_rest)

    source_height = estimate_height(source_rest) if source_rest else estimate_height(source_rest_raw)
    target_height = float(payload["estimated_height_m"])
    motion_scale = target_height / source_height if source_height > 1e-6 else 1.0
    if source_height <= 1e-6:
        warnings.append("source BVH height estimate was near zero; falling back to motion_scale=1.0")
    source_root_rest = source_rest.get("Hips", source_rest_raw.get(animation.root_name, (0.0, 0.0, 0.0)))
    leaf_lengths = {
        joint_name: max(float(armature_object.data.bones[joint_name].length), 0.05)
        for joint_name in LEAF_DIRECTIONS
    }

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(source_frames)
    scene.render.fps = max(1, int(round(1.0 / animation.frame_time)))

    joint_targets, tip_targets = create_motion_targets()
    add_track_constraints(armature_object, joint_targets, tip_targets)

    armature_object.location = Vector((0.0, 0.0, 0.0))
    if armature_object.animation_data is not None:
        armature_object.animation_data_clear()

    for frame_index, source_frame in enumerate(source_frames, start=1):
        source_root = source_frame.get("Hips", source_root_rest)
        root_motion = (
            source_root[0] - source_root_rest[0],
            source_root[1] - source_root_rest[1],
            source_root[2] - source_root_rest[2],
        )
        locomotion = source_vector_to_blender(root_motion, motion_scale)
        armature_object.location = locomotion
        armature_object.keyframe_insert(data_path="location", frame=frame_index)

        for joint_name in JOINT_ORDER:
            source_rest_position = source_rest.get(joint_name, source_root_rest)
            source_frame_position = source_frame.get(joint_name, source_rest_position)
            source_rest_relative = (
                source_rest_position[0] - source_root_rest[0],
                source_rest_position[1] - source_root_rest[1],
                source_rest_position[2] - source_root_rest[2],
            )
            source_frame_relative = (
                source_frame_position[0] - source_root[0],
                source_frame_position[1] - source_root[1],
                source_frame_position[2] - source_root[2],
            )
            pose_delta = (
                source_frame_relative[0] - source_rest_relative[0],
                source_frame_relative[1] - source_rest_relative[1],
                source_frame_relative[2] - source_rest_relative[2],
            )
            world_position = (
                target_rest["Hips"]
                + locomotion
                + target_rest_relative[joint_name]
                + source_vector_to_blender(pose_delta, motion_scale)
            )
            joint_targets[joint_name].location = world_position
            joint_targets[joint_name].keyframe_insert(data_path="location", frame=frame_index)

        for joint_name, tip_target in tip_targets.items():
            parent_name = fallback_leaf_parent(joint_name)
            parent_target = joint_targets[parent_name].location
            joint_target = joint_targets[joint_name].location
            direction = joint_target - parent_target
            if direction.length < 1e-6:
                direction = Vector(LEAF_DIRECTIONS[joint_name])
            else:
                direction.normalize()
            tip_target.location = joint_target + direction * leaf_lengths[joint_name]
            tip_target.keyframe_insert(data_path="location", frame=frame_index)

    bpy.context.view_layer.objects.active = armature_object
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.nla.bake(
        frame_start=1,
        frame_end=len(source_frames),
        step=1,
        only_selected=True,
        visual_keying=True,
        clear_constraints=True,
        use_current_action=True,
        clean_curves=False,
        bake_types={"POSE"},
    )
    bpy.ops.object.mode_set(mode="OBJECT")

    set_linear_interpolation(armature_object.animation_data.action if armature_object.animation_data else None)

    report = {
        "status": "ok",
        "blend_in": str(args.blend_in),
        "bvh": str(args.bvh),
        "blend_out": str(args.blend_out),
        "source_root": animation.root_name,
        "frame_count": len(source_frames),
        "frame_time": animation.frame_time,
        "motion_scale": float(motion_scale),
        "axis_multipliers": list(axis_multipliers),
        "joint_mapping": mapping_result.mapping,
        "warnings": warnings,
    }
    write_report(args.report_json, report)

    args.blend_out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.blend_out))


if __name__ == "__main__":
    main()
