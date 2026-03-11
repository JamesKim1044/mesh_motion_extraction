from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


CANONICAL_BONE_ALIASES: dict[str, list[str]] = {
    "Hips": ["Hips", "hips", "Pelvis", "pelvis", "mixamorig:Hips"],
    "Spine": ["Spine", "spine", "Spine1", "spine_01", "mixamorig:Spine"],
    "Chest": ["Chest", "chest", "Spine2", "spine_02", "mixamorig:Spine1"],
    "Neck": ["Neck", "neck", "neck_01", "mixamorig:Neck"],
    "Head": ["Head", "head", "mixamorig:Head"],
    "LeftShoulder": ["LeftShoulder", "shoulder_l", "mixamorig:LeftShoulder"],
    "LeftElbow": ["LeftElbow", "lowerarm_l", "forearm_l", "mixamorig:LeftForeArm"],
    "LeftWrist": ["LeftWrist", "hand_l", "mixamorig:LeftHand"],
    "LeftHand": ["LeftHand", "hand_l_end", "mixamorig:LeftHandMiddle1"],
    "RightShoulder": ["RightShoulder", "shoulder_r", "mixamorig:RightShoulder"],
    "RightElbow": ["RightElbow", "lowerarm_r", "forearm_r", "mixamorig:RightForeArm"],
    "RightWrist": ["RightWrist", "hand_r", "mixamorig:RightHand"],
    "RightHand": ["RightHand", "hand_r_end", "mixamorig:RightHandMiddle1"],
    "LeftHip": ["LeftHip", "thigh_l", "mixamorig:LeftUpLeg"],
    "LeftKnee": ["LeftKnee", "calf_l", "shin_l", "mixamorig:LeftLeg"],
    "LeftAnkle": ["LeftAnkle", "foot_l", "mixamorig:LeftFoot"],
    "LeftFoot": ["LeftFoot", "toe_l", "ball_l", "mixamorig:LeftToeBase"],
    "RightHip": ["RightHip", "thigh_r", "mixamorig:RightUpLeg"],
    "RightKnee": ["RightKnee", "calf_r", "shin_r", "mixamorig:RightLeg"],
    "RightAnkle": ["RightAnkle", "foot_r", "mixamorig:RightFoot"],
    "RightFoot": ["RightFoot", "toe_r", "ball_r", "mixamorig:RightToeBase"],
}

COPY_ROTATION_PREFIX = "BVH_ROT_"
COPY_LOCATION_PREFIX = "BVH_LOC_"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a BVH animation, map it to a target armature, and bake joint rotations.",
    )
    parser.add_argument("--bvh", type=Path, required=True, help="BVH file to import")
    parser.add_argument("--target-armature", type=str, required=True, help="name of the target armature object")
    parser.add_argument("--blend-in", type=Path, default=None, help="optional source .blend file to open first")
    parser.add_argument("--blend-out", type=Path, default=None, help="optional output .blend path")
    parser.add_argument(
        "--bone-map-json",
        type=Path,
        default=None,
        help="optional JSON file mapping target bone names to source bone names",
    )
    parser.add_argument(
        "--global-scale",
        type=float,
        default=1.0,
        help="scale used during BVH import",
    )
    parser.add_argument(
        "--copy-root-location",
        action="store_true",
        help="copy root bone translation in addition to rotations",
    )
    parser.add_argument(
        "--keep-source-armature",
        action="store_true",
        help="keep the imported BVH armature after baking",
    )
    return parser.parse_args(argv)


def ensure_object_mode() -> None:
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def open_blend_if_needed(blend_path: Path | None) -> None:
    if blend_path is None:
        return
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))


def import_bvh_armature(bvh_path: Path, global_scale: float) -> bpy.types.Object:
    if not bvh_path.exists():
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")

    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.import_anim.bvh(
        filepath=str(bvh_path),
        global_scale=global_scale,
        use_fps_scale=True,
        update_scene_fps=True,
        update_scene_duration=True,
        rotate_mode="NATIVE",
    )
    imported_armatures = [obj for obj in bpy.context.selected_objects if obj.type == "ARMATURE"]
    if not imported_armatures:
        raise RuntimeError(f"Blender did not create an armature while importing {bvh_path}")
    source_armature = imported_armatures[0]
    source_armature.name = f"{source_armature.name}_BVH"
    return source_armature


def get_target_armature(name: str) -> bpy.types.Object:
    armature_object = bpy.data.objects.get(name)
    if armature_object is None or armature_object.type != "ARMATURE":
        raise RuntimeError(f"target armature not found or not an armature: {name}")
    return armature_object


def find_bone_by_alias(armature_object: bpy.types.Object, aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in armature_object.pose.bones:
            return alias
    return None


def auto_build_bone_map(
    source_armature: bpy.types.Object,
    target_armature: bpy.types.Object,
) -> dict[str, str]:
    # Resolve a usable source->target mapping by matching canonical humanoid names first.
    mapping: dict[str, str] = {}
    for canonical_name, aliases in CANONICAL_BONE_ALIASES.items():
        source_bone_name = find_bone_by_alias(source_armature, aliases)
        target_bone_name = find_bone_by_alias(target_armature, aliases)
        if source_bone_name is not None and target_bone_name is not None:
            mapping[target_bone_name] = source_bone_name

    if not mapping:
        for bone_name in source_armature.pose.bones.keys():
            if bone_name in target_armature.pose.bones:
                mapping[bone_name] = bone_name
    return mapping


def load_bone_map(
    source_armature: bpy.types.Object,
    target_armature: bpy.types.Object,
    mapping_json_path: Path | None,
) -> dict[str, str]:
    if mapping_json_path is None:
        mapping = auto_build_bone_map(source_armature, target_armature)
        if not mapping:
            raise RuntimeError("failed to auto-map any bones; provide --bone-map-json")
        return mapping

    payload = json.loads(mapping_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"bone map JSON must contain an object: {mapping_json_path}")

    mapping: dict[str, str] = {}
    for target_bone_name, source_bone_name in payload.items():
        if target_bone_name not in target_armature.pose.bones:
            raise ValueError(f"target bone missing on armature {target_armature.name}: {target_bone_name}")
        if source_bone_name not in source_armature.pose.bones:
            raise ValueError(f"source bone missing on armature {source_armature.name}: {source_bone_name}")
        mapping[str(target_bone_name)] = str(source_bone_name)
    return mapping


def clear_retarget_constraints(target_armature: bpy.types.Object) -> None:
    for pose_bone in target_armature.pose.bones:
        for constraint in list(pose_bone.constraints):
            if constraint.name.startswith(COPY_ROTATION_PREFIX) or constraint.name.startswith(COPY_LOCATION_PREFIX):
                pose_bone.constraints.remove(constraint)


def apply_rotation_constraints(
    source_armature: bpy.types.Object,
    target_armature: bpy.types.Object,
    bone_map: dict[str, str],
    *,
    copy_root_location: bool,
) -> None:
    clear_retarget_constraints(target_armature)

    for target_bone_name, source_bone_name in bone_map.items():
        target_bone = target_armature.pose.bones[target_bone_name]
        target_bone.rotation_mode = "QUATERNION"

        # Drive each target pose bone from the imported BVH pose bone, then bake the result.
        rotation_constraint = target_bone.constraints.new(type="COPY_ROTATION")
        rotation_constraint.name = f"{COPY_ROTATION_PREFIX}{target_bone_name}"
        rotation_constraint.target = source_armature
        rotation_constraint.subtarget = source_bone_name
        rotation_constraint.owner_space = "LOCAL"
        rotation_constraint.target_space = "LOCAL"
        if hasattr(rotation_constraint, "mix_mode"):
            rotation_constraint.mix_mode = "REPLACE"

    if copy_root_location:
        root_target_bone = next(
            (target_name for target_name, source_name in bone_map.items() if source_name == "Hips"),
            next(iter(bone_map)),
        )
        root_source_bone = bone_map[root_target_bone]
        root_bone = target_armature.pose.bones[root_target_bone]
        location_constraint = root_bone.constraints.new(type="COPY_LOCATION")
        location_constraint.name = f"{COPY_LOCATION_PREFIX}{root_target_bone}"
        location_constraint.target = source_armature
        location_constraint.subtarget = root_source_bone
        location_constraint.owner_space = "LOCAL"
        location_constraint.target_space = "LOCAL"


def frame_range_for_action(armature_object: bpy.types.Object) -> tuple[int, int]:
    animation_data = armature_object.animation_data
    if animation_data is None or animation_data.action is None:
        raise RuntimeError(f"armature does not have an animation action: {armature_object.name}")
    frame_start, frame_end = animation_data.action.frame_range
    return int(frame_start), int(frame_end)


def bake_constraints_to_action(
    source_armature: bpy.types.Object,
    target_armature: bpy.types.Object,
    bone_map: dict[str, str],
) -> None:
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = target_armature
    target_armature.select_set(True)
    if target_armature.animation_data is None:
        target_armature.animation_data_create()
    if target_armature.animation_data.action is None:
        target_armature.animation_data.action = bpy.data.actions.new(
            name=f"{target_armature.name}_BVH_Retarget",
        )
    bpy.ops.object.mode_set(mode="POSE")

    frame_start, frame_end = frame_range_for_action(source_armature)
    bpy.ops.nla.bake(
        frame_start=frame_start,
        frame_end=frame_end,
        step=1,
        only_selected=False,
        visual_keying=True,
        clear_constraints=True,
        use_current_action=True,
        clean_curves=False,
        bake_types={"POSE"},
    )
    bpy.ops.object.mode_set(mode="OBJECT")


def remove_object(object_name: str) -> None:
    obj = bpy.data.objects.get(object_name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parse_args(argv)

    open_blend_if_needed(args.blend_in)
    target_armature = get_target_armature(args.target_armature)
    source_armature = import_bvh_armature(args.bvh, args.global_scale)
    bone_map = load_bone_map(source_armature, target_armature, args.bone_map_json)

    apply_rotation_constraints(
        source_armature,
        target_armature,
        bone_map,
        copy_root_location=args.copy_root_location,
    )
    bake_constraints_to_action(source_armature, target_armature, bone_map)

    if not args.keep_source_armature:
        remove_object(source_armature.name)

    if args.blend_out is not None:
        args.blend_out.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.blend_out))

    print(f"Imported BVH: {args.bvh}")
    print(f"Target armature: {target_armature.name}")
    print(f"Mapped bones: {len(bone_map)}")
    for target_bone_name, source_bone_name in sorted(bone_map.items()):
        print(f"{target_bone_name} <- {source_bone_name}")


if __name__ == "__main__":
    main()
