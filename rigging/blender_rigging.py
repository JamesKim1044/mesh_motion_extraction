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
    ARMATURE_DATA_NAME,
    ARMATURE_OBJECT_NAME,
    BONE_NAMING_CONVENTION,
    CHILDREN_MAP,
    JOINT_ORDER,
    LEAF_DIRECTIONS,
    MESH_OBJECT_NAME,
    PARENT_MAP,
    JOINT_SCHEMA_NAME,
    read_json,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a rigged Blender scene from a procedural OBJ.")
    parser.add_argument("--mesh-obj", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--blend-out", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args(argv)


def require_payload_keys(payload: dict[str, object], required_keys: list[str]) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise RuntimeError(f"metadata is missing required keys: {', '.join(missing)}")


def load_joint_positions(payload: dict[str, object]) -> dict[str, Vector]:
    joint_positions_raw = payload.get("joint_positions")
    if not isinstance(joint_positions_raw, dict):
        raise RuntimeError("metadata joint_positions must be an object keyed by schema joint names")

    joint_positions: dict[str, Vector] = {}
    missing = [joint_name for joint_name in JOINT_ORDER if joint_name not in joint_positions_raw]
    if missing:
        raise RuntimeError(f"metadata joint_positions is missing schema joints: {', '.join(missing)}")

    for joint_name in JOINT_ORDER:
        raw_value = joint_positions_raw[joint_name]
        if not isinstance(raw_value, list) or len(raw_value) != 3:
            raise RuntimeError(f"metadata joint_positions.{joint_name} must be a 3-element list")
        joint_positions[joint_name] = Vector((float(raw_value[0]), float(raw_value[1]), float(raw_value[2])))
    return joint_positions


def write_report(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_set_bsdf_input(material: bpy.types.Material, input_name: str, value: object) -> None:
    bsdf = material.node_tree.nodes.get("Principled BSDF") if material.node_tree is not None else None
    if bsdf is None:
        return
    input_socket = bsdf.inputs.get(input_name)
    if input_socket is not None:
        input_socket.default_value = value


def bone_tail_for_joint(joint_name: str, joint_positions: dict[str, Vector]) -> Vector:
    children = CHILDREN_MAP[joint_name]
    if children:
        average = Vector((0.0, 0.0, 0.0))
        for child_name in children:
            average += joint_positions[child_name]
        tail = average / len(children)
    else:
        direction = Vector(LEAF_DIRECTIONS[joint_name])
        tail = joint_positions[joint_name] + direction
    if (tail - joint_positions[joint_name]).length < 1e-4:
        tail = joint_positions[joint_name] + Vector((0.0, 0.05, 0.05))
    return tail


def import_mesh(mesh_path: Path) -> bpy.types.Object:
    if not mesh_path.exists():
        raise FileNotFoundError(f"mesh OBJ not found: {mesh_path}")
    if not hasattr(bpy.ops.wm, "obj_import"):
        raise RuntimeError("this Blender build does not expose bpy.ops.wm.obj_import for OBJ loading")
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.obj_import(filepath=str(mesh_path))
    imported_meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not imported_meshes:
        raise RuntimeError(f"no mesh object imported from {mesh_path}")
    mesh_object = imported_meshes[0]
    mesh_object.name = MESH_OBJECT_NAME
    bpy.context.view_layer.objects.active = mesh_object
    bpy.ops.object.shade_smooth()
    return mesh_object


def normalize_mesh_scale(mesh_object: bpy.types.Object, target_height: float) -> float:
    if target_height <= 0:
        raise RuntimeError(f"estimated_height_m must be > 0, got {target_height}")

    bpy.context.view_layer.objects.active = mesh_object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bbox_points = [mesh_object.matrix_world @ Vector(corner) for corner in mesh_object.bound_box]
    min_z = min(point.z for point in bbox_points)
    max_z = max(point.z for point in bbox_points)
    current_height = max_z - min_z
    if current_height <= 1e-6:
        raise RuntimeError("imported mesh bounding box height is zero; cannot normalize mesh scale")

    scale_factor = target_height / current_height
    mesh_object.scale = (
        mesh_object.scale.x * scale_factor,
        mesh_object.scale.y * scale_factor,
        mesh_object.scale.z * scale_factor,
    )
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bbox_points = [mesh_object.matrix_world @ Vector(corner) for corner in mesh_object.bound_box]
    mesh_object.location.z -= min(point.z for point in bbox_points)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    return scale_factor


def create_armature(joint_positions: dict[str, Vector]) -> bpy.types.Object:
    armature_data = bpy.data.armatures.new(ARMATURE_DATA_NAME)
    armature_object = bpy.data.objects.new(ARMATURE_OBJECT_NAME, armature_data)
    bpy.context.scene.collection.objects.link(armature_object)
    bpy.context.view_layer.objects.active = armature_object
    armature_object.select_set(True)

    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature_data.edit_bones
    created = {}
    for joint_name in JOINT_ORDER:
        bone = edit_bones.new(joint_name)
        bone.head = joint_positions[joint_name]
        bone.tail = bone_tail_for_joint(joint_name, joint_positions)
        bone.roll = 0.0
        created[joint_name] = bone

    for joint_name in JOINT_ORDER:
        parent_name = PARENT_MAP[joint_name]
        if parent_name is None:
            continue
        created[joint_name].parent = created[parent_name]
        created[joint_name].use_connect = False
        created[joint_name].roll = 0.0

    bpy.ops.object.mode_set(mode="OBJECT")
    armature_object.data.display_type = "STICK"
    armature_object.show_in_front = True
    return armature_object


def validate_vertex_groups(
    vertex_groups: object,
    vertex_count: int,
) -> tuple[dict[str, list[list[int]]], list[str]]:
    if not isinstance(vertex_groups, dict):
        raise RuntimeError("metadata vertex_groups must be an object")

    cleaned: dict[str, list[list[int]]] = {}
    warnings: list[str] = []
    for joint_name in JOINT_ORDER:
        ranges = vertex_groups.get(joint_name)
        if not isinstance(ranges, list) or not ranges:
            warnings.append(f"vertex group missing or empty for {joint_name}")
            continue

        valid_ranges: list[list[int]] = []
        for raw_range in ranges:
            if not isinstance(raw_range, list) or len(raw_range) != 2:
                raise RuntimeError(f"vertex_groups.{joint_name} entries must be [start, end] ranges")
            start, end = int(raw_range[0]), int(raw_range[1])
            if not (0 <= start < end <= vertex_count):
                raise RuntimeError(
                    f"vertex_groups.{joint_name} has invalid range [{start}, {end}) for vertex count {vertex_count}"
                )
            valid_ranges.append([start, end])
        cleaned[joint_name] = valid_ranges

    if not cleaned:
        raise RuntimeError("metadata vertex_groups did not contain any valid schema groups")
    return cleaned, warnings


def assign_vertex_groups(
    mesh_object: bpy.types.Object,
    armature_object: bpy.types.Object,
    vertex_groups: dict[str, list[list[int]]],
) -> None:
    for bone_name, ranges in vertex_groups.items():
        group = mesh_object.vertex_groups.get(bone_name)
        if group is None:
            group = mesh_object.vertex_groups.new(name=bone_name)
        for start, end in ranges:
            group.add(list(range(start, end)), 1.0, "REPLACE")

    modifier = mesh_object.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = armature_object
    modifier.use_vertex_groups = True
    mesh_object.parent = armature_object


def apply_material(mesh_object: bpy.types.Object, base_color: list[float]) -> None:
    material = bpy.data.materials.new(name="BodyMaterial")
    material.use_nodes = True
    safe_set_bsdf_input(material, "Base Color", (*base_color, 1.0))
    safe_set_bsdf_input(material, "Roughness", 0.65)
    safe_set_bsdf_input(material, "Specular IOR Level", 0.2)
    safe_set_bsdf_input(material, "Specular", 0.2)
    if mesh_object.data.materials:
        mesh_object.data.materials[0] = material
    else:
        mesh_object.data.materials.append(material)


def main() -> None:
    args = parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    payload = read_json(args.metadata_json)
    require_payload_keys(
        payload,
        [
            "joint_positions",
            "vertex_groups",
            "estimated_height_m",
            "base_color",
            "joint_schema",
            "coordinate_system",
        ],
    )
    joint_schema = payload.get("joint_schema")
    if not isinstance(joint_schema, dict) or joint_schema.get("name") != JOINT_SCHEMA_NAME:
        raise RuntimeError(
            "metadata joint_schema is missing or unexpected; regenerate generated_human.json before rigging"
        )
    joint_positions = load_joint_positions(payload)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0

    mesh_object = import_mesh(args.mesh_obj)
    scale_factor = normalize_mesh_scale(mesh_object, float(payload["estimated_height_m"]))
    cleaned_vertex_groups, warnings = validate_vertex_groups(
        payload["vertex_groups"],
        vertex_count=len(mesh_object.data.vertices),
    )
    apply_material(mesh_object, payload.get("base_color", [0.7, 0.7, 0.7]))
    armature_object = create_armature(joint_positions)
    assign_vertex_groups(mesh_object, armature_object, cleaned_vertex_groups)

    report = {
        "status": "ok",
        "mesh_obj": str(args.mesh_obj),
        "metadata_json": str(args.metadata_json),
        "blend_out": str(args.blend_out),
        "bone_naming_convention": BONE_NAMING_CONVENTION,
        "armature_name": armature_object.name,
        "mesh_object_name": mesh_object.name,
        "bone_count": len(armature_object.data.bones),
        "vertex_count": len(mesh_object.data.vertices),
        "estimated_height_m": float(payload["estimated_height_m"]),
        "scale_factor": float(scale_factor),
        "warnings": warnings,
    }
    write_report(args.report_json, report)

    args.blend_out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.blend_out))


if __name__ == "__main__":
    main()
