from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import bpy
from mathutils import Vector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import ARMATURE_OBJECT_NAME, read_json  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the retargeted animation to MP4.")
    parser.add_argument("--blend-in", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--resolution-x", type=int, default=1920)
    parser.add_argument("--resolution-y", type=int, default=1080)
    parser.add_argument("--engine", choices=["auto", "eevee", "cycles"], default="auto")
    parser.add_argument("--samples", type=int, default=64)
    return parser.parse_args(argv)


def delete_object_if_present(object_name: str) -> None:
    existing = bpy.data.objects.get(object_name)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)


def require_payload_keys(payload: dict[str, object], required_keys: list[str]) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise RuntimeError(f"render metadata is missing required keys: {', '.join(missing)}")


def safe_set_bsdf_input(material: bpy.types.Material, input_name: str, value: object) -> None:
    bsdf = material.node_tree.nodes.get("Principled BSDF") if material.node_tree is not None else None
    if bsdf is None:
        return
    input_socket = bsdf.inputs.get(input_name)
    if input_socket is not None:
        input_socket.default_value = value


def configure_world() -> None:
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs[0].default_value = (0.985, 0.985, 0.99, 1.0)
        background.inputs[1].default_value = 0.8


def ensure_ground_plane() -> None:
    delete_object_if_present("Ground")
    bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "Ground"
    material = bpy.data.materials.new(name="GroundMaterial")
    material.use_nodes = True
    safe_set_bsdf_input(material, "Base Color", (0.92, 0.91, 0.88, 1.0))
    safe_set_bsdf_input(material, "Roughness", 0.9)
    plane.data.materials.clear()
    plane.data.materials.append(material)


def ensure_lighting(height: float) -> None:
    delete_object_if_present("KeySun")
    delete_object_if_present("FillArea")

    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 6.0 * height))
    sun = bpy.context.active_object
    sun.name = "KeySun"
    sun.data.energy = 2.2

    bpy.ops.object.light_add(type="AREA", location=(2.4 * height, -3.0 * height, 2.8 * height))
    area = bpy.context.active_object
    area.name = "FillArea"
    area.data.energy = 2500.0
    area.data.shape = "RECTANGLE"
    area.data.size = 3.0 * height
    area.data.size_y = 2.0 * height
    area.rotation_euler = (1.0, 0.0, 0.75)


def ensure_track_target(armature_object: bpy.types.Object, hips_height: float) -> bpy.types.Object:
    delete_object_if_present("CameraTarget")
    target = bpy.data.objects.new("CameraTarget", None)
    target.empty_display_type = "PLAIN_AXES"
    target.empty_display_size = 0.15
    bpy.context.scene.collection.objects.link(target)
    target.location = Vector((0.0, 0.0, hips_height))

    constraint = target.constraints.new(type="COPY_LOCATION")
    constraint.target = armature_object
    constraint.use_offset = True
    constraint.use_x = True
    constraint.use_y = True
    constraint.use_z = True
    return target


def ensure_camera(track_target: bpy.types.Object, height: float) -> None:
    delete_object_if_present("RenderCamera")
    bpy.ops.object.camera_add(location=(2.6 * height, -5.4 * height, 1.7 * height))
    camera = bpy.context.active_object
    camera.name = "RenderCamera"
    camera.data.lens = 35

    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.target = track_target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    bpy.context.scene.camera = camera


def has_animation(armature_object: bpy.types.Object) -> bool:
    animation_data = armature_object.animation_data
    if animation_data is None:
        return False
    if animation_data.action is not None:
        return True
    if animation_data.nla_tracks:
        return any(track.strips for track in animation_data.nla_tracks)
    return False


def pick_render_engine(choice: str) -> str:
    engine_items = bpy.context.scene.render.bl_rna.properties["engine"].enum_items.keys()
    if choice == "cycles":
        if "CYCLES" not in engine_items:
            raise RuntimeError("Cycles is not available in this Blender build")
        return "CYCLES"
    if choice == "eevee":
        if "BLENDER_EEVEE_NEXT" in engine_items:
            return "BLENDER_EEVEE_NEXT"
        if "BLENDER_EEVEE" in engine_items:
            return "BLENDER_EEVEE"
        raise RuntimeError("Eevee is not available in this Blender build")
    if "CYCLES" in engine_items:
        return "CYCLES"
    if "BLENDER_EEVEE_NEXT" in engine_items:
        return "BLENDER_EEVEE_NEXT"
    if "BLENDER_EEVEE" in engine_items:
        return "BLENDER_EEVEE"
    return "CYCLES"


def configure_render_settings(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.engine = pick_render_engine(args.engine)
    scene.render.resolution_x = args.resolution_x
    scene.render.resolution_y = args.resolution_y
    scene.render.resolution_percentage = 100
    scene.render.fps = max(scene.render.fps, 1)
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = max(1, int(args.samples))


def render_png_sequence(output_video: Path) -> Path:
    sequence_dir = output_video.parent / f"{output_video.stem}_frames"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in sorted(sequence_dir.glob("frame_*.png")):
        frame_path.unlink()
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.filepath = str(sequence_dir / "frame_")
    bpy.ops.render.render(animation=True)
    return sequence_dir


def encode_sequence_with_ffmpeg(sequence_dir: Path, output_video: Path, fps: int) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "Blender was built without FFMPEG movie output and system ffmpeg was not found in PATH"
        )
    command = [
        ffmpeg_path,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(sequence_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "\n".join(
                [
                    "ffmpeg failed while encoding the rendered PNG sequence",
                    completed.stdout.strip(),
                    completed.stderr.strip(),
                ]
            ).strip()
        )


def try_movie_render(output_video: Path) -> bool:
    scene = bpy.context.scene
    try:
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        scene.render.filepath = str(output_video)
        bpy.ops.render.render(animation=True)
    except (TypeError, RuntimeError):
        return False
    return output_video.exists() and output_video.stat().st_size > 0


def main() -> None:
    args = parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    payload = read_json(args.metadata_json)
    require_payload_keys(payload, ["estimated_height_m", "joint_positions"])
    height = float(payload["estimated_height_m"])
    hips_height = float(payload["joint_positions"]["Hips"][2])

    bpy.ops.wm.open_mainfile(filepath=str(args.blend_in))
    armature_object = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    if armature_object is None or armature_object.type != "ARMATURE":
        raise RuntimeError(f"animated blend file does not contain an {ARMATURE_OBJECT_NAME} armature")
    if not has_animation(armature_object):
        raise RuntimeError("animated blend file does not contain baked animation on the target armature")

    scene = bpy.context.scene
    if scene.frame_end <= scene.frame_start:
        raise RuntimeError("scene frame range is invalid for animation rendering")

    configure_world()
    ensure_ground_plane()
    ensure_lighting(height)
    target = ensure_track_target(armature_object, hips_height)
    ensure_camera(target, height)
    configure_render_settings(args)
    print(f"Render engine: {scene.render.engine}")
    print(f"Render resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    if args.output_video.exists():
        args.output_video.unlink()
    if not try_movie_render(args.output_video):
        print("Movie render unavailable or failed; falling back to PNG sequence + ffmpeg.")
        sequence_dir = render_png_sequence(args.output_video)
        encode_sequence_with_ffmpeg(
            sequence_dir=sequence_dir,
            output_video=args.output_video,
            fps=scene.render.fps,
        )
    print(f"Rendered video: {args.output_video}")


if __name__ == "__main__":
    main()
