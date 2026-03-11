from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import bpy
from mathutils import Euler


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the current Blender animation to an H.264 MP4 file.",
    )
    parser.add_argument(
        "--blend-in",
        type=Path,
        default=None,
        help="optional .blend file to open before rendering",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output MP4 path",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["eevee", "cycles"],
        default="eevee",
        help="render engine to use",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="render samples for Eevee or Cycles",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="override scene FPS",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=None,
        help="optional render start frame override",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        default=None,
        help="optional render end frame override",
    )
    return parser.parse_args(argv)


def open_blend_if_needed(blend_path: Path | None) -> None:
    if blend_path is None:
        return
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))


def set_render_engine(engine_name: str, samples: int) -> None:
    scene = bpy.context.scene
    if engine_name == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = max(1, samples)
    else:
        engine_items = scene.render.bl_rna.properties["engine"].enum_items.keys()
        if "BLENDER_EEVEE_NEXT" in engine_items:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        elif "BLENDER_EEVEE" in engine_items:
            scene.render.engine = "BLENDER_EEVEE"
        else:
            raise RuntimeError("Eevee is not available in this Blender build")
        if hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = max(1, samples)


def configure_scene(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    set_render_engine(args.engine, args.samples)
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100

    if args.fps is not None:
        scene.render.fps = args.fps
    if args.frame_start is not None:
        scene.frame_start = args.frame_start
    if args.frame_end is not None:
        scene.frame_end = args.frame_end


def ensure_camera() -> None:
    scene = bpy.context.scene
    if scene.camera is not None:
        return

    bpy.ops.object.camera_add(location=(7.0, -7.0, 5.0))
    camera = bpy.context.active_object
    camera.rotation_euler = Euler((1.05, 0.0, 0.78), "XYZ")
    scene.camera = camera


def ensure_light() -> None:
    existing_lights = [obj for obj in bpy.data.objects if obj.type == "LIGHT"]
    if existing_lights:
        return

    bpy.ops.object.light_add(type="SUN", location=(4.0, -4.0, 8.0))
    light = bpy.context.active_object
    light.data.energy = 3.0


def render_png_sequence(output_path: Path) -> Path:
    sequence_dir = output_path.parent / f"{output_path.stem}_frames"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in sorted(sequence_dir.glob("frame_*.png")):
        frame_path.unlink()
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(sequence_dir / "frame_")
    bpy.ops.render.render(animation=True)
    return sequence_dir


def encode_sequence_with_ffmpeg(sequence_dir: Path, output_path: Path, fps: int) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("system ffmpeg was not found in PATH")

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
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "\n".join(
                [
                    "ffmpeg failed while encoding the image sequence to H.264 MP4",
                    completed.stdout.strip(),
                    completed.stderr.strip(),
                ]
            ).strip()
        )


def render_h264_mp4(output_path: Path) -> None:
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(animation=True)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parse_args(argv)

    open_blend_if_needed(args.blend_in)
    configure_scene(args)
    ensure_camera()
    ensure_light()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    try:
        render_h264_mp4(args.output)
    except (TypeError, RuntimeError):
        print("Movie render unavailable or failed; falling back to PNG sequence + ffmpeg.")
        sequence_dir = render_png_sequence(args.output)
        encode_sequence_with_ffmpeg(sequence_dir, args.output, bpy.context.scene.render.fps)

    print(f"Rendered MP4: {args.output}")
    print(f"Engine: {bpy.context.scene.render.engine}")
    print(
        f"Resolution: {bpy.context.scene.render.resolution_x}x{bpy.context.scene.render.resolution_y}",
    )


if __name__ == "__main__":
    main()
