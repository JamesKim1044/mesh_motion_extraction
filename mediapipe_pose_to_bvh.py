#!/usr/bin/env python3
"""Extract pose from a video with MediaPipe, export BVH, and render an overlay video.

This writes a lightweight BVH with per-joint translation channels so the raw
MediaPipe motion can be exported without a full IK/retarget step. It can also
render the exported skeleton back over the source video into a new MP4.

Install dependencies:
    pip install mediapipe opencv-python numpy

Example:
    python mediapipe_pose_to_bvh.py
    python mediapipe_pose_to_bvh.py --input taekwondo-1.mp4 --output taekwondo-1.bvh
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np


DEFAULT_INPUT = Path(__file__).with_name("taekwondo-1.mp4")
DEFAULT_OUTPUT = DEFAULT_INPUT.with_suffix(".bvh")
DEFAULT_OVERLAY_OUTPUT = DEFAULT_INPUT.with_name(f"{DEFAULT_INPUT.stem}_bvh_overlay.mp4")
EPS = 1e-8

LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

PARENT_MAP: dict[str, str | None] = {
    "Hips": None,
    "Spine": "Hips",
    "Chest": "Spine",
    "Neck": "Chest",
    "Head": "Neck",
    "LeftShoulder": "Chest",
    "LeftElbow": "LeftShoulder",
    "LeftWrist": "LeftElbow",
    "LeftHand": "LeftWrist",
    "RightShoulder": "Chest",
    "RightElbow": "RightShoulder",
    "RightWrist": "RightElbow",
    "RightHand": "RightWrist",
    "LeftHip": "Hips",
    "LeftKnee": "LeftHip",
    "LeftAnkle": "LeftKnee",
    "LeftFoot": "LeftAnkle",
    "RightHip": "Hips",
    "RightKnee": "RightHip",
    "RightAnkle": "RightKnee",
    "RightFoot": "RightAnkle",
}

CHILDREN_MAP: dict[str, list[str]] = {joint: [] for joint in PARENT_MAP}
for joint_name, parent_name in PARENT_MAP.items():
    if parent_name is not None:
        CHILDREN_MAP[parent_name].append(joint_name)

JOINT_ORDER = [
    "Hips",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftElbow",
    "LeftWrist",
    "LeftHand",
    "RightShoulder",
    "RightElbow",
    "RightWrist",
    "RightHand",
    "LeftHip",
    "LeftKnee",
    "LeftAnkle",
    "LeftFoot",
    "RightHip",
    "RightKnee",
    "RightAnkle",
    "RightFoot",
]

OVERLAY_CONNECTIONS = [
    ("Hips", "Spine"),
    ("Spine", "Chest"),
    ("Chest", "Neck"),
    ("Neck", "Head"),
    ("Chest", "LeftShoulder"),
    ("LeftShoulder", "LeftElbow"),
    ("LeftElbow", "LeftWrist"),
    ("LeftWrist", "LeftHand"),
    ("Chest", "RightShoulder"),
    ("RightShoulder", "RightElbow"),
    ("RightElbow", "RightWrist"),
    ("RightWrist", "RightHand"),
    ("Hips", "LeftHip"),
    ("LeftHip", "LeftKnee"),
    ("LeftKnee", "LeftAnkle"),
    ("LeftAnkle", "LeftFoot"),
    ("Hips", "RightHip"),
    ("RightHip", "RightKnee"),
    ("RightKnee", "RightAnkle"),
    ("RightAnkle", "RightFoot"),
]


def import_mediapipe_pose():
    try:
        return importlib.import_module("mediapipe")
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe could not be imported. Install it with `python3 -m pip install mediapipe`."
        ) from exc


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def average(points: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(points, axis=0), axis=0)


def copy_frame(frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in frame.items()}


def as_named_points(landmarks) -> dict[str, np.ndarray]:
    return {
        name: np.array(
            [
                float(landmarks[index].x),
                float(landmarks[index].y),
                float(landmarks[index].z),
            ],
            dtype=np.float64,
        )
        for index, name in enumerate(LANDMARK_NAMES)
    }


def estimate_meters_per_pixel(
    image_points: dict[str, np.ndarray],
    world_points: dict[str, np.ndarray],
    width: int,
    height: int,
    previous_value: float | None,
) -> float:
    pair_candidates = [
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
    ]

    estimates: list[float] = []
    for left_name, right_name in pair_candidates:
        world_dist = float(np.linalg.norm(world_points[left_name] - world_points[right_name]))
        image_delta = image_points[left_name][:2] - image_points[right_name][:2]
        pixel_dist = float(
            np.linalg.norm(
                np.array([image_delta[0] * width, image_delta[1] * height], dtype=np.float64)
            )
        )
        if world_dist > EPS and pixel_dist > EPS:
            estimates.append(world_dist / pixel_dist)

    if estimates:
        return float(np.median(np.asarray(estimates, dtype=np.float64)))
    if previous_value is not None:
        return previous_value
    return 0.0025


def world_to_bvh_axes(point: np.ndarray) -> np.ndarray:
    return np.array([point[0], -point[1], -point[2]], dtype=np.float64)


def build_pose_joints(
    image_points: dict[str, np.ndarray],
    world_points: dict[str, np.ndarray],
    width: int,
    height: int,
    rest_image_root: np.ndarray | None,
    previous_meters_per_pixel: float | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    meters_per_pixel = estimate_meters_per_pixel(
        image_points=image_points,
        world_points=world_points,
        width=width,
        height=height,
        previous_value=previous_meters_per_pixel,
    )

    image_root = midpoint(image_points["left_hip"], image_points["right_hip"])
    if rest_image_root is None:
        root_translation = np.zeros(3, dtype=np.float64)
    else:
        root_translation = np.array(
            [
                (image_root[0] - rest_image_root[0]) * width * meters_per_pixel,
                -(image_root[1] - rest_image_root[1]) * height * meters_per_pixel,
                -(image_root[2] - rest_image_root[2]) * width * meters_per_pixel,
            ],
            dtype=np.float64,
        )

    world_root = midpoint(world_points["left_hip"], world_points["right_hip"])
    global_points = {
        name: world_to_bvh_axes(world_point - world_root) + root_translation
        for name, world_point in world_points.items()
    }

    hips = root_translation
    chest = midpoint(global_points["left_shoulder"], global_points["right_shoulder"])
    spine = hips + (chest - hips) * 0.5

    head_center = average(
        [
            global_points["nose"],
            global_points["left_ear"],
            global_points["right_ear"],
        ]
    )
    neck = chest + (head_center - chest) * 0.35

    left_hand = average(
        [
            global_points["left_wrist"],
            global_points["left_index"],
            global_points["left_pinky"],
            global_points["left_thumb"],
        ]
    )
    right_hand = average(
        [
            global_points["right_wrist"],
            global_points["right_index"],
            global_points["right_pinky"],
            global_points["right_thumb"],
        ]
    )
    left_foot = average(
        [
            global_points["left_heel"],
            global_points["left_foot_index"],
        ]
    )
    right_foot = average(
        [
            global_points["right_heel"],
            global_points["right_foot_index"],
        ]
    )

    joints = {
        "Hips": hips,
        "Spine": spine,
        "Chest": chest,
        "Neck": neck,
        "Head": head_center,
        "LeftShoulder": global_points["left_shoulder"],
        "LeftElbow": global_points["left_elbow"],
        "LeftWrist": global_points["left_wrist"],
        "LeftHand": left_hand,
        "RightShoulder": global_points["right_shoulder"],
        "RightElbow": global_points["right_elbow"],
        "RightWrist": global_points["right_wrist"],
        "RightHand": right_hand,
        "LeftHip": global_points["left_hip"],
        "LeftKnee": global_points["left_knee"],
        "LeftAnkle": global_points["left_ankle"],
        "LeftFoot": left_foot,
        "RightHip": global_points["right_hip"],
        "RightKnee": global_points["right_knee"],
        "RightAnkle": global_points["right_ankle"],
        "RightFoot": right_foot,
    }

    return joints, image_root, meters_per_pixel


def default_end_offset(rest_joints: dict[str, np.ndarray], joint_name: str) -> np.ndarray:
    base_vectors = {
        "Head": rest_joints["Head"] - rest_joints["Neck"],
        "LeftHand": rest_joints["LeftHand"] - rest_joints["LeftWrist"],
        "RightHand": rest_joints["RightHand"] - rest_joints["RightWrist"],
        "LeftFoot": rest_joints["LeftFoot"] - rest_joints["LeftAnkle"],
        "RightFoot": rest_joints["RightFoot"] - rest_joints["RightAnkle"],
    }

    vector = base_vectors.get(joint_name, np.array([0.0, 0.08, 0.0], dtype=np.float64))
    length = float(np.linalg.norm(vector))
    if length <= EPS:
        return np.array([0.0, 0.08, 0.0], dtype=np.float64)
    return vector * max(0.35, 0.06 / length)


def build_rest_offsets(
    rest_joints: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    offsets: dict[str, np.ndarray] = {"Hips": np.zeros(3, dtype=np.float64)}
    end_offsets: dict[str, np.ndarray] = {}

    for joint_name in JOINT_ORDER[1:]:
        parent_name = PARENT_MAP[joint_name]
        if parent_name is None:
            continue
        offsets[joint_name] = rest_joints[joint_name] - rest_joints[parent_name]

    for joint_name in JOINT_ORDER:
        if not CHILDREN_MAP[joint_name]:
            end_offsets[joint_name] = default_end_offset(rest_joints, joint_name)

    return offsets, end_offsets


def append_joint_hierarchy(
    lines: list[str],
    joint_name: str,
    rest_offsets: dict[str, np.ndarray],
    end_offsets: dict[str, np.ndarray],
    scale: float,
    depth: int = 0,
) -> None:
    indent = "  " * depth
    node_type = "ROOT" if PARENT_MAP[joint_name] is None else "JOINT"
    offset = rest_offsets[joint_name] * scale

    lines.append(f"{indent}{node_type} {joint_name}")
    lines.append(f"{indent}{{")
    lines.append(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
    lines.append(
        f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"
    )

    children = CHILDREN_MAP[joint_name]
    if children:
        for child_name in children:
            append_joint_hierarchy(lines, child_name, rest_offsets, end_offsets, scale, depth + 1)
    else:
        end_offset = end_offsets[joint_name] * scale
        lines.append(f"{indent}  End Site")
        lines.append(f"{indent}  {{")
        lines.append(
            f"{indent}    OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}"
        )
        lines.append(f"{indent}  }}")

    lines.append(f"{indent}}}")


def frame_channels(
    frame_joints: dict[str, np.ndarray],
    rest_joints: dict[str, np.ndarray],
    rest_offsets: dict[str, np.ndarray],
    scale: float,
) -> list[float]:
    values: list[float] = []

    for joint_name in JOINT_ORDER:
        parent_name = PARENT_MAP[joint_name]
        if parent_name is None:
            local_position = frame_joints[joint_name] - rest_joints[joint_name]
        else:
            local_position = frame_joints[joint_name] - frame_joints[parent_name]
            local_position = local_position - rest_offsets[joint_name]

        values.extend((local_position * scale).tolist())
        values.extend([0.0, 0.0, 0.0])

    return values


def point_to_pixel(
    point: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int]:
    x = int(round(float(point[0]) * width))
    y = int(round(float(point[1]) * height))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def build_overlay_joints(
    image_points: dict[str, np.ndarray],
    width: int,
    height: int,
) -> dict[str, tuple[int, int]]:
    hips = midpoint(image_points["left_hip"], image_points["right_hip"])
    chest = midpoint(image_points["left_shoulder"], image_points["right_shoulder"])
    spine = hips + (chest - hips) * 0.5
    head_center = average(
        [
            image_points["nose"],
            image_points["left_ear"],
            image_points["right_ear"],
        ]
    )
    neck = chest + (head_center - chest) * 0.35
    left_hand = average(
        [
            image_points["left_wrist"],
            image_points["left_index"],
            image_points["left_pinky"],
            image_points["left_thumb"],
        ]
    )
    right_hand = average(
        [
            image_points["right_wrist"],
            image_points["right_index"],
            image_points["right_pinky"],
            image_points["right_thumb"],
        ]
    )
    left_foot = average(
        [
            image_points["left_heel"],
            image_points["left_foot_index"],
        ]
    )
    right_foot = average(
        [
            image_points["right_heel"],
            image_points["right_foot_index"],
        ]
    )

    projected: dict[str, tuple[int, int]] = {}
    projected["Hips"] = point_to_pixel(hips, width, height)
    projected["Spine"] = point_to_pixel(spine, width, height)
    projected["Chest"] = point_to_pixel(chest, width, height)
    projected["Neck"] = point_to_pixel(neck, width, height)
    projected["Head"] = point_to_pixel(head_center, width, height)
    projected["LeftShoulder"] = point_to_pixel(image_points["left_shoulder"], width, height)
    projected["LeftElbow"] = point_to_pixel(image_points["left_elbow"], width, height)
    projected["LeftWrist"] = point_to_pixel(image_points["left_wrist"], width, height)
    projected["LeftHand"] = point_to_pixel(left_hand, width, height)
    projected["RightShoulder"] = point_to_pixel(image_points["right_shoulder"], width, height)
    projected["RightElbow"] = point_to_pixel(image_points["right_elbow"], width, height)
    projected["RightWrist"] = point_to_pixel(image_points["right_wrist"], width, height)
    projected["RightHand"] = point_to_pixel(right_hand, width, height)
    projected["LeftHip"] = point_to_pixel(image_points["left_hip"], width, height)
    projected["LeftKnee"] = point_to_pixel(image_points["left_knee"], width, height)
    projected["LeftAnkle"] = point_to_pixel(image_points["left_ankle"], width, height)
    projected["LeftFoot"] = point_to_pixel(left_foot, width, height)
    projected["RightHip"] = point_to_pixel(image_points["right_hip"], width, height)
    projected["RightKnee"] = point_to_pixel(image_points["right_knee"], width, height)
    projected["RightAnkle"] = point_to_pixel(image_points["right_ankle"], width, height)
    projected["RightFoot"] = point_to_pixel(right_foot, width, height)
    return projected


def draw_overlay(
    frame_bgr: np.ndarray,
    joint_pixels: dict[str, tuple[int, int]],
    line_thickness: int,
    joint_radius: int,
) -> np.ndarray:
    overlay = frame_bgr.copy()

    for parent_name, child_name in OVERLAY_CONNECTIONS:
        if parent_name not in joint_pixels or child_name not in joint_pixels:
            continue
        color = (80, 220, 255)
        if parent_name.startswith("Left") or child_name.startswith("Left"):
            color = (80, 160, 255)
        elif parent_name.startswith("Right") or child_name.startswith("Right"):
            color = (255, 160, 80)
        cv2.line(overlay, joint_pixels[parent_name], joint_pixels[child_name], color, line_thickness)

    for joint_name, point in joint_pixels.items():
        color = (60, 240, 120)
        if joint_name.startswith("Left"):
            color = (60, 140, 255)
        elif joint_name.startswith("Right"):
            color = (255, 140, 60)
        cv2.circle(overlay, point, joint_radius, color, -1)

    return cv2.addWeighted(overlay, 0.8, frame_bgr, 0.2, 0.0)


def remux_overlay_audio(source_video: Path, overlay_video: Path) -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None or not overlay_video.exists():
        return False

    temp_output = overlay_video.with_name(f"{overlay_video.stem}.audio_tmp{overlay_video.suffix}")
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(overlay_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(temp_output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        if temp_output.exists():
            temp_output.unlink()
        return False

    temp_output.replace(overlay_video)
    return True


def write_bvh(
    output_path: Path,
    frames: list[dict[str, np.ndarray]],
    fps: float,
    scale: float,
) -> None:
    if not frames:
        raise RuntimeError("no pose frames available for BVH export")

    rest_joints = frames[0]
    rest_offsets, end_offsets = build_rest_offsets(rest_joints)

    lines = ["HIERARCHY"]
    append_joint_hierarchy(lines, "Hips", rest_offsets, end_offsets, scale)
    lines.append("MOTION")
    lines.append(f"Frames: {len(frames)}")
    lines.append(f"Frame Time: {1.0 / fps:.8f}")

    for frame in frames:
        values = frame_channels(frame, rest_joints, rest_offsets, scale)
        lines.append(" ".join(f"{value:.6f}" for value in values))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_frames(
    video_path: Path,
    frame_step: int,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    overlay_output: Path | None,
    overlay_line_thickness: int,
    overlay_joint_radius: int,
) -> tuple[list[dict[str, np.ndarray]], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    mp = import_mediapipe_pose()
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frames: list[dict[str, np.ndarray]] = []
    previous_frame: dict[str, np.ndarray] | None = None
    previous_overlay_joints: dict[str, tuple[int, int]] | None = None
    rest_image_root: np.ndarray | None = None
    meters_per_pixel: float | None = None
    frame_index = 0
    overlay_writer: cv2.VideoWriter | None = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            height, width = frame_bgr.shape[:2]
            if overlay_output is not None and overlay_writer is None:
                overlay_output.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                overlay_writer = cv2.VideoWriter(
                    str(overlay_output),
                    fourcc,
                    fps / float(frame_step),
                    (width, height),
                )
                if not overlay_writer.isOpened():
                    raise RuntimeError(f"failed to open video writer: {overlay_output}")

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if not results.pose_landmarks or not results.pose_world_landmarks:
                if previous_frame is not None:
                    frames.append(copy_frame(previous_frame))
                    if overlay_writer is not None and previous_overlay_joints is not None:
                        overlay_frame = draw_overlay(
                            frame_bgr=frame_bgr,
                            joint_pixels=previous_overlay_joints,
                            line_thickness=overlay_line_thickness,
                            joint_radius=overlay_joint_radius,
                        )
                        overlay_writer.write(overlay_frame)
                    elif overlay_writer is not None:
                        overlay_writer.write(frame_bgr)
                elif overlay_writer is not None:
                    overlay_writer.write(frame_bgr)
                frame_index += 1
                continue

            image_points = as_named_points(results.pose_landmarks.landmark)
            world_points = as_named_points(results.pose_world_landmarks.landmark)
            overlay_joints = build_overlay_joints(
                image_points=image_points,
                width=width,
                height=height,
            )
            joints, image_root, meters_per_pixel = build_pose_joints(
                image_points=image_points,
                world_points=world_points,
                width=width,
                height=height,
                rest_image_root=rest_image_root,
                previous_meters_per_pixel=meters_per_pixel,
            )

            if rest_image_root is None:
                rest_image_root = image_root.copy()

            previous_frame = joints
            previous_overlay_joints = dict(overlay_joints)
            frames.append(copy_frame(joints))
            if overlay_writer is not None:
                overlay_frame = draw_overlay(
                    frame_bgr=frame_bgr,
                    joint_pixels=overlay_joints,
                    line_thickness=overlay_line_thickness,
                    joint_radius=overlay_joint_radius,
                )
                overlay_writer.write(overlay_frame)
            frame_index += 1
    finally:
        pose.close()
        cap.release()
        if overlay_writer is not None:
            overlay_writer.release()

    if not frames:
        raise RuntimeError("MediaPipe did not produce any pose frames")

    return frames, fps / float(frame_step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="input video path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output BVH path")
    parser.add_argument(
        "--overlay-output",
        type=Path,
        default=DEFAULT_OVERLAY_OUTPUT,
        help="output video path for source video with BVH skeleton overlay",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="BVH unit scale. 100.0 converts MediaPipe meters to centimeters.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="process every Nth frame for faster export",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe pose model complexity",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe detection confidence threshold",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe tracking confidence threshold",
    )
    parser.add_argument(
        "--overlay-line-thickness",
        type=int,
        default=3,
        help="line thickness for overlay skeleton rendering",
    )
    parser.add_argument(
        "--overlay-joint-radius",
        type=int,
        default=4,
        help="joint radius for overlay skeleton rendering",
    )
    parser.add_argument(
        "--skip-audio-remux",
        action="store_true",
        help="leave the overlay video as video-only instead of copying source audio with ffmpeg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    if not args.input.exists():
        raise FileNotFoundError(f"input video not found: {args.input}")

    frames, fps = extract_frames(
        video_path=args.input,
        frame_step=args.frame_step,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        overlay_output=args.overlay_output,
        overlay_line_thickness=args.overlay_line_thickness,
        overlay_joint_radius=args.overlay_joint_radius,
    )
    write_bvh(output_path=args.output, frames=frames, fps=fps, scale=args.scale)
    print(f"wrote {len(frames)} frames to {args.output}")
    if args.overlay_output is not None:
        if not args.skip_audio_remux:
            if remux_overlay_audio(source_video=args.input, overlay_video=args.overlay_output):
                print(f"wrote overlay video with audio to {args.overlay_output}")
            else:
                print(f"wrote overlay video to {args.overlay_output}")
                print("audio remux skipped because ffmpeg was unavailable or remux failed")
        else:
            print(f"wrote overlay video to {args.overlay_output}")


if __name__ == "__main__":
    main()
