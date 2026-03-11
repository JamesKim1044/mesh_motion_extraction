"""Generate a procedural humanoid proxy mesh from a single image.

This module does not reconstruct a true person-specific mesh. It estimates a
commercial-friendly proxy body from silhouette and pose cues, then emits a
deterministic OBJ plus rigging metadata that downstream Blender stages can bind
and retarget reliably.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import warnings

import cv2
import numpy as np

from common import (
    BONE_NAMING_CONVENTION,
    COORDINATE_SYSTEM,
    DEFAULT_BODY_HEIGHT_M,
    DEFAULT_BONE_MAPPING_HINTS,
    JOINT_ORDER,
    JOINT_SCHEMA_NAME,
    PARENT_MAP,
    SCALE_METADATA,
    write_json,
)


LANDMARK_INDEX = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

EPS = 1e-8


@dataclass(slots=True)
class MeshGenerationResult:
    mesh_path: Path
    metadata_path: Path
    estimated_height_m: float


def landmark_to_pixel(landmark, width: int, height: int) -> np.ndarray:
    return np.array([float(landmark.x) * width, float(landmark.y) * height], dtype=np.float64)


def safe_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def clamp_row(mask: np.ndarray, row: float) -> int:
    return max(0, min(mask.shape[0] - 1, int(round(row))))


def mask_row_width(mask: np.ndarray, row: float) -> float | None:
    clamped_row = clamp_row(mask, row)
    columns = np.flatnonzero(mask[clamped_row])
    if columns.size < 2:
        return None
    return float(columns[-1] - columns[0] + 1)


def load_image_with_foreground_mask(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"image not found or unreadable: {image_path}")

    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
    elif raw.shape[2] == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)

    bgr = raw[:, :, :3].astype(np.float32)
    alpha = raw[:, :, 3].astype(np.float32) / 255.0

    alpha_mask = alpha > 0.05
    color_mask = np.any(bgr > 15.0, axis=2)
    if np.mean(alpha_mask) < 0.98 and np.any(alpha_mask):
        foreground_mask = alpha_mask
    elif np.any(color_mask):
        foreground_mask = color_mask
    else:
        foreground_mask = np.ones(raw.shape[:2], dtype=bool)

    composed_bgr = bgr * alpha[:, :, None] + 255.0 * (1.0 - alpha[:, :, None])
    if np.mean(alpha_mask) >= 0.98:
        # Opaque PNGs with black backgrounds benefit from compositing the detected foreground onto white.
        composed_bgr = np.where(foreground_mask[:, :, None], bgr, 255.0)

    image_rgb = cv2.cvtColor(composed_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return image_rgb, foreground_mask


def should_skip_mediapipe(image_rgb: np.ndarray, mask: np.ndarray) -> bool:
    foreground_fraction = float(np.mean(mask))
    if not (0.05 <= foreground_fraction <= 0.85):
        return False

    patch = 24
    corners = [
        image_rgb[:patch, :patch],
        image_rgb[:patch, -patch:],
        image_rgb[-patch:, :patch],
        image_rgb[-patch:, -patch:],
    ]
    corner_means = [corner.reshape(-1, 3).mean(axis=0) for corner in corners]
    max_corner_delta = max(
        float(np.linalg.norm(a - b))
        for index, a in enumerate(corner_means)
        for b in corner_means[index + 1 :]
    )
    average_corner_brightness = float(np.mean(corner_means))
    return max_corner_delta < 12.0 and (
        average_corner_brightness < 24.0 or average_corner_brightness > 231.0
    )


def import_mediapipe():
    if os.environ.get("PROCEDURAL_MESH_DISABLE_MEDIAPIPE") == "1":
        return None
    try:
        import mediapipe as mp  # type: ignore
    except Exception:
        return None
    return mp


def mask_bounds(mask: np.ndarray, width_threshold: float = 1.0) -> tuple[int, int]:
    row_widths = np.sum(mask, axis=1)
    rows = np.flatnonzero(row_widths >= width_threshold)
    if rows.size == 0:
        rows = np.flatnonzero(row_widths > 0)
    if rows.size == 0:
        return 0, mask.shape[0] - 1
    return int(rows[0]), int(rows[-1])


def average_mask_row_width(mask: np.ndarray, center_row: float, radius: int = 6) -> float:
    widths: list[float] = []
    for offset in range(-radius, radius + 1):
        width = mask_row_width(mask, center_row + offset)
        if width is not None:
            widths.append(width)
    return float(np.mean(widths)) if widths else 0.0


def fallback_measurements_from_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    target_height_m: float,
) -> dict[str, object]:
    row_widths = np.sum(mask, axis=1).astype(np.float64)
    active_widths = row_widths[row_widths > 0]
    if active_widths.size == 0:
        mask = np.ones(mask.shape, dtype=bool)
        row_widths = np.sum(mask, axis=1).astype(np.float64)
        active_widths = row_widths[row_widths > 0]

    width_threshold = max(8.0, float(np.percentile(active_widths, 10) * 0.7))
    top_px, bottom_px = mask_bounds(mask, width_threshold=width_threshold)
    height_px = max(float(bottom_px - top_px), 1.0)
    meters_per_pixel = target_height_m / height_px

    shoulder_row = top_px + 0.28 * height_px
    hip_row = top_px + 0.56 * height_px
    head_row = top_px + 0.12 * height_px

    shoulder_width_px = average_mask_row_width(mask, shoulder_row) * 0.52
    hip_width_px = average_mask_row_width(mask, hip_row) * 0.34
    head_width_px = average_mask_row_width(mask, head_row) * 0.48

    if shoulder_width_px <= 1.0:
        shoulder_width_px = height_px * 0.19
    if hip_width_px <= 1.0:
        hip_width_px = height_px * 0.13
    if head_width_px <= 1.0:
        head_width_px = height_px * 0.11

    masked_pixels = image_rgb[mask]
    if masked_pixels.size:
        average_rgb = np.mean(masked_pixels.astype(np.float64), axis=0) / 255.0
    else:
        average_rgb = np.array([0.72, 0.72, 0.72], dtype=np.float64)

    return {
        "image_size": [int(image_rgb.shape[1]), int(image_rgb.shape[0])],
        "meters_per_pixel": float(meters_per_pixel),
        "target_height_m": float(target_height_m),
        "base_color": [float(channel) for channel in average_rgb],
        "measurement_backend": "silhouette_fallback",
        "measurements_m": {
            "height": float(target_height_m),
            "shoulder_width": float(max(shoulder_width_px * meters_per_pixel, target_height_m * 0.18)),
            "hip_width": float(max(hip_width_px * meters_per_pixel, target_height_m * 0.12)),
            "torso_length": float(target_height_m * 0.30),
            "upper_arm_length": float(target_height_m * 0.18),
            "lower_arm_length": float(target_height_m * 0.17),
            "hand_length": float(target_height_m * 0.06),
            "upper_leg_length": float(target_height_m * 0.24),
            "lower_leg_length": float(target_height_m * 0.23),
            "foot_length": float(target_height_m * 0.14),
            "head_radius": float(max(head_width_px * meters_per_pixel * 0.5, target_height_m * 0.085)),
        },
    }


def axis_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis_z = direction / (np.linalg.norm(direction) + EPS)
    seed = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(axis_z, seed))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis_x = np.cross(seed, axis_z)
    axis_x /= np.linalg.norm(axis_x) + EPS
    axis_y = np.cross(axis_z, axis_x)
    axis_y /= np.linalg.norm(axis_y) + EPS
    return axis_x, axis_y, axis_z


def append_mesh(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    vertex_groups: dict[str, list[list[int]]],
    part_vertices: list[tuple[float, float, float]],
    part_faces: list[tuple[int, int, int]],
    group_name: str,
) -> None:
    start = len(vertices)
    vertices.extend(part_vertices)
    faces.extend((a + start, b + start, c + start) for a, b, c in part_faces)
    end = len(vertices)
    vertex_groups.setdefault(group_name, []).append([start, end])


def build_uv_sphere(
    center: np.ndarray,
    radius: np.ndarray | float,
    *,
    segments: int = 18,
    rings: int = 10,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    if isinstance(radius, (float, int)):
        radii = np.array([float(radius), float(radius), float(radius)], dtype=np.float64)
    else:
        radii = np.asarray(radius, dtype=np.float64)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    vertices.append(tuple(center + np.array([0.0, 0.0, radii[2]], dtype=np.float64)))
    for ring in range(1, rings):
        phi = np.pi * ring / rings
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        for segment in range(segments):
            theta = 2.0 * np.pi * segment / segments
            point = np.array(
                [
                    radii[0] * sin_phi * np.cos(theta),
                    radii[1] * sin_phi * np.sin(theta),
                    radii[2] * cos_phi,
                ],
                dtype=np.float64,
            )
            vertices.append(tuple(center + point))
    bottom_index = len(vertices)
    vertices.append(tuple(center + np.array([0.0, 0.0, -radii[2]], dtype=np.float64)))

    first_ring_start = 1
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        faces.append((0, first_ring_start + next_segment, first_ring_start + segment))

    for ring in range(rings - 2):
        ring_start = 1 + ring * segments
        next_ring_start = ring_start + segments
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            current = ring_start + segment
            current_next = ring_start + next_segment
            lower = next_ring_start + segment
            lower_next = next_ring_start + next_segment
            faces.append((current, lower_next, lower))
            faces.append((current, current_next, lower_next))

    last_ring_start = 1 + (rings - 2) * segments
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        faces.append((last_ring_start + segment, last_ring_start + next_segment, bottom_index))

    return vertices, faces


def build_cylinder(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    *,
    segments: int = 16,
    cap_ends: bool = False,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < EPS:
        return [], []

    axis_x, axis_y, axis_z = axis_basis(direction)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    for ring_origin in (start, end):
        for segment in range(segments):
            theta = 2.0 * np.pi * segment / segments
            offset = radius * (np.cos(theta) * axis_x + np.sin(theta) * axis_y)
            vertices.append(tuple(ring_origin + offset))

    for segment in range(segments):
        next_segment = (segment + 1) % segments
        top0 = segment
        top1 = next_segment
        bottom0 = segments + segment
        bottom1 = segments + next_segment
        faces.append((top0, bottom1, bottom0))
        faces.append((top0, top1, bottom1))

    if cap_ends:
        start_center = len(vertices)
        vertices.append(tuple(start))
        end_center = len(vertices)
        vertices.append(tuple(end))
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            faces.append((start_center, segment, next_segment))
            faces.append((end_center, segments + next_segment, segments + segment))

    return vertices, faces


def build_capsule(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    *,
    segments: int = 16,
    rings: int = 8,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    cylinder_vertices, cylinder_faces = build_cylinder(
        start=start,
        end=end,
        radius=radius,
        segments=segments,
        cap_ends=False,
    )
    append_mesh(vertices, faces, {"_": []}, cylinder_vertices, cylinder_faces, "_")
    sphere_a_vertices, sphere_a_faces = build_uv_sphere(start, radius, segments=segments, rings=rings)
    append_mesh(vertices, faces, {"_": []}, sphere_a_vertices, sphere_a_faces, "_")
    sphere_b_vertices, sphere_b_faces = build_uv_sphere(end, radius, segments=segments, rings=rings)
    append_mesh(vertices, faces, {"_": []}, sphere_b_vertices, sphere_b_faces, "_")
    return vertices, faces


def build_box(
    center: np.ndarray,
    extents: np.ndarray,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    half = extents * 0.5
    corners = [
        center + np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=np.float64)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]
    faces = [
        (0, 1, 3),
        (0, 3, 2),
        (4, 6, 7),
        (4, 7, 5),
        (0, 4, 5),
        (0, 5, 1),
        (2, 3, 7),
        (2, 7, 6),
        (0, 2, 6),
        (0, 6, 4),
        (1, 5, 7),
        (1, 7, 3),
    ]
    return [tuple(corner) for corner in corners], faces


def write_obj(
    path: Path,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Procedural humanoid mesh"]
    lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def extract_measurements(image_path: Path, target_height_m: float) -> dict[str, object]:
    image_rgb, foreground_mask = load_image_with_foreground_mask(image_path)
    height, width = image_rgb.shape[:2]

    if should_skip_mediapipe(image_rgb, foreground_mask):
        warnings.warn(
            f"{image_path.name} looks like a segmented illustration or cutout; using silhouette fallback measurements instead.",
            RuntimeWarning,
        )
        return fallback_measurements_from_mask(
            image_rgb=image_rgb,
            mask=foreground_mask,
            target_height_m=target_height_m,
        )

    mp = import_mediapipe()
    if mp is None or not hasattr(mp, "solutions") or not hasattr(mp.solutions, "pose"):
        warnings.warn(
            "MediaPipe Pose is unavailable in this Python environment; using silhouette fallback measurements instead.",
            RuntimeWarning,
        )
        return fallback_measurements_from_mask(
            image_rgb=image_rgb,
            mask=foreground_mask,
            target_height_m=target_height_m,
        )

    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5,
    )
    segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    try:
        pose_result = pose.process(image_rgb)
        segmentation_result = segmenter.process(image_rgb)
    finally:
        pose.close()
        segmenter.close()

    if not pose_result.pose_landmarks:
        warnings.warn(
            f"MediaPipe Pose could not detect a full body in {image_path.name}; using silhouette fallback measurements instead.",
            RuntimeWarning,
        )
        return fallback_measurements_from_mask(
            image_rgb=image_rgb,
            mask=foreground_mask,
            target_height_m=target_height_m,
        )

    landmarks = pose_result.pose_landmarks.landmark
    points = {
        name: landmark_to_pixel(landmarks[index], width=width, height=height)
        for name, index in LANDMARK_INDEX.items()
    }

    mask = segmentation_result.segmentation_mask > 0.35
    mask = np.logical_or(mask, foreground_mask)
    if not np.any(mask):
        mask = foreground_mask if np.any(foreground_mask) else np.ones((height, width), dtype=bool)

    rows = np.flatnonzero(mask.any(axis=1))
    top_px = float(rows[0]) if rows.size else float(min(point[1] for point in points.values()))
    bottom_px = float(rows[-1]) if rows.size else float(max(point[1] for point in points.values()))
    height_px = max(bottom_px - top_px, 1.0)
    meters_per_pixel = target_height_m / height_px

    shoulder_center = 0.5 * (points["left_shoulder"] + points["right_shoulder"])
    hip_center = 0.5 * (points["left_hip"] + points["right_hip"])

    shoulder_row_width = mask_row_width(mask, shoulder_center[1]) or 0.0
    hip_row_width = mask_row_width(mask, hip_center[1]) or 0.0

    shoulder_width_px = max(
        safe_distance(points["left_shoulder"], points["right_shoulder"]),
        shoulder_row_width * 0.65,
    )
    hip_width_px = max(
        safe_distance(points["left_hip"], points["right_hip"]),
        hip_row_width * 0.40,
    )
    torso_length_px = max(safe_distance(shoulder_center, hip_center), height_px * 0.24)

    upper_arm_length_px = 0.5 * (
        safe_distance(points["left_shoulder"], points["left_elbow"])
        + safe_distance(points["right_shoulder"], points["right_elbow"])
    )
    lower_arm_length_px = 0.5 * (
        safe_distance(points["left_elbow"], points["left_wrist"])
        + safe_distance(points["right_elbow"], points["right_wrist"])
    )
    hand_length_px = max(
        0.5
        * (
            safe_distance(points["left_wrist"], points["left_index"])
            + safe_distance(points["right_wrist"], points["right_index"])
        ),
        lower_arm_length_px * 0.28,
    )
    upper_leg_length_px = 0.5 * (
        safe_distance(points["left_hip"], points["left_knee"])
        + safe_distance(points["right_hip"], points["right_knee"])
    )
    lower_leg_length_px = 0.5 * (
        safe_distance(points["left_knee"], points["left_ankle"])
        + safe_distance(points["right_knee"], points["right_ankle"])
    )
    foot_length_px = max(
        0.5
        * (
            safe_distance(points["left_ankle"], points["left_foot_index"])
            + safe_distance(points["right_ankle"], points["right_foot_index"])
        ),
        lower_leg_length_px * 0.38,
    )

    head_radius_m = max(
        meters_per_pixel
        * max(
            safe_distance(points["left_ear"], points["right_ear"]) * 0.33,
            safe_distance(points["nose"], shoulder_center) * 0.36,
        ),
        target_height_m * 0.08,
    )

    masked_pixels = image_rgb[mask]
    if masked_pixels.size:
        average_rgb = np.mean(masked_pixels.astype(np.float64), axis=0) / 255.0
    else:
        average_rgb = np.array([0.72, 0.72, 0.72], dtype=np.float64)

    return {
        "image_size": [int(width), int(height)],
        "meters_per_pixel": float(meters_per_pixel),
        "target_height_m": float(target_height_m),
        "base_color": [float(channel) for channel in average_rgb],
        "measurement_backend": "mediapipe_pose",
        "measurements_m": {
            "height": float(target_height_m),
            "shoulder_width": float(shoulder_width_px * meters_per_pixel),
            "hip_width": float(max(hip_width_px * meters_per_pixel, target_height_m * 0.12)),
            "torso_length": float(max(torso_length_px * meters_per_pixel, target_height_m * 0.22)),
            "upper_arm_length": float(max(upper_arm_length_px * meters_per_pixel, target_height_m * 0.15)),
            "lower_arm_length": float(max(lower_arm_length_px * meters_per_pixel, target_height_m * 0.14)),
            "hand_length": float(max(hand_length_px * meters_per_pixel, target_height_m * 0.05)),
            "upper_leg_length": float(max(upper_leg_length_px * meters_per_pixel, target_height_m * 0.21)),
            "lower_leg_length": float(max(lower_leg_length_px * meters_per_pixel, target_height_m * 0.20)),
            "foot_length": float(max(foot_length_px * meters_per_pixel, target_height_m * 0.10)),
            "head_radius": float(head_radius_m),
        },
    }


def build_canonical_joints(measurements: dict[str, float]) -> dict[str, np.ndarray]:
    shoulder_half = measurements["shoulder_width"] * 0.5
    hip_half = measurements["hip_width"] * 0.5
    foot_thickness = max(measurements["height"] * 0.025, 0.035)
    ankle_height = foot_thickness
    knee_height = ankle_height + measurements["lower_leg_length"]
    hip_height = knee_height + measurements["upper_leg_length"]
    spine_height = hip_height + measurements["torso_length"] * 0.32
    chest_height = hip_height + measurements["torso_length"] * 0.72
    neck_height = hip_height + measurements["torso_length"] * 0.94

    joints = {
        "Hips": np.array([0.0, 0.0, hip_height], dtype=np.float64),
        "Spine": np.array([0.0, 0.0, spine_height], dtype=np.float64),
        "Chest": np.array([0.0, 0.0, chest_height], dtype=np.float64),
        "Neck": np.array([0.0, 0.0, neck_height], dtype=np.float64),
        "Head": np.array([0.0, 0.0, neck_height + measurements["head_radius"] * 1.35], dtype=np.float64),
        "LeftShoulder": np.array([shoulder_half, 0.0, chest_height], dtype=np.float64),
        "LeftElbow": np.array(
            [shoulder_half + measurements["upper_arm_length"], 0.0, chest_height],
            dtype=np.float64,
        ),
        "LeftWrist": np.array(
            [shoulder_half + measurements["upper_arm_length"] + measurements["lower_arm_length"], 0.0, chest_height],
            dtype=np.float64,
        ),
        "LeftHand": np.array(
            [
                shoulder_half
                + measurements["upper_arm_length"]
                + measurements["lower_arm_length"]
                + measurements["hand_length"],
                0.0,
                chest_height,
            ],
            dtype=np.float64,
        ),
        "RightShoulder": np.array([-shoulder_half, 0.0, chest_height], dtype=np.float64),
        "RightElbow": np.array(
            [-shoulder_half - measurements["upper_arm_length"], 0.0, chest_height],
            dtype=np.float64,
        ),
        "RightWrist": np.array(
            [
                -shoulder_half - measurements["upper_arm_length"] - measurements["lower_arm_length"],
                0.0,
                chest_height,
            ],
            dtype=np.float64,
        ),
        "RightHand": np.array(
            [
                -shoulder_half
                - measurements["upper_arm_length"]
                - measurements["lower_arm_length"]
                - measurements["hand_length"],
                0.0,
                chest_height,
            ],
            dtype=np.float64,
        ),
        "LeftHip": np.array([hip_half, 0.0, hip_height], dtype=np.float64),
        "LeftKnee": np.array([hip_half, 0.0, knee_height], dtype=np.float64),
        "LeftAnkle": np.array([hip_half, 0.0, ankle_height], dtype=np.float64),
        "LeftFoot": np.array([hip_half, measurements["foot_length"] * 0.62, foot_thickness * 0.5], dtype=np.float64),
        "RightHip": np.array([-hip_half, 0.0, hip_height], dtype=np.float64),
        "RightKnee": np.array([-hip_half, 0.0, knee_height], dtype=np.float64),
        "RightAnkle": np.array([-hip_half, 0.0, ankle_height], dtype=np.float64),
        "RightFoot": np.array(
            [-hip_half, measurements["foot_length"] * 0.62, foot_thickness * 0.5],
            dtype=np.float64,
        ),
    }
    return joints


def build_humanoid_mesh(
    joints: dict[str, np.ndarray],
    measurements: dict[str, float],
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], dict[str, list[list[int]]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    vertex_groups: dict[str, list[list[int]]] = {}

    torso_radius = max(measurements["shoulder_width"] * 0.16, measurements["height"] * 0.055)
    pelvis_radius = max(measurements["hip_width"] * 0.18, measurements["height"] * 0.06)
    upper_arm_radius = max(measurements["shoulder_width"] * 0.07, measurements["height"] * 0.028)
    lower_arm_radius = upper_arm_radius * 0.82
    upper_leg_radius = max(measurements["hip_width"] * 0.10, measurements["height"] * 0.04)
    lower_leg_radius = upper_leg_radius * 0.82
    hand_radius = lower_arm_radius * 0.85
    foot_box = np.array(
        [
            measurements["foot_length"] * 0.42,
            measurements["foot_length"],
            max(measurements["height"] * 0.035, 0.04),
        ],
        dtype=np.float64,
    )

    mesh_parts = [
        (
            "Hips",
            *build_uv_sphere(
                center=0.5 * (joints["Hips"] + joints["Spine"]),
                radius=np.array([pelvis_radius, pelvis_radius * 0.85, torso_radius], dtype=np.float64),
            ),
        ),
        ("Spine", *build_capsule(joints["Hips"], joints["Spine"], pelvis_radius)),
        ("Chest", *build_capsule(joints["Spine"], joints["Chest"], torso_radius)),
        ("Neck", *build_capsule(joints["Chest"], joints["Neck"], torso_radius * 0.55)),
        (
            "Head",
            *build_uv_sphere(
                center=joints["Head"] + np.array([0.0, 0.0, measurements["head_radius"] * 0.15], dtype=np.float64),
                radius=np.array(
                    [
                        measurements["head_radius"] * 0.82,
                        measurements["head_radius"] * 0.76,
                        measurements["head_radius"],
                    ],
                    dtype=np.float64,
                ),
            ),
        ),
        ("LeftShoulder", *build_capsule(joints["LeftShoulder"], joints["LeftElbow"], upper_arm_radius)),
        ("LeftElbow", *build_capsule(joints["LeftElbow"], joints["LeftWrist"], lower_arm_radius)),
        ("LeftWrist", *build_capsule(joints["LeftWrist"], joints["LeftHand"], hand_radius)),
        ("LeftHand", *build_uv_sphere(joints["LeftHand"], hand_radius)),
        ("RightShoulder", *build_capsule(joints["RightShoulder"], joints["RightElbow"], upper_arm_radius)),
        ("RightElbow", *build_capsule(joints["RightElbow"], joints["RightWrist"], lower_arm_radius)),
        ("RightWrist", *build_capsule(joints["RightWrist"], joints["RightHand"], hand_radius)),
        ("RightHand", *build_uv_sphere(joints["RightHand"], hand_radius)),
        ("LeftHip", *build_capsule(joints["LeftHip"], joints["LeftKnee"], upper_leg_radius)),
        ("LeftKnee", *build_capsule(joints["LeftKnee"], joints["LeftAnkle"], lower_leg_radius)),
        ("LeftAnkle", *build_capsule(joints["LeftAnkle"], joints["LeftFoot"], lower_leg_radius * 0.88)),
        (
            "LeftFoot",
            *build_box(
                center=0.5 * (joints["LeftAnkle"] + joints["LeftFoot"]) + np.array([0.0, foot_box[1] * 0.1, -foot_box[2] * 0.1], dtype=np.float64),
                extents=foot_box,
            ),
        ),
        ("RightHip", *build_capsule(joints["RightHip"], joints["RightKnee"], upper_leg_radius)),
        ("RightKnee", *build_capsule(joints["RightKnee"], joints["RightAnkle"], lower_leg_radius)),
        ("RightAnkle", *build_capsule(joints["RightAnkle"], joints["RightFoot"], lower_leg_radius * 0.88)),
        (
            "RightFoot",
            *build_box(
                center=0.5 * (joints["RightAnkle"] + joints["RightFoot"]) + np.array([0.0, foot_box[1] * 0.1, -foot_box[2] * 0.1], dtype=np.float64),
                extents=foot_box,
            ),
        ),
    ]

    for group_name, part_vertices, part_faces in mesh_parts:
        append_mesh(vertices, faces, vertex_groups, part_vertices, part_faces, group_name)

    return vertices, faces, vertex_groups


def generate_humanoid_mesh_from_image(
    image_path: Path,
    output_obj_path: Path,
    metadata_path: Path,
    *,
    target_height_m: float = DEFAULT_BODY_HEIGHT_M,
) -> MeshGenerationResult:
    """Build a procedural proxy humanoid mesh and metadata contract from one image.

    The generated OBJ is intended for rigging and motion-transfer smoke tests.
    It is not a photoreal or identity-preserving human reconstruction.
    """
    measurements_payload = extract_measurements(image_path=image_path, target_height_m=target_height_m)
    measurements = measurements_payload["measurements_m"]
    assert isinstance(measurements, dict)

    joints = build_canonical_joints(measurements)
    vertices, faces, vertex_groups = build_humanoid_mesh(joints=joints, measurements=measurements)
    write_obj(output_obj_path, vertices=vertices, faces=faces)
    joint_positions = {joint: [float(value) for value in joints[joint]] for joint in JOINT_ORDER}

    metadata_payload = {
        "image_path": str(image_path),
        "mesh_path": str(output_obj_path),
        "estimated_height_m": float(measurements["height"]),
        "base_color": measurements_payload["base_color"],
        "measurement_backend": measurements_payload.get("measurement_backend", "unknown"),
        "joint_schema": {
            "name": JOINT_SCHEMA_NAME,
            "joints": JOINT_ORDER,
            "root": "Hips",
            "bone_naming_convention": BONE_NAMING_CONVENTION,
        },
        "coordinate_system": dict(COORDINATE_SYSTEM),
        "scale": {
            **SCALE_METADATA,
            "target_height_m": float(measurements["height"]),
        },
        "rest_pose": {
            "name": "proxy_t_pose",
            "joint_positions": joint_positions,
        },
        "bone_mapping_hints": dict(DEFAULT_BONE_MAPPING_HINTS),
        "proxy_mesh": True,
        "reconstruction_type": "procedural_proxy",
        "joint_positions": joint_positions,
        "parent_map": PARENT_MAP,
        "vertex_groups": vertex_groups,
        "measurements_m": measurements,
        "image_size": measurements_payload["image_size"],
        "meters_per_pixel": measurements_payload["meters_per_pixel"],
    }
    write_json(metadata_path, metadata_payload)

    return MeshGenerationResult(
        mesh_path=output_obj_path,
        metadata_path=metadata_path,
        estimated_height_m=float(measurements["height"]),
    )
