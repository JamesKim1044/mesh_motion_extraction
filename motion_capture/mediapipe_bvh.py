from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import JOINT_ORDER
from mediapipe_pose_to_bvh import (
    as_named_points,
    build_overlay_joints,
    build_pose_joints,
    draw_overlay,
    remux_overlay_audio,
    write_bvh,
)

JOINT_INDEX = {joint_name: index for index, joint_name in enumerate(JOINT_ORDER)}
JOINT_CONFIDENCE_LANDMARKS = {
    "Hips": ["left_hip", "right_hip"],
    "Spine": ["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
    "Chest": ["left_shoulder", "right_shoulder"],
    "Neck": ["left_shoulder", "right_shoulder", "nose"],
    "Head": ["nose", "left_ear", "right_ear"],
    "LeftShoulder": ["left_shoulder"],
    "LeftElbow": ["left_elbow"],
    "LeftWrist": ["left_wrist"],
    "LeftHand": ["left_wrist", "left_index", "left_pinky", "left_thumb"],
    "RightShoulder": ["right_shoulder"],
    "RightElbow": ["right_elbow"],
    "RightWrist": ["right_wrist"],
    "RightHand": ["right_wrist", "right_index", "right_pinky", "right_thumb"],
    "LeftHip": ["left_hip"],
    "LeftKnee": ["left_knee"],
    "LeftAnkle": ["left_ankle"],
    "LeftFoot": ["left_ankle", "left_heel", "left_foot_index"],
    "RightHip": ["right_hip"],
    "RightKnee": ["right_knee"],
    "RightAnkle": ["right_ankle"],
    "RightFoot": ["right_ankle", "right_heel", "right_foot_index"],
}
JOINT_FALLBACKS = {
    "Head": "Neck",
    "LeftHand": "LeftWrist",
    "RightHand": "RightWrist",
    "LeftFoot": "LeftAnkle",
    "RightFoot": "RightAnkle",
}


@dataclass(slots=True)
class MotionCaptureResult:
    video_path: Path
    bvh_path: Path
    overlay_path: Path | None
    fps: float
    frame_count: int
    warnings: list[str] = field(default_factory=list)
    foot_contact_frames: dict[str, int] = field(default_factory=dict)


def import_mediapipe_pose() -> object:
    try:
        return importlib.import_module("mediapipe")
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe could not be imported for motion extraction. "
            "Install it in the active environment with `python3 -m pip install mediapipe`."
        ) from exc


def _moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.shape[0] <= 2:
        return values.copy()
    radius = max(1, int(window_size) // 2)
    kernel = np.arange(1, radius + 2, dtype=np.float64)
    kernel = np.concatenate([kernel, kernel[-2::-1]])
    kernel /= np.sum(kernel)
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.stack(
        [
            np.convolve(padded[:, axis], kernel, mode="valid")
            for axis in range(values.shape[1])
        ],
        axis=1,
    )


def _interpolate_nan_series(values: np.ndarray) -> tuple[np.ndarray, bool]:
    frame_indices = np.arange(values.shape[0], dtype=np.float64)
    valid = np.isfinite(values)
    if np.count_nonzero(valid) == 0:
        return np.zeros_like(values), True
    if np.count_nonzero(valid) == values.shape[0]:
        return values.copy(), False
    interpolated = values.copy()
    interpolated[~valid] = np.interp(frame_indices[~valid], frame_indices[valid], values[valid])
    return interpolated, True


def interpolate_missing_landmarks(positions: np.ndarray) -> tuple[np.ndarray, list[str]]:
    interpolated = positions.copy()
    warnings: list[str] = []
    for joint_index, joint_name in enumerate(JOINT_ORDER):
        joint_values = interpolated[:, joint_index, :]
        if not np.isfinite(joint_values).any():
            fallback_joint_name = JOINT_FALLBACKS.get(joint_name)
            if fallback_joint_name is not None:
                fallback_joint_index = JOINT_INDEX[fallback_joint_name]
                interpolated[:, joint_index, :] = interpolated[:, fallback_joint_index, :]
                warnings.append(
                    f"{joint_name} was missing in all frames; copied positions from {fallback_joint_name}"
                )
                continue
        joint_interpolated = False
        for axis in range(3):
            series, changed = _interpolate_nan_series(interpolated[:, joint_index, axis])
            interpolated[:, joint_index, axis] = series
            joint_interpolated = joint_interpolated or changed
        if joint_interpolated:
            warnings.append(f"interpolated missing samples for {joint_name}")
    return interpolated, warnings


def apply_temporal_smoothing(positions: np.ndarray, smoothing_window: int) -> np.ndarray:
    smoothed = positions.copy()
    for joint_index in range(positions.shape[1]):
        smoothed[:, joint_index, :] = _moving_average(smoothed[:, joint_index, :], smoothing_window)
    return smoothed


def stabilize_root(positions: np.ndarray, root_window: int) -> np.ndarray:
    stabilized = positions.copy()
    root_index = JOINT_INDEX["Hips"]
    root = stabilized[:, root_index, :]
    root_smoothed = root.copy()
    root_smoothed[:, [0, 2]] = _moving_average(root[:, [0, 2]], root_window)
    root_smoothed[:, [1]] = _moving_average(root[:, [1]], max(3, root_window - 2))
    delta = root_smoothed - root
    stabilized += delta[:, None, :]
    return stabilized


def detect_foot_contact_frames(
    positions: np.ndarray,
    *,
    velocity_threshold: float = 0.018,
    height_threshold: float = 0.035,
) -> dict[str, np.ndarray]:
    contacts: dict[str, np.ndarray] = {}
    for foot_name in ("LeftFoot", "RightFoot"):
        foot = positions[:, JOINT_INDEX[foot_name], :]
        planar_velocity = np.zeros(foot.shape[0], dtype=bool)
        if foot.shape[0] > 1:
            delta = np.diff(foot[:, [0, 2]], axis=0)
            speed = np.linalg.norm(delta, axis=1)
            planar_velocity[1:] = speed < velocity_threshold
        min_height = float(np.percentile(foot[:, 1], 20))
        low_height = foot[:, 1] < (min_height + height_threshold)
        contacts[foot_name] = planar_velocity & low_height
    return contacts


def apply_anti_foot_sliding(
    positions: np.ndarray,
    contacts: dict[str, np.ndarray],
    *,
    minimum_contact_frames: int = 3,
) -> np.ndarray:
    corrected = positions.copy()
    correction = np.zeros((positions.shape[0], 3), dtype=np.float64)
    correction_count = np.zeros(positions.shape[0], dtype=np.float64)

    for foot_name, contact_mask in contacts.items():
        foot_index = JOINT_INDEX[foot_name]
        start = None
        for frame_index, is_contact in enumerate(contact_mask.tolist() + [False]):
            if is_contact and start is None:
                start = frame_index
                continue
            if is_contact:
                continue
            if start is None:
                continue
            end = frame_index
            if end - start >= minimum_contact_frames:
                anchor = corrected[start, foot_index, :].copy()
                for contact_frame in range(start, end):
                    drift = corrected[contact_frame, foot_index, :] - anchor
                    correction[contact_frame, 0] += drift[0]
                    correction[contact_frame, 2] += drift[2]
                    correction_count[contact_frame] += 1.0
            start = None

    valid = correction_count > 0
    if np.any(valid):
        correction[valid] /= correction_count[valid, None]
        corrected[:, :, 0] -= correction[:, 0][:, None]
        corrected[:, :, 2] -= correction[:, 2][:, None]
    return corrected


def joint_confidences(landmarks) -> np.ndarray:
    visibility = {
        name: float(landmarks[index].visibility)
        for index, name in enumerate(
            [
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
        )
    }
    confidence = np.zeros(len(JOINT_ORDER), dtype=np.float64)
    for joint_index, joint_name in enumerate(JOINT_ORDER):
        names = JOINT_CONFIDENCE_LANDMARKS[joint_name]
        confidence[joint_index] = float(np.mean([visibility[name] for name in names]))
    return confidence


def positions_to_frames(positions: np.ndarray) -> list[dict[str, np.ndarray]]:
    frames: list[dict[str, np.ndarray]] = []
    for frame in positions:
        frames.append(
            {
                joint_name: frame[joint_index].astype(np.float64).copy()
                for joint_index, joint_name in enumerate(JOINT_ORDER)
            }
        )
    return frames


def extract_motion_sequence(
    *,
    video_path: Path,
    frame_step: int,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    overlay_output: Path | None,
    overlay_line_thickness: int,
    overlay_joint_radius: int,
    confidence_threshold: float,
) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"failed to open video: {video_path}. Verify the file exists and OpenCV supports its codec."
        )

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

    positions: list[np.ndarray] = []
    rest_image_root: np.ndarray | None = None
    meters_per_pixel: float | None = None
    frame_index = 0
    overlay_writer: cv2.VideoWriter | None = None
    previous_overlay_joints: dict[str, tuple[int, int]] | None = None

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
                positions.append(np.full((len(JOINT_ORDER), 3), np.nan, dtype=np.float64))
                if overlay_writer is not None and previous_overlay_joints is not None:
                    overlay_writer.write(
                        draw_overlay(
                            frame_bgr=frame_bgr,
                            joint_pixels=previous_overlay_joints,
                            line_thickness=overlay_line_thickness,
                            joint_radius=overlay_joint_radius,
                        )
                    )
                elif overlay_writer is not None:
                    overlay_writer.write(frame_bgr)
                frame_index += 1
                continue

            image_points = as_named_points(results.pose_landmarks.landmark)
            world_points = as_named_points(results.pose_world_landmarks.landmark)
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

            joint_position_array = np.stack([joints[joint_name] for joint_name in JOINT_ORDER], axis=0)
            confidence = joint_confidences(results.pose_landmarks.landmark)
            joint_position_array[confidence < confidence_threshold] = np.nan
            positions.append(joint_position_array)

            if overlay_writer is not None:
                overlay_joints = build_overlay_joints(image_points=image_points, width=width, height=height)
                previous_overlay_joints = dict(overlay_joints)
                overlay_writer.write(
                    draw_overlay(
                        frame_bgr=frame_bgr,
                        joint_pixels=overlay_joints,
                        line_thickness=overlay_line_thickness,
                        joint_radius=overlay_joint_radius,
                    )
                )

            frame_index += 1
    finally:
        pose.close()
        cap.release()
        if overlay_writer is not None:
            overlay_writer.release()

    if not positions:
        raise RuntimeError("MediaPipe did not produce any pose samples")
    return np.asarray(positions, dtype=np.float64), fps / float(frame_step)


def capture_motion_to_bvh(
    video_path: Path,
    bvh_path: Path,
    overlay_path: Path | None = None,
    *,
    scale: float = 1.0,
    frame_step: int = 1,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    overlay_line_thickness: int = 3,
    overlay_joint_radius: int = 4,
    remux_audio: bool = True,
    smoothing_window: int = 5,
    root_window: int = 9,
    confidence_threshold: float = 0.35,
    detect_foot_contacts: bool = True,
    anti_foot_sliding: bool = True,
) -> MotionCaptureResult:
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be >= 1")
    if root_window < 1:
        raise ValueError("root_window must be >= 1")
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    bvh_path.parent.mkdir(parents=True, exist_ok=True)
    if overlay_path is not None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)

    raw_positions, fps = extract_motion_sequence(
        video_path=video_path,
        frame_step=frame_step,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        overlay_output=overlay_path,
        overlay_line_thickness=overlay_line_thickness,
        overlay_joint_radius=overlay_joint_radius,
        confidence_threshold=confidence_threshold,
    )
    if not np.isfinite(raw_positions).any():
        raise RuntimeError("MediaPipe did not produce any usable pose samples")

    processed_positions, warnings = interpolate_missing_landmarks(raw_positions)
    processed_positions = apply_temporal_smoothing(processed_positions, smoothing_window=max(1, smoothing_window))
    processed_positions = stabilize_root(processed_positions, root_window=max(3, root_window))
    if not np.isfinite(processed_positions).all():
        raise RuntimeError("pose processing produced invalid NaN or infinite values after smoothing/interpolation")

    foot_contacts: dict[str, np.ndarray] = {}
    if detect_foot_contacts or anti_foot_sliding:
        foot_contacts = detect_foot_contact_frames(processed_positions)
    if anti_foot_sliding and foot_contacts:
        processed_positions = apply_anti_foot_sliding(processed_positions, foot_contacts)

    frames = positions_to_frames(processed_positions)
    write_bvh(output_path=bvh_path, frames=frames, fps=fps, scale=scale)

    if overlay_path is not None and remux_audio:
        remux_ok = remux_overlay_audio(source_video=video_path, overlay_video=overlay_path)
        if not remux_ok:
            warnings.append("overlay audio remux failed or ffmpeg was unavailable; overlay video was left video-only")

    return MotionCaptureResult(
        video_path=video_path,
        bvh_path=bvh_path,
        overlay_path=overlay_path,
        fps=fps,
        frame_count=len(frames),
        warnings=warnings,
        foot_contact_frames={
            foot_name: int(np.count_nonzero(contact_mask))
            for foot_name, contact_mask in foot_contacts.items()
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract human pose from a video with MediaPipe and export a BVH skeleton.",
    )
    parser.add_argument("--input", type=Path, required=True, help="input video path, for example video.mp4")
    parser.add_argument("--output", type=Path, required=True, help="output BVH path")
    parser.add_argument(
        "--overlay-output",
        type=Path,
        default=None,
        help="optional MP4 path for a skeleton overlay preview",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="BVH unit scale multiplier")
    parser.add_argument("--frame-step", type=int, default=1, help="process every Nth frame")
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
        "--confidence-threshold",
        type=float,
        default=0.35,
        help="joint visibility threshold below which samples are interpolated",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="temporal smoothing window in frames",
    )
    parser.add_argument(
        "--root-window",
        type=int,
        default=9,
        help="root stabilization window in frames",
    )
    parser.add_argument("--skip-audio-remux", action="store_true", help="do not copy source audio into the overlay video")
    parser.add_argument(
        "--disable-foot-contact-detection",
        action="store_true",
        help="skip foot contact detection diagnostics",
    )
    parser.add_argument(
        "--disable-anti-foot-sliding",
        action="store_true",
        help="disable simple root correction during foot contact runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = capture_motion_to_bvh(
        video_path=args.input,
        bvh_path=args.output,
        overlay_path=args.overlay_output,
        scale=args.scale,
        frame_step=args.frame_step,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        remux_audio=not args.skip_audio_remux,
        smoothing_window=args.smoothing_window,
        root_window=args.root_window,
        confidence_threshold=args.confidence_threshold,
        detect_foot_contacts=not args.disable_foot_contact_detection,
        anti_foot_sliding=not args.disable_anti_foot_sliding,
    )
    print(f"BVH written to {result.bvh_path}")
    print(f"Frames: {result.frame_count}")
    print(f"FPS: {result.fps:.3f}")
    if result.foot_contact_frames:
        print(f"Foot contacts: {result.foot_contact_frames}")
    if result.overlay_path is not None:
        print(f"Overlay video: {result.overlay_path}")


if __name__ == "__main__":
    main()
