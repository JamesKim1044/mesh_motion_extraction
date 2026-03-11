from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

JOINT_SCHEMA_NAME = "humanoid_21_joint_v1"
BONE_NAMING_CONVENTION = "schema_joint_name"
ARMATURE_OBJECT_NAME = "TargetRig"
ARMATURE_DATA_NAME = "TargetRigData"
MESH_OBJECT_NAME = "GeneratedHuman"
COORDINATE_SYSTEM = {
    "space": "local_proxy",
    "handedness": "right",
    "up_axis": "Z",
    "forward_axis": "Y",
    "unit": "meter",
}
SCALE_METADATA = {
    "meters_per_unit": 1.0,
    "bvh_unit_scale": 1.0,
}
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

CHILDREN_MAP: dict[str, list[str]] = {joint: [] for joint in JOINT_ORDER}
for joint_name, parent_name in PARENT_MAP.items():
    if parent_name is not None:
        CHILDREN_MAP[parent_name].append(joint_name)

PRIMARY_CHILD_MAP: dict[str, str | None] = {
    "Hips": "Spine",
    "Spine": "Chest",
    "Chest": "Neck",
    "Neck": "Head",
    "Head": None,
    "LeftShoulder": "LeftElbow",
    "LeftElbow": "LeftWrist",
    "LeftWrist": "LeftHand",
    "LeftHand": None,
    "RightShoulder": "RightElbow",
    "RightElbow": "RightWrist",
    "RightWrist": "RightHand",
    "RightHand": None,
    "LeftHip": "LeftKnee",
    "LeftKnee": "LeftAnkle",
    "LeftAnkle": "LeftFoot",
    "LeftFoot": None,
    "RightHip": "RightKnee",
    "RightKnee": "RightAnkle",
    "RightAnkle": "RightFoot",
    "RightFoot": None,
}

LEAF_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "Head": (0.0, 0.0, 0.18),
    "LeftHand": (0.12, 0.0, 0.0),
    "RightHand": (-0.12, 0.0, 0.0),
    "LeftFoot": (0.0, 0.18, -0.01),
    "RightFoot": (0.0, 0.18, -0.01),
}

DEFAULT_BODY_HEIGHT_M = 1.70
DEFAULT_BONE_MAPPING_HINTS = {
    "Hips": ["Hips", "hips", "hip", "pelvis", "root"],
    "Spine": ["Spine", "spine", "spine1", "spine_01"],
    "Chest": ["Chest", "chest", "spine2", "spine_02", "upperchest"],
    "Neck": ["Neck", "neck"],
    "Head": ["Head", "head"],
    "LeftShoulder": ["LeftShoulder", "leftshoulder", "left_shoulder", "l_shoulder", "shoulder_l"],
    "LeftElbow": ["LeftElbow", "leftelbow", "left_elbow", "l_elbow", "lowerarm_l"],
    "LeftWrist": ["LeftWrist", "leftwrist", "left_wrist", "l_wrist", "hand_l"],
    "LeftHand": ["LeftHand", "lefthand", "left_hand", "l_hand"],
    "RightShoulder": ["RightShoulder", "rightshoulder", "right_shoulder", "r_shoulder", "shoulder_r"],
    "RightElbow": ["RightElbow", "rightelbow", "right_elbow", "r_elbow", "lowerarm_r"],
    "RightWrist": ["RightWrist", "rightwrist", "right_wrist", "r_wrist", "hand_r"],
    "RightHand": ["RightHand", "righthand", "right_hand", "r_hand"],
    "LeftHip": ["LeftHip", "lefthip", "left_hip", "l_hip", "thigh_l", "upleg_l"],
    "LeftKnee": ["LeftKnee", "leftknee", "left_knee", "l_knee", "calf_l", "leg_l"],
    "LeftAnkle": ["LeftAnkle", "leftankle", "left_ankle", "l_ankle", "foot_l"],
    "LeftFoot": ["LeftFoot", "leftfoot", "left_foot", "l_foot", "toe_l"],
    "RightHip": ["RightHip", "righthip", "right_hip", "r_hip", "thigh_r", "upleg_r"],
    "RightKnee": ["RightKnee", "rightknee", "right_knee", "r_knee", "calf_r", "leg_r"],
    "RightAnkle": ["RightAnkle", "rightankle", "right_ankle", "r_ankle", "foot_r"],
    "RightFoot": ["RightFoot", "rightfoot", "right_foot", "r_foot", "toe_r"],
}


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bvh_vector_to_blender(vector: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    x, y, z = vector
    return float(x), float(-z), float(y)


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
