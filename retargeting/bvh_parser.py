from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


@dataclass(slots=True)
class BVHJoint:
    name: str
    parent: str | None
    offset: tuple[float, float, float]
    channels: list[str]
    children: list[str] = field(default_factory=list)
    end_offset: tuple[float, float, float] | None = None


@dataclass(slots=True)
class BVHAnimation:
    root_name: str
    joints: dict[str, BVHJoint]
    joint_order: list[str]
    frames: list[list[float]]
    frame_time: float


@dataclass(slots=True)
class JointMappingResult:
    mapping: dict[str, str]
    missing_targets: list[str]
    warnings: list[str]


def joint_has_translation_channels(joint: BVHJoint) -> bool:
    return any(channel_name.endswith("position") for channel_name in joint.channels)


def _parse_floats(parts: list[str]) -> tuple[float, float, float]:
    return float(parts[0]), float(parts[1]), float(parts[2])


def _normalized_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _parse_joint(
    lines: list[str],
    index: int,
    parent_name: str | None,
    joints: dict[str, BVHJoint],
    joint_order: list[str],
) -> tuple[str, int]:
    tokens = lines[index].split()
    node_type = tokens[0]
    if node_type not in {"ROOT", "JOINT"}:
        raise ValueError(f"expected ROOT or JOINT at line {index + 1}: {lines[index]}")
    joint_name = tokens[1]
    joint_order.append(joint_name)
    index += 1

    if lines[index] != "{":
        raise ValueError(f"expected '{{' after joint {joint_name}")
    index += 1

    offset: tuple[float, float, float] | None = None
    channels: list[str] | None = None
    children: list[str] = []
    end_offset: tuple[float, float, float] | None = None

    while index < len(lines):
        line = lines[index]
        if line == "}":
            if offset is None or channels is None:
                raise ValueError(f"incomplete BVH joint block for {joint_name}")
            joints[joint_name] = BVHJoint(
                name=joint_name,
                parent=parent_name,
                offset=offset,
                channels=channels,
                children=children,
                end_offset=end_offset,
            )
            return joint_name, index + 1
        if line.startswith("OFFSET "):
            offset = _parse_floats(line.split()[1:4])
            index += 1
            continue
        if line.startswith("CHANNELS "):
            parts = line.split()
            channel_count = int(parts[1])
            channels = parts[2 : 2 + channel_count]
            index += 1
            continue
        if line.startswith("JOINT "):
            child_name, index = _parse_joint(lines, index, joint_name, joints, joint_order)
            children.append(child_name)
            continue
        if line == "End Site":
            if lines[index + 1] != "{":
                raise ValueError(f"expected '{{' after End Site for {joint_name}")
            offset_line = lines[index + 2]
            if not offset_line.startswith("OFFSET "):
                raise ValueError(f"expected End Site OFFSET for {joint_name}")
            end_offset = _parse_floats(offset_line.split()[1:4])
            if lines[index + 3] != "}":
                raise ValueError(f"expected '}}' after End Site for {joint_name}")
            index += 4
            continue
        raise ValueError(f"unexpected BVH line {index + 1}: {line}")

    raise ValueError(f"unexpected end of file while parsing {joint_name}")


def load_bvh_animation(path: Path) -> BVHAnimation:
    raw_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_lines or raw_lines[0] != "HIERARCHY":
        raise ValueError(f"not a BVH file: {path}")

    joints: dict[str, BVHJoint] = {}
    joint_order: list[str] = []
    root_name, index = _parse_joint(raw_lines, 1, None, joints, joint_order)

    if index >= len(raw_lines) or raw_lines[index] != "MOTION":
        raise ValueError("BVH MOTION block not found")
    frame_count = int(raw_lines[index + 1].split(":")[1].strip())
    frame_time = float(raw_lines[index + 2].split(":")[1].strip())

    frames: list[list[float]] = []
    motion_start = index + 3
    for row_index in range(frame_count):
        frame_index = motion_start + row_index
        if frame_index >= len(raw_lines):
            raise ValueError("BVH MOTION block ended before the declared frame count")
        frames.append([float(value) for value in raw_lines[frame_index].split()])

    return BVHAnimation(
        root_name=root_name,
        joints=joints,
        joint_order=joint_order,
        frames=frames,
        frame_time=frame_time,
    )


def build_joint_mapping(
    animation: BVHAnimation,
    target_joint_names: list[str],
    bone_mapping_hints: dict[str, list[str]] | None = None,
) -> JointMappingResult:
    normalized_source = {_normalized_label(name): name for name in animation.joints}
    mapping: dict[str, str] = {}
    missing_targets: list[str] = []
    warnings: list[str] = []

    for target_name in target_joint_names:
        aliases = [target_name]
        if bone_mapping_hints and target_name in bone_mapping_hints:
            aliases.extend(str(alias) for alias in bone_mapping_hints[target_name])

        match = None
        for alias in aliases:
            source_name = normalized_source.get(_normalized_label(alias))
            if source_name is not None:
                match = source_name
                break

        if match is None:
            missing_targets.append(target_name)
            warnings.append(f"source BVH is missing a joint mapping for target joint {target_name}")
            continue
        mapping[target_name] = match

    return JointMappingResult(mapping=mapping, missing_targets=missing_targets, warnings=warnings)


def translation_channel_warnings(
    animation: BVHAnimation,
    mapping: dict[str, str],
) -> list[str]:
    warnings: list[str] = []
    for target_name, source_name in mapping.items():
        joint = animation.joints.get(source_name)
        if joint is None:
            continue
        if not joint_has_translation_channels(joint):
            warnings.append(
                f"source joint {source_name} mapped to {target_name} has no translation channels; "
                "position-driven retargeting may remain close to rest pose"
            )
    return warnings


def rest_positions(animation: BVHAnimation) -> dict[str, tuple[float, float, float]]:
    positions: dict[str, tuple[float, float, float]] = {}

    def visit(joint_name: str) -> None:
        joint = animation.joints[joint_name]
        if joint.parent is None:
            positions[joint_name] = joint.offset
        else:
            parent = positions[joint.parent]
            positions[joint_name] = (
                parent[0] + joint.offset[0],
                parent[1] + joint.offset[1],
                parent[2] + joint.offset[2],
            )
        for child_name in joint.children:
            visit(child_name)

    visit(animation.root_name)
    return positions


def frame_positions(animation: BVHAnimation, frame_values: list[float]) -> dict[str, tuple[float, float, float]]:
    positions: dict[str, tuple[float, float, float]] = {}
    cursor = 0

    local_deltas: dict[str, tuple[float, float, float]] = {}
    for joint_name in animation.joint_order:
        joint = animation.joints[joint_name]
        joint_channel_count = len(joint.channels)
        values = frame_values[cursor : cursor + joint_channel_count]
        if len(values) != joint_channel_count:
            raise ValueError(f"frame does not contain enough channel values for joint {joint_name}")
        cursor += joint_channel_count
        x = y = z = 0.0
        for channel_name, value in zip(joint.channels, values):
            if channel_name == "Xposition":
                x = value
            elif channel_name == "Yposition":
                y = value
            elif channel_name == "Zposition":
                z = value
        local_deltas[joint_name] = (x, y, z)

    def visit(joint_name: str) -> None:
        joint = animation.joints[joint_name]
        delta = local_deltas[joint_name]
        if joint.parent is None:
            positions[joint_name] = (
                joint.offset[0] + delta[0],
                joint.offset[1] + delta[1],
                joint.offset[2] + delta[2],
            )
        else:
            parent_position = positions[joint.parent]
            positions[joint_name] = (
                parent_position[0] + joint.offset[0] + delta[0],
                parent_position[1] + joint.offset[1] + delta[1],
                parent_position[2] + joint.offset[2] + delta[2],
            )
        for child_name in joint.children:
            visit(child_name)

    visit(animation.root_name)
    return positions


def all_frame_positions(animation: BVHAnimation) -> list[dict[str, tuple[float, float, float]]]:
    return [frame_positions(animation, frame) for frame in animation.frames]


def apply_axis_multipliers(
    positions: dict[str, tuple[float, float, float]],
    axis_multipliers: tuple[float, float, float],
) -> dict[str, tuple[float, float, float]]:
    return {
        joint_name: (
            point[0] * axis_multipliers[0],
            point[1] * axis_multipliers[1],
            point[2] * axis_multipliers[2],
        )
        for joint_name, point in positions.items()
    }


def map_positions_to_targets(
    positions: dict[str, tuple[float, float, float]],
    mapping: dict[str, str],
) -> dict[str, tuple[float, float, float]]:
    return {target_name: positions[source_name] for target_name, source_name in mapping.items() if source_name in positions}


def normalize_root_motion(
    frames: list[dict[str, tuple[float, float, float]]],
    *,
    root_name: str,
) -> list[dict[str, tuple[float, float, float]]]:
    if not frames or root_name not in frames[0]:
        return frames
    initial_root = frames[0][root_name]
    normalized: list[dict[str, tuple[float, float, float]]] = []
    for frame in frames:
        frame_root = frame[root_name]
        delta = (
            frame_root[0] - initial_root[0],
            frame_root[1] - initial_root[1],
            frame_root[2] - initial_root[2],
        )
        normalized.append(
            {
                joint_name: (
                    point[0] - initial_root[0],
                    point[1] - initial_root[1],
                    point[2] - initial_root[2],
                )
                if joint_name == root_name
                else (
                    point[0] - delta[0],
                    point[1] - delta[1],
                    point[2] - delta[2],
                )
                for joint_name, point in frame.items()
            }
        )
    return normalized


def estimate_height(positions: dict[str, tuple[float, float, float]]) -> float:
    ys = [value[1] for value in positions.values()]
    return max(ys) - min(ys)
