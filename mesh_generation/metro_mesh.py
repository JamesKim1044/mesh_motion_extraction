from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_IMAGE = PROJECT_ROOT / "zelda.png"
DEFAULT_OUTPUT_OBJ = PROJECT_ROOT / "human.obj"


@dataclass(slots=True)
class MetroMeshResult:
    input_image_path: Path
    output_obj_path: Path
    preprocessed_image_path: Path | None
    metro_result_obj_path: Path


def letterbox_person_image(
    input_image_path: Path,
    output_image_path: Path,
    *,
    image_size: int = 224,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Path:
    """Resize an input image into METRO's expected 224x224 canvas.

    The official METRO demo expects the person to already be centered in a 224x224
    image. This helper keeps aspect ratio and pads the remaining area so the image
    can be passed to the official inference script directly.
    """

    image = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {input_image_path}")

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError(f"invalid image dimensions: {input_image_path}")

    scale = min(image_size / width, image_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    canvas = np.full((image_size, image_size, 3), background_color, dtype=np.uint8)
    offset_x = (image_size - resized_width) // 2
    offset_y = (image_size - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_image_path), canvas):
        raise RuntimeError(f"failed to write preprocessed image: {output_image_path}")
    return output_image_path


def _find_latest_obj(directory: Path) -> Path:
    candidates = sorted(directory.rglob("*.obj"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"METRO did not produce an OBJ file under {directory}")
    return candidates[0]


def _resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def default_metro_repo_path() -> Path | None:
    env_path = _resolve_path(os.environ.get("METRO_REPO"))
    if env_path is not None:
        return env_path

    candidates = [
        PROJECT_ROOT.parent / "MeshTransformer",
        PROJECT_ROOT / "MeshTransformer",
        PROJECT_ROOT.parent / "meshtransformer",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def default_checkpoint_path(metro_repo_path: Path | None) -> Path | None:
    env_path = _resolve_path(os.environ.get("METRO_CHECKPOINT"))
    if env_path is not None:
        return env_path
    if metro_repo_path is None:
        return None

    candidates = [
        metro_repo_path / "models" / "metro_release" / "metro_3dpw_state_dict.bin",
        metro_repo_path / "models" / "metro_release" / "metro_bodymesh.bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def generate_human_mesh_with_metro(
    input_image_path: Path,
    output_obj_path: Path,
    *,
    metro_repo_path: Path,
    checkpoint_path: Path,
    python_executable: str = "python3",
    keep_workdir: bool = False,
) -> MetroMeshResult:
    """Generate a human OBJ mesh from a single image using the official METRO demo.

    Expected external setup:
    - local checkout of https://github.com/microsoft/MeshTransformer
    - METRO checkpoint, for example:
      models/metro_release/metro_3dpw_state_dict.bin
    - required SMPL assets placed under:
      metro/modeling/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
    """

    if not input_image_path.exists():
        raise FileNotFoundError(f"input image not found: {input_image_path}")
    if not metro_repo_path.exists():
        raise FileNotFoundError(f"METRO repository not found: {metro_repo_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"METRO checkpoint not found: {checkpoint_path}")
    output_obj_path.parent.mkdir(parents=True, exist_ok=True)

    workdir_path: Path
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if keep_workdir:
        workdir_path = output_obj_path.parent / f"{output_obj_path.stem}_metro_work"
        workdir_path.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="metro_mesh_", dir=str(output_obj_path.parent))
        workdir_path = Path(temp_dir.name)

    input_dir = workdir_path / "samples" / "human-body"
    input_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_image_path = input_dir / f"{input_image_path.stem}_224.png"
    letterbox_person_image(input_image_path, preprocessed_image_path)

    command = [
        python_executable,
        "metro/tools/end2end_inference_bodymesh.py",
        "--resume_checkpoint",
        str(checkpoint_path),
        "--image_file_or_path",
        str(input_dir),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(metro_repo_path),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "\n".join(
                    [
                        "METRO inference failed.",
                        f"Command: {' '.join(command)}",
                        completed.stdout.strip(),
                        completed.stderr.strip(),
                    ]
                ).strip()
            )

        metro_result_obj_path = _find_latest_obj(input_dir)
        shutil.copy2(metro_result_obj_path, output_obj_path)

        return MetroMeshResult(
            input_image_path=input_image_path,
            output_obj_path=output_obj_path,
            preprocessed_image_path=preprocessed_image_path if keep_workdir else None,
            metro_result_obj_path=metro_result_obj_path,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def parse_args() -> argparse.Namespace:
    guessed_metro_repo = default_metro_repo_path()
    guessed_checkpoint = default_checkpoint_path(guessed_metro_repo)

    parser = argparse.ArgumentParser(
        description="Generate a 3D human OBJ mesh from a single image using METRO.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_IMAGE,
        help=f"input image, default: {DEFAULT_INPUT_IMAGE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_OBJ,
        help=f"output OBJ path, default: {DEFAULT_OUTPUT_OBJ}",
    )
    parser.add_argument(
        "--metro-repo",
        type=Path,
        default=guessed_metro_repo,
        help="path to a local MeshTransformer checkout, or set METRO_REPO",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=guessed_checkpoint,
        help="METRO body checkpoint path, or set METRO_CHECKPOINT",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default="python3",
        help="Python executable used to run METRO",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="keep the temporary METRO sample directory for debugging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metro_repo is None:
        raise FileNotFoundError(
            "METRO repository path was not provided. Use --metro-repo or set METRO_REPO."
        )
    if args.checkpoint is None:
        raise FileNotFoundError(
            "METRO checkpoint path was not provided. Use --checkpoint or set METRO_CHECKPOINT."
        )
    result = generate_human_mesh_with_metro(
        input_image_path=args.input,
        output_obj_path=args.output,
        metro_repo_path=args.metro_repo,
        checkpoint_path=args.checkpoint,
        python_executable=args.python_exe,
        keep_workdir=args.keep_workdir,
    )
    print(f"OBJ written to {result.output_obj_path}")
    if result.preprocessed_image_path is not None:
        print(f"Preprocessed input: {result.preprocessed_image_path}")


if __name__ == "__main__":
    main()
