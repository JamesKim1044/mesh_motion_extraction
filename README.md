# Motion Retargeting Pipeline

`mesh_motion_extraction_v3` is a staged Python and Blender pipeline for:

1. video -> MediaPipe motion -> BVH
2. image -> procedural humanoid proxy mesh
3. mesh -> Blender rig
4. BVH -> retargeted animation
5. render -> MP4

The architecture stays close to the existing project layout. The refactor adds restartability, explicit artifact contracts, stage validation, and manifest reporting instead of redesigning the system.

## Project layout

```text
mesh_motion_extraction_v3
├── blender
│   ├── render_animation.py
│   ├── render_runner.py
│   └── runner.py
├── mesh_generation
│   └── procedural_human_mesh.py
├── motion_capture
│   └── mediapipe_bvh.py
├── retargeting
│   ├── blender_retarget.py
│   ├── bvh_parser.py
│   └── runner.py
├── rigging
│   ├── blender_rigging.py
│   └── runner.py
├── common.py
├── main.py
├── pipeline_environment.py
├── pipeline_validation.py
└── README.md
```

## Installation

### System dependencies

Ubuntu 22.04:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3-venv
```

Install Blender separately or make sure `blender` is already in `PATH`.

### Python dependencies

```bash
cd /home/root_james/Projects/pilot_AXON/mesh_motion_extraction_v3
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Minimum runtime modules used by the pipeline:

- `numpy`
- `opencv-python`
- `mediapipe`

## Pipeline overview

### Stage 1: Motion extraction

- Module: `motion_capture/mediapipe_bvh.py`
- Input: source video
- Output: `motion.bvh`
- Optional output: `motion_overlay.mp4`

This stage keeps the MediaPipe backend and adds:

- temporal smoothing
- missing sample interpolation
- confidence filtering
- root stabilization
- optional foot contact detection
- simple anti-foot-sliding root correction

### Stage 2: Mesh generation

- Module: `mesh_generation/procedural_human_mesh.py`
- Input: single RGB image
- Outputs:
  - `generated_human.obj`
  - `generated_human.json`

Important limitation:
The mesh is a procedural humanoid proxy for rigging and retargeting. It is not a true person-specific 3D reconstruction.

The metadata JSON now includes:

- joint schema
- coordinate system
- scale metadata
- rest pose
- bone mapping hints

### Stage 3: Rigging

- Module: `rigging/blender_rigging.py`
- Inputs:
  - `generated_human.obj`
  - `generated_human.json`
- Outputs:
  - `rigged_scene.blend`
  - optional `rig_report.json`

The rigging stage now validates required metadata, normalizes mesh scale, builds a deterministic armature, and validates vertex group ranges before binding.

### Stage 4: Retargeting

- Modules:
  - `retargeting/bvh_parser.py`
  - `retargeting/blender_retarget.py`
- Inputs:
  - `motion.bvh`
  - `generated_human.json`
  - `rigged_scene.blend`
- Outputs:
  - `animated_scene.blend`
  - optional `retarget_report.json`

The retargeter now uses:

- joint mapping hints
- axis normalization hooks
- rest-pose alignment
- root motion normalization
- missing-joint warnings

### Stage 5: Rendering

- Module: `blender/render_animation.py`
- Inputs:
  - `animated_scene.blend`
  - `generated_human.json`
- Output:
  - `retargeted_animation.mp4`

The renderer checks that animation exists before rendering, recreates a deterministic camera and lighting setup, and falls back to:

1. Blender movie render
2. PNG sequence + `ffmpeg`

## Validation and manifest

`main.py` validates stage artifacts and writes `pipeline_manifest.json`.

Validated artifacts:

- `motion.bvh`
- `generated_human.obj`
- `generated_human.json`
- `rigged_scene.blend`
- `animated_scene.blend`
- `retargeted_animation.mp4`

The manifest records:

- stage status
- timestamps
- input and output paths
- validation summaries
- warnings
- environment checks

## CLI flags

New stage-control flags:

- `--skip-motion`
- `--skip-mesh`
- `--skip-rig`
- `--skip-retarget`
- `--skip-render`
- `--resume`
- `--force`
- `--validate-only`

Behavior:

- `--resume`: reuse valid, up-to-date artifacts
- `--force`: recompute all non-skipped stages
- `--validate-only`: validate artifacts and write the manifest without executing stages

## Example commands

### Full run

```bash
python main.py --video input.mp4 --image person.png
```

### Resume

```bash
python main.py --resume
```

### Skip motion

```bash
python main.py --skip-motion --resume
```

### Validate only

```bash
python main.py --validate-only
```

### Custom output directory

```bash
python main.py \
  --video taekwondo-1.mp4 \
  --image zelda.png \
  --output-dir outputs/notebook_demo \
  --blender-exe blender
```

## Limitations

- The mesh stage emits a proxy humanoid, not a production body reconstruction.
- Retargeting is position-driven and optimized for the project’s 21-joint schema.
- External BVH files with very different naming or orientation may still require metadata hint updates.
- If Blender lacks native movie encoding, final MP4 export depends on system `ffmpeg`.
# mesh_motion_extraction
