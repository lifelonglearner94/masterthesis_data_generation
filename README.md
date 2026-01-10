# Kubric Data Generation Pipeline

A scientific data generation pipeline for creating synthetic video datasets of friction physics experiments using [Kubric](https://github.com/google-research/kubric), PyBullet, and Blender.

## 🎯 Overview

This pipeline generates 256×256 video clips of a cube receiving force impulses on surfaces with varying friction coefficients. Each clip includes:

- **RGB video** (32 frames @ 20 FPS)
- **Optical flow** (NumPy format)
- **Segmentation masks**
- **V-JEPA2 feature maps** (encoded ViT-L/16 embeddings)
- **Ground truth metadata** (friction, force vectors, mass, etc.)

## 📁 Project Structure

```
experiments/
├── config/
│   └── physics_config.yaml       # Physics parameter ranges
├── scripts/
│   ├── generate_single_clip.py   # Worker: generates one clip
│   ├── manager_benchmark.py      # Orchestrator: manages full dataset
│   └── encode_benchmark_vjepa2.py # V-JEPA2 feature extraction
└── output/
    └── [generated datasets]
```

---

## 🚀 Two-Phase Workflow

The pipeline runs in two phases:
1. **Phase 1 (Docker)**: Render clips using Kubric/Blender/PyBullet
2. **Phase 2 (Local)**: Encode clips using V-JEPA2 on GPU

---

## 📋 Option A: Test Scenario (6 Clips)

Quick validation that everything works.

### Step 1: Render Clips (Docker)

```bash
# From project root
cd /path/to/masterthesis_data_generation

# Ensure the floor texture exists (committed as generate_grid_image/black_grid.png).
# If you regenerated/removed it, recreate it once:
uv run python3 generate_grid_image/generate_black_grid_png.py

# Start Docker container with GPU and increased shared memory for parallel workers
docker run --rm -it \
    --shm-size=8g \
    -v "$(pwd):/workspace" \
    -w /workspace \
    kubricdockerhub/kubruntu:latest \
    /bin/bash
```

**Inside Docker container:**

```bash
# If this is the first run in a fresh container, the scripts may install
# `python3-yaml`/`python3-numpy` via apt (preferred) or fall back to pip.
python experiments/scripts/manager_benchmark.py \
    --test \
    --seed 42 \
    --test_a1_clips 3 \
    --test_b_clips 3 \
    --test_a2_clips 0 \
    --output_dir experiments/output/test_6clips
```

If your environment blocks outbound PyPI access and dependency installation fails,
run this once inside the container and retry:

```bash
apt-get update && apt-get install -y python3-yaml python3-numpy
```

Then exit Docker with `exit`.

### Step 2: Fix Permissions (Local)

```bash
sudo chown -R $USER:$USER experiments/output/test_6clips
```

### Step 3: Encode with V-JEPA2 (Local)

```bash
uv run experiments/scripts/encode_benchmark_vjepa2.py \
    --data_dir experiments/output/test_6clips \
    --in_place \
    --batch_size 4
```

### Verify Results

```bash
# Check that feature maps were created
ls experiments/output/test_6clips/clip_*/feature_maps/

# Inspect the first feature map found
python -c "import glob, numpy as np; p = sorted(glob.glob('experiments/output/test_6clips/clip_*/feature_maps/vjepa2_vitl16.npy'))[0]; a = np.load(p); print(f'File: {p} | Shape: {a.shape} | Dtype: {a.dtype}')"
```

---

## �� Option B: Full Production (17,000 Clips)

Generate the complete A-B-A dataset:
- **Phase A₁** (clips 0–14,999): Normal friction, random mass
- **Phase B** (clips 15,000–15,999): Slippery friction, fixed mass (OOD)
- **Phase A₂** (clips 16,000–16,999): Normal friction, fixed mass

### Step 1: Render All Clips (Docker)

```bash
# From project root
cd /path/to/masterthesis_data_generation

# Ensure the floor texture exists (committed as generate_grid_image/black_grid.png).
# If you regenerated/removed it, recreate it once:
python3 generate_grid_image/generate_black_grid_png.py

# Start Docker container with GPU and increased shared memory for parallel workers
# NOTE: --shm-size=8g is REQUIRED for parallel execution
docker run --rm -it \
    --gpus all \
    --shm-size=8g \
    -v "$(pwd):/workspace" \
    -w /workspace \
    kubricdockerhub/kubruntu:latest \
    /bin/bash
```

**Inside Docker container (parallel execution recommended):**

```bash
# Use parallel workers for faster generation (saturates CPU/GPU pipeline)
python experiments/scripts/manager_benchmark.py \
    --seed 42 \
    --output_dir experiments/output/friction_dataset_v1 \
    --start_job 0 \
    --end_job 16999 \
    --workers 4 \
    --auto_scale \
    --no_gif
```

> 💡 **Parallel Execution**: Using `--workers 4` with `--auto_scale` keeps the GPU busy while CPUs prepare the next frames. This can reduce total time by 50-70%.

> 🔄 **Auto-Resume**: If interrupted (even by shutdown), just re-run the same command. The pipeline automatically skips completed clips.

> ⏱️ **ETA Tracking**: The pipeline prints detailed time estimates every 500 clips (or every 30 minutes), using a rolling window of recent clip times for accurate predictions. This shows both recent and overall rates, plus expected completion time.

> ⏱️ **Estimated time**: ~3-4 days with 4 parallel workers (vs ~8-10 days sequential)

Then exit Docker with `exit`.

### Step 2: Fix Permissions (Local)

```bash
sudo chown -R $USER:$USER experiments/output/friction_dataset_v1
```

### Step 3: Encode with V-JEPA2 (Local)

```bash
uv run experiments/scripts/encode_benchmark_vjepa2.py \
    --data_dir experiments/output/friction_dataset_v1 \
    --in_place \
    --batch_size 4
```

> ⏱️ **Estimated time**: ~2-4 hours on a single GPU

---

## 📊 Output Structure

Each generated clip has this structure:

```
output/dataset_name/
├── benchmark_report.json       # Timing and success statistics
├── generation.log              # Full generation log
├── clip_00000/
│   ├── rgb/
│   │   ├── frame_00000.png
│   │   └── ... (32 frames)
│   ├── flow/
│   │   ├── flow_00000.npy
│   │   └── ...
│   ├── segmentation/
│   │   ├── seg_00000.png
│   │   └── ...
│   ├── feature_maps/
│   │   └── vjepa2_vitl16.npy   # (4096, 1024) float16
│   ├── preview.gif             # Quick visual preview (if --no_gif not used)
│   └── ground_truth.json
├── clip_00001/
└── ...
```

---

## 🔧 Configuration

Edit `experiments/config/physics_config.yaml` to customize physics parameters:

| Phase | Friction | Mass Mode | Clip Range |
|-------|----------|-----------|------------|
| A₁ | Normal (0.5–0.9) | Random | 0–14,999 |
| B | Slippery (0.01–0.15) | Fixed | 15,000–15,999 |
| A₂ | Normal (0.5–0.9) | Fixed | 16,000–16,999 |

---

## 📋 CLI Reference

### manager_benchmark.py (Rendering)

| Flag | Description |
|------|-------------|
| `--output_dir` | Base directory for all clips |
| `--seed` | Global seed for reproducibility |
| `--test` | Run benchmark mode |
| `--test_clips N` | Number of clips in test mode (default: 5) |
| `--start_job N` | Starting job ID (default: 0) |
| `--end_job N` | Ending job ID (default: 16999) |
| `--dry_run` | Metadata only, no rendering |
| `--workers N` | Parallel workers (default: CPU_COUNT-2). Use 1 for sequential |
| `--auto_scale` | VRAM-aware throttling: pause submission when GPU memory is high |
| `--force_restart` | Ignore existing completed clips and regenerate all (disables auto-resume) |
| `--no_gif` | Skip GIF preview generation (recommended for production) |

### encode_benchmark_vjepa2.py (Encoding)

| Flag | Description |
|------|-------------|
| `--data_dir` | Path to dataset with clip_* folders |
| `--in_place` | Save to clip_*/feature_maps/ (recommended) |
| `--out_dir` | Alternative: central output directory |
| `--batch_size` | Clips per batch (default: 4) |
| `--dtype` | fp16 or fp32 (default: fp16) |
| `--overwrite` | Overwrite existing embeddings |
| `--crop_size` | V-JEPA2 crop size (default: 256) |

---

## ⚠️ Troubleshooting

### GPU Not Detected in Docker

```bash
# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Permission Denied Errors

Docker creates files as root. Fix with:
```bash
sudo chown -R $USER:$USER experiments/output/
```

### Parallel Execution Crashes / Freezing

If parallel workers cause crashes or system freezing:

1. **Increase shared memory** (required for IPC):
   ```bash
   docker run --shm-size=8g ...  # Minimum 8GB recommended
   ```

2. **Reduce worker count** if CPU is overloaded:
   ```bash
   --workers 2  # Start conservative, increase gradually
   ```

3. **Enable VRAM monitoring** to prevent GPU OOM:
   ```bash
   --auto_scale  # Pauses new jobs when GPU memory > 85%
   ```

4. **Check for VRAM errors** in the error log:
   ```bash
   grep -i "memory\|vram\|cuda" /output/*/error_log.txt
   ```

### V-JEPA2 Dependencies Missing

```bash
# Install encoding dependencies (local machine)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install timm einops opencv-python
```

### Resume Interrupted Rendering

The pipeline **automatically resumes** from where it left off. Just re-run the exact same command:
```bash
# Simply re-run - completed clips are automatically skipped
cd /scripts && python manager_benchmark.py \
    --seed 42 \
    --output_dir /output/friction_dataset_v1 \
    --workers 4 \
    --auto_scale
```

To force regeneration of all clips (ignore existing):
```bash
cd /scripts && python manager_benchmark.py \
    --seed 42 \
    --output_dir /output/friction_dataset_v1 \
    --force_restart
```

To generate only a specific range:
```bash
# Generate clips 5000-9999 only
cd /scripts && python manager_benchmark.py \
    --seed 42 \
    --output_dir /output/friction_dataset_v1 \
    --start_job 5000 \
    --end_job 9999 \
    --workers 4
```

### Re-encode Specific Clips

The encoder skips existing feature maps. Use `--overwrite` to force re-encoding:
```bash
python experiments/scripts/encode_benchmark_vjepa2.py \
    --data_dir experiments/output/friction_dataset_v1 \
    --in_place \
    --overwrite
```

---

## 🧹 Cleanup

```bash
# Remove all outputs (requires sudo due to Docker ownership)
sudo rm -rf experiments/output/*
```

---

## 📚 References

- [Kubric GitHub](https://github.com/google-research/kubric)
- [V-JEPA2 GitHub](https://github.com/facebookresearch/vjepa2)
- [PyBullet](https://pybullet.org/)
