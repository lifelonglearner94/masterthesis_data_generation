# Kubric Data Generation Pipeline

A scientific data generation pipeline for creating synthetic video datasets of friction physics experiments using [Kubric](https://github.com/google-research/kubric), PyBullet, and Blender.

## 🎯 Overview

This pipeline generates 256×256 video clips of a cube receiving force impulses on surfaces with varying friction coefficients. Each clip includes:

- **RGB video** (32 frames @ 20 FPS)
- **Optical flow** (NumPy format)
- **Segmentation masks**
- **Ground truth metadata** (friction, force vectors, mass, etc.)

## 📁 Project Structure

```
experiments/
├── config/
│   └── physics_config.yaml    # Physics parameter ranges
├── scripts/
│   ├── generate_single_clip.py   # Worker: generates one clip
│   ├── manager_benchmark.py      # Orchestrator: manages full dataset
│   └── run_docker.sh             # Docker launcher with GPU
└── output/
    └── [generated datasets]
```

---

## 🚀 Quick Start: Two Ways to Use the Pipeline

### Prerequisites

1. **Docker** with NVIDIA Container Toolkit (for GPU acceleration)
2. **NVIDIA GPU** with CUDA support (optional, CPU works but is slower)

---

## Option A: Benchmark Test (5 Clips + Time Estimation)

**Purpose:** Quick validation that the pipeline works and estimation of full dataset generation time.

### Step 1: Start Docker Container

```bash
# From project root directory
cd /path/to/masterthesis_data_generation

# With GPU (recommended)
docker run --rm -it \
    --gpus all \
    -v "$(pwd)/experiments/scripts:/scripts" \
    -v "$(pwd)/experiments/config:/config" \
    -v "$(pwd)/experiments/output:/output" \
    kubricdockerhub/kubruntu:latest \
    /bin/bash

# Without GPU (CPU-only, slower)
docker run --rm -it \
    -v "$(pwd)/experiments/scripts:/scripts" \
    -v "$(pwd)/experiments/config:/config" \
    -v "$(pwd)/experiments/output:/output" \
    kubricdockerhub/kubruntu:latest \
    /bin/bash
```

### Step 2: Run Benchmark Test (Inside Container)

```bash
cd /scripts && python manager_benchmark.py \
    --test \
    --seed 42 \
    --output_dir /output/benchmark_test \
    --test_clips 5
```

### Expected Output

```
============================================================
           BENCHMARK REPORT
============================================================

Test Configuration:
  • Jobs tested:      5
  • Successful:       5
  • Failed:           0

Timing Statistics (successful jobs):
  • Average per clip: 45.23 seconds
  • Min time:         42.10 seconds
  • Max time:         48.50 seconds

────────────────────────────────────────────────────────────
  📊 PRODUCTION ESTIMATE for 16,000 clips:

     Average time per clip: 45.23 seconds
     Estimated total time:  201.0 hours
                           (8.4 days)
────────────────────────────────────────────────────────────
```

---

## Option B: Full Production Run (16,000 Clips)

**Purpose:** Generate the complete dataset for training.

### Step 1: Start Docker Container

```bash
# From project root directory
cd /path/to/masterthesis_data_generation

# With GPU (recommended)
docker run --rm -it \
    --gpus all \
    -v "$(pwd)/experiments/scripts:/scripts" \
    -v "$(pwd)/experiments/config:/config" \
    -v "$(pwd)/experiments/output:/output" \
    kubricdockerhub/kubruntu:latest \
    /bin/bash

# Without GPU (CPU-only, slower)
docker run --rm -it \
    -v "$(pwd)/experiments/scripts:/scripts" \
    -v "$(pwd)/experiments/config:/config" \
    -v "$(pwd)/experiments/output:/output" \
    kubricdockerhub/kubruntu:latest \
    /bin/bash
```

### Step 2: Run Full Generation (Inside Container)

```bash
cd /scripts && python manager_benchmark.py \
    --seed 42 \
    --output_dir /output/friction_dataset_v1
```

### Generate Specific Range (Optional)

For batch processing or resuming interrupted runs:

```bash
# Generate clips 0-4999 only
cd /scripts && python manager_benchmark.py \
    --seed 42 \
    --output_dir /output/batch_1 \
    --start_job 0 \
    --end_job 4999
```

---

## 🧹 Cleanup Output Folder

Files generated inside Docker are owned by root. Use sudo to clean up:

```bash
sudo rm -rf experiments/output/*
```

---

## 📊 Output Structure

Each generated clip follows this structure:

```
output/dataset_name/
├── benchmark_report.json   # Timing and success statistics
├── generation.log          # Full generation log
├── clip_00000/
│   ├── rgb/
│   │   ├── frame_00000.png
│   │   └── ...
│   ├── flow/
│   │   ├── flow_00000.npy
│   │   └── ...
│   ├── segmentation/
│   │   ├── seg_00000.png
│   │   └── ...
│   ├── video.mp4           # All-Intra encoded (lossless)
│   └── ground_truth.json
├── clip_00001/
└── ...
```

---

## 🔧 Configuration

Edit `config/physics_config.yaml` to customize physics parameters:

| Parameter | Normal Range | Slippery (OOD) Range |
|-----------|--------------|----------------------|
| Friction coefficient | 0.5 – 0.9 | 0.01 – 0.15 |
| Force magnitude | 5 – 25 N | 5 – 25 N |
| Object mass | 0.5 – 2.0 kg | 0.5 – 2.0 kg |

---

## 📋 CLI Reference

### manager_benchmark.py

| Flag | Description |
|------|-------------|
| `--output_dir` | Base directory for all clips |
| `--seed` | Global seed for reproducibility |
| `--test` | Run benchmark mode (default 5 clips) |
| `--test_clips N` | Number of clips in test mode |
| `--start_job N` | Starting job ID for production |
| `--end_job N` | Ending job ID for production |
| `--dry_run` | Metadata only (no rendering) |

---

## ⚠️ Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Files Owned by Root

Docker creates files as root. Clean up with:
```bash
sudo rm -rf experiments/output/*
```

---

## 📚 References

- [Kubric GitHub](https://github.com/google-research/kubric)
- [PyBullet](https://pybullet.org/)
