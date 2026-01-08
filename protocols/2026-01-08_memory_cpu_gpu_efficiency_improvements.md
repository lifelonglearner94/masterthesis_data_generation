# 2026-01-08 — Memory + CPU/GPU Efficiency Improvements

Date: 2026-01-08

## Purpose
Document the implementation changes introduced in the latest commit to improve end-to-end throughput and stability (CPU contention, GPU VRAM pressure, and orchestration robustness) for the Kubric data generation and V-JEPA2 encoding pipeline.

## Revision Range
Comparison range:
- Newer: `9896a15` ("enhanced memory and cpu/gpu efficiency by far")
- Older: `d6b064f`

Files changed in this revision:
- Modified
  - `experiments/scripts/manager_benchmark.py`
  - `experiments/scripts/generate_single_clip.py`
  - `experiments/scripts/encode_benchmark_vjepa2.py`
  - `README.md`
- Added
  - `uv.lock`

## Summary of Key Changes
1. **VRAM-aware job submission** was introduced in the orchestrator to reduce GPU OOMs and improve long-run stability.
2. **Thread limiting for CPU libraries (BLAS/OpenMP/etc.) and Blender** was centralized in the manager and reinforced in the worker to avoid multi-worker CPU oversubscription.
3. **Retry logic for VRAM-related failures** was added to re-attempt jobs that fail specifically due to GPU memory exhaustion.
4. **Parallel scheduling was refactored to a bounded-queue pattern** to keep a fixed number of active workers while allowing VRAM checks between submissions.
5. **V-JEPA2 encoding was made more reusable and dependency-safe**, including a cached encoder for repeated calls and clearer dependency failures.
6. **Documentation updates** aligned CLI defaults and recommended flags (`--workers`, `--auto_scale`, `uv run`) with the new behavior.

## Detailed Changes

### A. Orchestrator (`experiments/scripts/manager_benchmark.py`)

#### A1. Default worker count (CPU aware)
- Added `get_default_workers()` which computes a default parallelism level of `max(1, CPU_COUNT - 2)`.
- CLI change: `--workers` now defaults to `None`, and is set after parsing using `get_default_workers()`.

**Rationale:** keep the machine responsive and reduce contention by leaving CPU headroom for OS, Blender, and I/O.

#### A2. Thread-limited worker environment
- Added `build_worker_environment()` to inject:
  - `OMP_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `MKL_THREADING_LAYER=GNU`
  - `NUMEXPR_NUM_THREADS=1`
  - `VECLIB_MAXIMUM_THREADS=1`
  - `BLENDER_CPU_THREADS=1`
- The orchestrator now builds this environment once and passes it into all worker subprocesses.

**Rationale:** prevent each parallel worker from spawning multiple BLAS/OpenMP threads and causing CPU thrash.

#### A3. VRAM monitoring + throttled submission
- Introduced VRAM monitoring via `nvidia-smi`:
  - `get_gpu_vram_usage()` parses used/total MB.
  - `wait_for_vram_available()` blocks submission when VRAM usage exceeds a high watermark.
- Thresholds/intervals:
  - Pause above `VRAM_HIGH_THRESHOLD = 0.85`
  - Resume below `VRAM_LOW_THRESHOLD = 0.70`
  - Check every `VRAM_CHECK_INTERVAL = 5.0s`

**Rationale:** avoid piling additional GPU work (or GPU-influenced stages) when memory is already high.

#### A4. VRAM-aware auto-scaling flag
- Added CLI flag `--auto_scale` to enable the VRAM-aware throttling behavior.

#### A5. Retry logic for VRAM failures
- Added `run_single_job_with_retry()`:
  - Detects VRAM failures by exit code `EXIT_VRAM_ERROR = 2`.
  - Retries once by default (`max_retries=1`) after `VRAM_RETRY_DELAY = 30s`.

**Rationale:** transient OOM conditions can resolve after some GPU memory is freed.

#### A6. Parallel execution refactor: bounded queue pattern
- The parallel execution path was refactored to a bounded submission model:
  - Uses `ThreadPoolExecutor` to manage the subprocess-based workers.
  - Maintains a fixed number of in-flight futures (`max_workers`).
  - On each completion, processes results and submits the next job (optionally gated by VRAM availability).
- Progress accounting was made thread-safe via a `threading.Lock()` and mutable counters (`completed=[0]`, `failed=[0]`).

**Rationale:**
- Keep a stable level of concurrency.
- Create a control point between submissions where VRAM can be checked.
- Avoid unbounded queues / resource spikes.

### B. Worker (`experiments/scripts/generate_single_clip.py`)

#### B1. Early thread limiting before NumPy import
- Added `_apply_thread_limits()` at the top of the file (before importing NumPy/BLAS).
- Behavior:
  - If the manager already set `OMP_NUM_THREADS`, the worker respects it.
  - Otherwise, the worker sets conservative defaults (`*NUM_THREADS=2`) for standalone execution.

**Rationale:** thread env vars must be set before importing NumPy/BLAS to reliably affect thread pools.

#### B2. Blender thread control
- Added `get_blender_threads()` and `log_thread_configuration()` helpers.
- Uses `BLENDER_CPU_THREADS` env var to control Blender CPU threading.

#### B3. VRAM error classification + exit code
- Added VRAM error patterns and helper `is_vram_error()`.
- On exceptions, worker now selects exit code:
  - `EXIT_VRAM_ERROR = 2` if a VRAM/OOM pattern is detected
  - `EXIT_GENERAL_ERROR = 1` otherwise

**Rationale:** enables orchestrator-level retry and better error attribution.

### C. Encoder (`experiments/scripts/encode_benchmark_vjepa2.py`)

#### C1. Safer dependency handling
- Added `_ensure_light_dependencies()` which auto-installs only lightweight pure-python deps (e.g., `numpy`, `pillow`).
- Added `_require_imports()` to fail fast with clear instructions for heavy deps:
  - `torch`, `torchvision`, `timm`, `einops`, plus `Pillow`.

**Rationale:** reduce "mystery import" failures while avoiding auto-installing GPU-heavy stacks.

#### C2. Cached encoder for repeated encoding calls
- Introduced singleton cache `_ENCODER_CACHE` and `get_encoder()`:
  - Loads and keeps `model`, `preprocessor`, `device` alive across calls.
  - Enables `encode_single_clip()` to be called repeatedly without reloading the model each time.

**Rationale:** eliminates repeated `torch.hub.load()` overhead and reduces memory churn during iterative workflows.

#### C3. Encoding output and batching behaviors
- Maintains batch encoding flow in `main()` with:
  - `--in_place` mode (`clip_*/feature_maps/vjepa2_vitl16.npy`)
  - batching with `--batch_size` (also used for multi-GPU `DataParallel`)
  - timing breakdown (I/O, preprocess, encode, save) and optional JSON report.

### D. Documentation (`README.md`)
- Updated usage examples and CLI reference to reflect:
  - `--workers` default of `CPU_COUNT-2`
  - new `--auto_scale` flag and its intent (VRAM-aware throttling)
  - recommended invocation via `uv run` for encoding
  - explicit note that `--shm-size=8g` is required for parallel execution in Docker

### E. Dependency Lockfile (`uv.lock`)
- Added `uv.lock` to capture a reproducible Python dependency resolution for the project (including optional dev tooling).

## Expected Impact
- **Reduced CPU oversubscription** when running multiple clip-generation workers in parallel.
- **Fewer GPU VRAM-related failures** due to proactive submission throttling.
- **Faster recovery from transient VRAM errors** via targeted retry.
- **Lower overhead for repeated encoding calls** due to cached encoder/preprocessor.

## Validation / Reproduction Notes
Suggested minimal checks:
1. Render test run in Docker:
   - `python manager_benchmark.py --test --test_clips 5 --seed 42 --output_dir /output/test_5clips`
2. Production-like run (small range) in Docker:
   - `python manager_benchmark.py --seed 42 --output_dir /output/tmp --start_job 0 --end_job 99 --workers 4 --auto_scale`
3. Encode locally:
   - `uv run experiments/scripts/encode_benchmark_vjepa2.py --data_dir experiments/output/test_5clips --in_place --batch_size 4`

## Known Constraints / Assumptions
- VRAM monitoring depends on `nvidia-smi` availability. If absent or failing, VRAM usage queries return zeros and throttling may not activate meaningfully.
- `DataParallel` is only used when multiple CUDA devices are visible and `--batch_size > 1`.
- Thread limiting values (`1` in manager) prioritize stability and predictable multi-worker scaling over per-worker peak CPU performance.
