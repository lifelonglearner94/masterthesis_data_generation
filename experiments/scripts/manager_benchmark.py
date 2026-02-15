#!/usr/bin/env python3
"""
manager_benchmark.py - Orchestrator for Kubric Data Generation Pipeline

This script manages the generation of the full dataset by:
1. Running benchmark tests to estimate total compute time
2. Orchestrating parallel/sequential execution of worker scripts
3. Handling errors gracefully and logging failures

Usage:
    # Benchmark mode (test with 5 clips)
    python manager_benchmark.py --test --seed 42 --output_dir ./output/test_run

    # Production mode (full dataset)
    python manager_benchmark.py --seed 42 --output_dir ./output/dataset_v1

Author: Generated for Scientific Data Pipeline
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path for imports when running from workspace
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Ensure dependencies FIRST before importing utils (which needs yaml)
from experiments.scripts.utils import ensure_dependencies
ensure_dependencies()

# Now safe to import the rest of utils (yaml is now available)
from experiments.scripts.utils import (
    EXIT_SUCCESS,
    EXIT_GENERAL_ERROR,
    EXIT_VRAM_ERROR,
    check_gpu_available,
    get_gpu_count,
    get_gpu_vram_usage,
    load_physics_config,
    get_total_clips_from_config,
)

import argparse
import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# VRAM monitoring thresholds
VRAM_HIGH_THRESHOLD = 0.85  # Pause submission above 85% usage
VRAM_LOW_THRESHOLD = 0.70   # Resume submission below 70% usage
VRAM_CHECK_INTERVAL = 5.0   # Seconds between VRAM checks
VRAM_RETRY_DELAY = 30.0     # Seconds to wait before retrying VRAM failure


def get_default_workers() -> int:
    """Calculate default worker count based on CPU cores."""
    cpu_count = os.cpu_count() or 4
    return max(1, cpu_count - 2)


def wait_for_vram_available(threshold: float = VRAM_HIGH_THRESHOLD) -> None:
    """Block until VRAM usage drops below threshold."""
    used, total, fraction = get_gpu_vram_usage()
    if fraction > threshold:
        logger.info(f"VRAM usage high ({fraction*100:.1f}%), waiting for availability...")
        while fraction > VRAM_LOW_THRESHOLD:
            time.sleep(VRAM_CHECK_INTERVAL)
            used, total, fraction = get_gpu_vram_usage()
        logger.info(f"VRAM available ({fraction*100:.1f}%), resuming...")


def is_job_completed(output_dir: Path, job_id: int) -> bool:
    """
    Check if a job has already been completed successfully.

    A job is considered complete if its ground_truth.json file exists,
    since this file is only written at the very end of successful generation.

    Args:
        output_dir: Base output directory
        job_id: The job identifier

    Returns:
        True if the job is already complete, False otherwise
    """
    job_output_dir = output_dir / f"clip_{job_id:05d}"
    ground_truth_file = job_output_dir / "ground_truth.json"
    return ground_truth_file.exists()


def find_completed_jobs(output_dir: Path, job_ids: List[int]) -> set:
    """
    Scan output directory to find all already-completed jobs.

    Args:
        output_dir: Base output directory
        job_ids: List of job IDs to check

    Returns:
        Set of job IDs that are already completed
    """
    completed = set()
    for job_id in job_ids:
        if is_job_completed(output_dir, job_id):
            completed.add(job_id)
    return completed


def build_worker_environment(gpu_id: int = None) -> dict:
    """
    Build environment dict for worker subprocesses with thread limiting.

    Forces NumPy/OpenBLAS/MKL to use single-threaded mode to prevent
    resource contention when running multiple workers.

    Args:
        gpu_id: If specified, sets CUDA_VISIBLE_DEVICES to this GPU index.
                This enables multi-GPU distribution by isolating each worker
                to a specific GPU.

    Returns:
        Environment dictionary for subprocess execution
    """
    env = os.environ.copy()

    # Force single-threaded NumPy/BLAS operations
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["MKL_THREADING_LAYER"] = "GNU"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"

    # Limit Blender's internal threading
    env["BLENDER_CPU_THREADS"] = "1"

    # Multi-GPU support: isolate worker to specific GPU
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    return env


@dataclass
class JobResult:
    """Result from a single job execution."""
    job_id: int
    success: bool
    elapsed_time: float
    output_dir: str
    error_message: Optional[str] = None
    return_code: int = 0  # 0=success, 1=general error, 2=VRAM error (retriable)


@dataclass
class BenchmarkReport:
    """Summary statistics from a benchmark run."""
    num_jobs: int
    successful_jobs: int
    failed_jobs: int
    total_time: float
    avg_time_per_clip: float
    min_time: float
    max_time: float
    estimated_full_run: float  # In hours
    job_results: List[JobResult] = field(default_factory=list)


def run_single_job(
    worker_script: Path,
    output_dir: Path,
    job_id: int,
    seed: int,
    physics_config: Path,
    dry_run: bool = False,
    timeout: int = 600,  # 10 minute timeout per job
    worker_env: Optional[dict] = None,
    no_gif: bool = False,
    gpu_id: Optional[int] = None,
) -> JobResult:
    """
    Execute a single worker script as a subprocess.

    Args:
        worker_script: Path to generate_single_clip.py
        output_dir: Base output directory
        job_id: The job identifier
        seed: Global seed value
        physics_config: Path to physics config YAML
        dry_run: If True, run in dry-run mode
        timeout: Maximum time to wait for job completion
        worker_env: Environment dict for subprocess (with thread limits)
        no_gif: If True, skip GIF preview generation
        gpu_id: If specified, run worker on this specific GPU (multi-GPU support)

    Returns:
        JobResult with execution details
    """
    # Create job-specific output directory
    job_output_dir = output_dir / f"clip_{job_id:05d}"

    # Build command
    cmd = [
        sys.executable,
        str(worker_script),
        "--output_dir", str(job_output_dir),
        "--job_id", str(job_id),
        "--seed", str(seed),
        "--physics_config", str(physics_config),
    ]

    if dry_run:
        cmd.append("--dry_run")

    if no_gif:
        cmd.append("--no_gif")

    # Use thread-limited environment if provided, with optional GPU assignment
    if worker_env is not None:
        env = worker_env
    else:
        env = build_worker_environment(gpu_id=gpu_id)

    start_time = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        elapsed = time.perf_counter() - start_time

        if result.returncode == EXIT_SUCCESS:
            return JobResult(
                job_id=job_id,
                success=True,
                elapsed_time=elapsed,
                output_dir=str(job_output_dir),
                return_code=EXIT_SUCCESS,
            )
        else:
            return JobResult(
                job_id=job_id,
                success=False,
                elapsed_time=elapsed,
                output_dir=str(job_output_dir),
                error_message=result.stderr or result.stdout,
                return_code=result.returncode,
            )

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start_time
        return JobResult(
            job_id=job_id,
            success=False,
            elapsed_time=elapsed,
            output_dir=str(job_output_dir),
            error_message=f"Job timed out after {timeout} seconds",
            return_code=EXIT_GENERAL_ERROR,
        )
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return JobResult(
            job_id=job_id,
            success=False,
            elapsed_time=elapsed,
            output_dir=str(job_output_dir),
            error_message=str(e),
            return_code=EXIT_GENERAL_ERROR,
        )


def run_single_job_with_retry(
    worker_script: Path,
    output_dir: Path,
    job_id: int,
    seed: int,
    physics_config: Path,
    dry_run: bool = False,
    timeout: int = 600,
    worker_env: Optional[dict] = None,
    max_retries: int = 1,
    no_gif: bool = False,
    gpu_id: Optional[int] = None,
) -> JobResult:
    """
    Execute a job with retry logic for VRAM errors.

    VRAM errors (exit code 2) are retried once after a cooldown period.
    Other errors fail immediately.

    Args:
        gpu_id: If specified, run worker on this specific GPU (multi-GPU support)
    """
    result = run_single_job(
        worker_script, output_dir, job_id, seed, physics_config,
        dry_run, timeout, worker_env, no_gif, gpu_id
    )

    if result.success:
        return result

    # Retry only for VRAM errors
    if result.return_code == EXIT_VRAM_ERROR and max_retries > 0:
        logger.warning(f"Job {job_id} failed with VRAM error, retrying in {VRAM_RETRY_DELAY}s...")
        time.sleep(VRAM_RETRY_DELAY)
        return run_single_job_with_retry(
            worker_script, output_dir, job_id, seed, physics_config,
            dry_run, timeout, worker_env, max_retries - 1, no_gif, gpu_id
        )

    return result


def run_benchmark(
    worker_script: Path,
    output_dir: Path,
    seed: int,
    physics_config: Path,
    num_test_jobs: int = 5,
    total_production_jobs: int = 16000,
    dry_run: bool = False,
    test_phase_a1_clips: Optional[int] = None,
    test_phase_b_clips: Optional[int] = None,
    test_phase_a2_clips: Optional[int] = None,
) -> BenchmarkReport:
    """
    Run a benchmark test with a small number of clips.

    Args:
        worker_script: Path to worker script
        output_dir: Base output directory
        seed: Global seed
        physics_config: Path to physics config
        num_test_jobs: Number of test jobs to run
        total_production_jobs: Total jobs in production for time estimation
        dry_run: If True, run in dry-run mode

    Returns:
        BenchmarkReport with timing statistics
    """
    # Allow explicit control over how many samples come from each dataset phase.
    # If any are provided, missing phases default to 0 and the total overrides num_test_jobs.
    if any(v is not None for v in (test_phase_a1_clips, test_phase_b_clips, test_phase_a2_clips)):
        test_phase_a1_clips = int(test_phase_a1_clips or 0)
        test_phase_b_clips = int(test_phase_b_clips or 0)
        test_phase_a2_clips = int(test_phase_a2_clips or 0)

        if test_phase_a1_clips < 0 or test_phase_b_clips < 0 or test_phase_a2_clips < 0:
            raise ValueError("Phase clip counts must be non-negative")

        requested_total = test_phase_a1_clips + test_phase_b_clips + test_phase_a2_clips
        if requested_total <= 0:
            raise ValueError("At least one test clip must be requested")

        if requested_total != num_test_jobs:
            logger.info(
                "Overriding --test_clips=%s with explicit phase counts total=%s (A1=%s, B=%s, A2=%s)",
                num_test_jobs,
                requested_total,
                test_phase_a1_clips,
                test_phase_b_clips,
                test_phase_a2_clips,
            )
        num_test_jobs = requested_total

    logger.info(f"Starting benchmark with {num_test_jobs} test clips...")

    # Sample random job IDs to cover all phases
    config = load_physics_config(physics_config)

    test_job_ids = []
    random.seed(seed)

    # Sample from each phase if available
    if "phase_a1" in config["dataset"]:
        phase_a1 = config["dataset"]["phase_a1"]
        phase_b = config["dataset"]["phase_b"]
        phase_a2 = config["dataset"]["phase_a2"]

        # Distribute samples across phases.
        # Default behavior (legacy): prioritize Phase A_1, then one each from B and A_2.
        if any(v is not None for v in (test_phase_a1_clips, test_phase_b_clips, test_phase_a2_clips)):
            phase_a1_samples = int(test_phase_a1_clips or 0)
            phase_b_samples = int(test_phase_b_clips or 0)
            phase_a2_samples = int(test_phase_a2_clips or 0)
        else:
            phase_a1_samples = max(1, num_test_jobs - 2)
            phase_b_samples = 1
            phase_a2_samples = min(1, num_test_jobs - phase_a1_samples - phase_b_samples)

        # Sample from Phase A_1
        a1_pool = list(range(phase_a1["range"][0], min(phase_a1["range"][1] + 1, phase_a1["range"][0] + 1000)))
        test_job_ids.extend(random.sample(a1_pool, min(phase_a1_samples, len(a1_pool))))

        # Sample from Phase B
        b_pool = list(range(phase_b["range"][0], phase_b["range"][1] + 1))
        if b_pool and phase_b_samples > 0:
            test_job_ids.extend(random.sample(b_pool, min(phase_b_samples, len(b_pool))))

        # Sample from Phase A_2
        a2_pool = list(range(phase_a2["range"][0], phase_a2["range"][1] + 1))
        if a2_pool and phase_a2_samples > 0:
            test_job_ids.extend(random.sample(a2_pool, min(phase_a2_samples, len(a2_pool))))
    else:
        raise ValueError(
            "Missing phase configuration in config. Expected 'phase_a1', 'phase_b', 'phase_a2' in dataset config."
        )

    logger.info(f"Test job IDs: {test_job_ids}")

    # Run test jobs sequentially for accurate timing
    results = []
    overall_start = time.perf_counter()

    for i, job_id in enumerate(test_job_ids):
        logger.info(f"Running test job {i+1}/{num_test_jobs} (ID: {job_id})...")

        result = run_single_job(
            worker_script=worker_script,
            output_dir=output_dir,
            job_id=job_id,
            seed=seed,
            physics_config=physics_config,
            dry_run=dry_run,
        )

        results.append(result)

        status = "✓" if result.success else "✗"
        logger.info(f"  {status} Job {job_id}: {result.elapsed_time:.2f}s")

        if not result.success:
            logger.warning(f"  Error: {result.error_message[:200] if result.error_message else 'Unknown'}")

    overall_time = time.perf_counter() - overall_start

    # Calculate statistics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if successful:
        times = [r.elapsed_time for r in successful]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
    else:
        avg_time = min_time = max_time = 0.0

    # Estimate full run time
    estimated_seconds = avg_time * total_production_jobs
    estimated_hours = estimated_seconds / 3600

    report = BenchmarkReport(
        num_jobs=num_test_jobs,
        successful_jobs=len(successful),
        failed_jobs=len(failed),
        total_time=overall_time,
        avg_time_per_clip=avg_time,
        min_time=min_time,
        max_time=max_time,
        estimated_full_run=estimated_hours,
        job_results=results,
    )

    return report


def print_benchmark_report(report: BenchmarkReport, total_jobs: int):
    """Print a formatted benchmark report to console."""
    print("\n" + "=" * 60)
    print("           BENCHMARK REPORT")
    print("=" * 60)
    print(f"\nTest Configuration:")
    print(f"  • Jobs tested:      {report.num_jobs}")
    print(f"  • Successful:       {report.successful_jobs}")
    print(f"  • Failed:           {report.failed_jobs}")
    print()
    print(f"Timing Statistics (successful jobs):")
    print(f"  • Total test time:  {report.total_time:.2f} seconds")
    print(f"  • Average per clip: {report.avg_time_per_clip:.2f} seconds")
    print(f"  • Min time:         {report.min_time:.2f} seconds")
    print(f"  • Max time:         {report.max_time:.2f} seconds")
    print()
    print("─" * 60)
    print(f"  📊 PRODUCTION ESTIMATE for {total_jobs:,} clips:")
    print()
    print(f"     Average time per clip: {report.avg_time_per_clip:.2f} seconds")
    print(f"     Estimated total time:  {report.estimated_full_run:.1f} hours")
    print(f"                           ({report.estimated_full_run/24:.1f} days)")
    print("─" * 60)

    if report.failed_jobs > 0:
        print("\n⚠️  FAILED JOBS:")
        for result in report.job_results:
            if not result.success:
                print(f"  • Job {result.job_id}: {result.error_message[:100] if result.error_message else 'Unknown error'}")

    print("=" * 60 + "\n")


def run_production(
    worker_script: Path,
    output_dir: Path,
    seed: int,
    physics_config: Path,
    start_job: int = 0,
    end_job: int = 16999,
    dry_run: bool = False,
    max_workers: int = 1,
    auto_scale: bool = False,
    force_restart: bool = False,
    no_gif: bool = False,
    multi_gpu: bool = False,
) -> None:
    """
    Run the full production dataset generation with pipeline saturation.

    Uses a bounded process pool pattern to keep workers saturated while
    preventing resource exhaustion. Each worker runs with thread-limited
    environment to avoid CPU contention.

    RESUME SUPPORT: Automatically detects and skips already-completed jobs
    by checking for the existence of ground_truth.json in each clip directory.
    This allows the pipeline to resume from where it left off after interruption.

    MULTI-GPU SUPPORT: When multi_gpu=True, workers are distributed across
    available GPUs in round-robin fashion using CUDA_VISIBLE_DEVICES.

    Args:
        worker_script: Path to worker script
        output_dir: Base output directory
        seed: Global seed
        physics_config: Path to physics config
        start_job: Starting job ID (inclusive)
        end_job: Ending job ID (inclusive)
        dry_run: If True, run in dry-run mode
        max_workers: Number of parallel workers
        auto_scale: If True, pause submission when VRAM is high
        force_restart: If True, ignore existing completed jobs and regenerate all
        no_gif: If True, skip GIF preview generation (faster for production)
        multi_gpu: If True, distribute workers across multiple GPUs
    """
    total_jobs_requested = end_job - start_job + 1
    logger.info(f"Production run requested: jobs {start_job} to {end_job} ({total_jobs_requested:,} total)")
    logger.info(f"Parallel workers: {max_workers} (auto_scale={auto_scale})")

    # Multi-GPU setup
    num_gpus = get_gpu_count() if multi_gpu else 1
    if multi_gpu:
        if num_gpus > 1:
            logger.info(f"Multi-GPU mode enabled: distributing workers across {num_gpus} GPUs")
        elif num_gpus == 1:
            logger.warning("Multi-GPU requested but only 1 GPU found. Running on single GPU.")
        else:
            logger.warning("Multi-GPU requested but no GPUs found. Workers will use CPU.")
            num_gpus = 1  # Avoid division by zero

    # Build full job list
    all_job_ids = list(range(start_job, end_job + 1))

    # RESUME SUPPORT: Check for already completed jobs
    if not force_restart:
        logger.info("Scanning for previously completed jobs (resume mode)...")
        completed_jobs = find_completed_jobs(output_dir, all_job_ids)
        num_completed = len(completed_jobs)

        if num_completed > 0:
            logger.info(f"✓ Found {num_completed:,} already completed jobs - skipping these")
            job_ids = [j for j in all_job_ids if j not in completed_jobs]
            logger.info(f"  Resuming with {len(job_ids):,} remaining jobs")
        else:
            logger.info("No previously completed jobs found - starting fresh")
            job_ids = all_job_ids
    else:
        logger.warning("Force restart enabled - regenerating ALL jobs (ignoring existing)")
        job_ids = all_job_ids

    total_jobs = len(job_ids)

    if total_jobs == 0:
        logger.info("All jobs already completed! Nothing to do.")
        return

    logger.info(f"Starting generation of {total_jobs:,} clips...")

    # Create error log file
    error_log_path = output_dir / "error_log.txt"

    # Track progress with thread-safe counter
    progress_lock = threading.Lock()
    completed = [0]  # Mutable container for closure
    failed = [0]
    start_time = time.perf_counter()

    # Rolling window for recent timing data (for more accurate ETA)
    recent_times = []  # List of (completion_time, elapsed_since_start)
    ROLLING_WINDOW_SIZE = 50  # Number of recent clips to use for ETA
    ETA_REPORT_INTERVAL = 500  # Print detailed ETA every N clips
    last_eta_report_time = [start_time]  # Last time we printed a detailed ETA

    # Create global metadata (only if not resuming or first run)
    metadata_path = output_dir / "metadata.yaml"
    if not metadata_path.exists():
        save_global_metadata(output_dir, seed, physics_config, start_job, end_job)
    else:
        logger.info(f"Metadata already exists at {metadata_path}, preserving original")

    # Build thread-limited environment once (without GPU assignment - will be per-job for multi-GPU)
    worker_env = build_worker_environment() if not multi_gpu else None

    # GPU slot counter for round-robin distribution
    gpu_slot_counter = [0]  # Mutable container for closure

    def get_next_gpu_id() -> Optional[int]:
        """Get the next GPU ID for round-robin distribution."""
        if not multi_gpu or num_gpus <= 1:
            return None
        gpu_id = gpu_slot_counter[0] % num_gpus
        gpu_slot_counter[0] += 1
        return gpu_id

    # Note: job_ids is already filtered above for resume support

    def process_result(result: JobResult) -> None:
        """Thread-safe result processing with rolling-window ETA estimation."""
        with progress_lock:
            completed[0] += 1
            current = completed[0]
            now = time.perf_counter()
            elapsed = now - start_time

            if result.success:
                # Track timing for rolling window ETA
                recent_times.append((now, result.elapsed_time))
                # Keep only the last ROLLING_WINDOW_SIZE entries
                while len(recent_times) > ROLLING_WINDOW_SIZE:
                    recent_times.pop(0)

                # Basic progress update every 100 clips
                if current % 100 == 0 or current == 1:
                    rate = current / elapsed if elapsed > 0 else 0
                    remaining = (total_jobs - current) / rate / 3600 if rate > 0 else 0
                    logger.info(
                        f"Progress: {current}/{total_jobs} "
                        f"({100*current/total_jobs:.1f}%) - "
                        f"Rate: {rate:.2f} clips/s - "
                        f"Est. remaining: {remaining:.1f}h"
                    )

                # Detailed ETA report every ETA_REPORT_INTERVAL clips or every 30 minutes
                time_since_last_report = now - last_eta_report_time[0]
                should_report = (
                    current % ETA_REPORT_INTERVAL == 0 or
                    time_since_last_report >= 1800  # 30 minutes
                )

                if should_report and len(recent_times) >= 10:
                    last_eta_report_time[0] = now

                    # Calculate recent rate from rolling window
                    recent_elapsed = recent_times[-1][0] - recent_times[0][0]
                    recent_count = len(recent_times)
                    recent_rate = (recent_count - 1) / recent_elapsed if recent_elapsed > 0 else 0

                    # Calculate average time per clip from recent window
                    recent_avg_time = sum(t[1] for t in recent_times) / len(recent_times)

                    # Overall rate for comparison
                    overall_rate = current / elapsed if elapsed > 0 else 0

                    # ETA based on recent rate (more accurate for current conditions)
                    remaining_jobs = total_jobs - current
                    eta_recent_hours = remaining_jobs / recent_rate / 3600 if recent_rate > 0 else 0
                    eta_overall_hours = remaining_jobs / overall_rate / 3600 if overall_rate > 0 else 0

                    # Calculate completion time
                    eta_datetime = datetime.now() + timedelta(hours=eta_recent_hours)

                    # Print detailed ETA report
                    logger.info("")
                    logger.info("─" * 60)
                    logger.info("📊 DETAILED ETA REPORT (based on recent performance)")
                    logger.info("─" * 60)
                    logger.info(f"  Progress:        {current:,} / {total_jobs:,} clips ({100*current/total_jobs:.1f}%)")
                    logger.info(f"  Elapsed time:    {elapsed/3600:.2f} hours")
                    logger.info(f"  ")
                    logger.info(f"  Recent rate:     {recent_rate:.3f} clips/sec ({recent_avg_time:.1f}s per clip)")
                    logger.info(f"  Overall rate:    {overall_rate:.3f} clips/sec")
                    logger.info(f"  ")
                    logger.info(f"  Remaining:       {remaining_jobs:,} clips")
                    logger.info(f"  ETA (recent):    {eta_recent_hours:.1f} hours ({eta_recent_hours/24:.1f} days)")
                    logger.info(f"  ETA (overall):   {eta_overall_hours:.1f} hours ({eta_overall_hours/24:.1f} days)")
                    logger.info(f"  Expected finish: {eta_datetime.strftime('%Y-%m-%d %H:%M')}")
                    logger.info("─" * 60)
                    logger.info("")
            else:
                failed[0] += 1
                log_error(error_log_path, result)
                logger.warning(
                    f"Job {result.job_id} failed (exit={result.return_code}): "
                    f"{result.error_message[:100] if result.error_message else 'Unknown'}"
                )

    if max_workers > 1:
        # Parallel execution with bounded submission
        # Use ThreadPoolExecutor to manage subprocess workers
        # This allows VRAM monitoring between submissions
        logger.info(f"Using parallel execution with {max_workers} workers")
        if multi_gpu and num_gpus > 1:
            logger.info(f"Multi-GPU: distributing jobs across GPUs 0-{num_gpus-1}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            pending_count = 0
            job_iter = iter(job_ids)
            jobs_remaining = True

            # Submit initial batch up to max_workers
            for _ in range(max_workers):
                try:
                    job_id = next(job_iter)

                    # VRAM-aware submission throttling
                    if auto_scale:
                        wait_for_vram_available()

                    # Get GPU assignment for this job (round-robin for multi-GPU)
                    gpu_id = get_next_gpu_id()

                    future = executor.submit(
                        run_single_job_with_retry,
                        worker_script,
                        output_dir,
                        job_id,
                        seed,
                        physics_config,
                        dry_run,
                        600,  # timeout
                        worker_env,
                        1,  # max_retries
                        no_gif,
                        gpu_id,
                    )
                    futures[future] = job_id
                    pending_count += 1
                except StopIteration:
                    jobs_remaining = False
                    break

            # Process completions and submit new jobs (bounded queue pattern)
            while futures:
                # Wait for at least one future to complete
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)

                if not done_futures:
                    # No futures done yet, wait a bit
                    time.sleep(0.1)
                    continue

                # Process completed futures
                for future in done_futures:
                    job_id = futures.pop(future)
                    pending_count -= 1

                    try:
                        result = future.result()
                        process_result(result)
                    except Exception as e:
                        with progress_lock:
                            failed[0] += 1
                        logger.error(f"Job {job_id} raised exception: {e}")

                    # Submit a new job if any remaining
                    if jobs_remaining:
                        try:
                            next_job_id = next(job_iter)

                            # VRAM-aware throttling
                            if auto_scale:
                                wait_for_vram_available()

                            # Get GPU assignment for this job (round-robin for multi-GPU)
                            gpu_id = get_next_gpu_id()

                            new_future = executor.submit(
                                run_single_job_with_retry,
                                worker_script,
                                output_dir,
                                next_job_id,
                                seed,
                                physics_config,
                                dry_run,
                                600,
                                worker_env,
                                1,  # max_retries
                                no_gif,
                                gpu_id,
                            )
                            futures[new_future] = next_job_id
                            pending_count += 1
                        except StopIteration:
                            jobs_remaining = False
    else:
        # Sequential execution (standard for single GPU)
        logger.info("Using sequential execution (single worker)")
        for job_id in job_ids:
            gpu_id = get_next_gpu_id() if multi_gpu else None
            result = run_single_job_with_retry(
                worker_script=worker_script,
                output_dir=output_dir,
                job_id=job_id,
                seed=seed,
                physics_config=physics_config,
                dry_run=dry_run,
                worker_env=worker_env,
                no_gif=no_gif,
                gpu_id=gpu_id,
            )
            process_result(result)

    # Final summary
    total_time = time.perf_counter() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"PRODUCTION RUN COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total clips processed: {completed[0]}")
    logger.info(f"Successful: {completed[0] - failed[0]}")
    logger.info(f"Failed: {failed[0]}")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Average time per clip: {total_time/completed[0]:.2f}s" if completed[0] > 0 else "N/A")

    if failed[0] > 0:
        logger.warning(f"Check {error_log_path} for failure details")


def log_error(error_log_path: Path, result: JobResult):
    """Append error information to the error log file."""
    with open(error_log_path, "a") as f:
        timestamp = datetime.now().isoformat()
        f.write(f"\n{'='*60}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Job ID: {result.job_id}\n")
        f.write(f"Output Dir: {result.output_dir}\n")
        f.write(f"Elapsed Time: {result.elapsed_time:.2f}s\n")
        f.write(f"Error:\n{result.error_message}\n")


def save_global_metadata(
    output_dir: Path,
    seed: int,
    physics_config: Path,
    start_job: int,
    end_job: int
):
    """Save global dataset metadata."""
    config = load_physics_config(physics_config)

    metadata = {
        "dataset_name": output_dir.name,
        "generation_timestamp": datetime.now().isoformat(),
        "global_seed": seed,
        "job_range": {
            "start": start_job,
            "end": end_job,
            "total": end_job - start_job + 1,
        },
        "physics_config": config,
        "generator_version": "1.0.0",
    }

    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved global metadata to: {metadata_path}")


def main():
    """Main entry point for the orchestrator."""
    parser = argparse.ArgumentParser(
        description="Orchestrate Kubric dataset generation with benchmarking support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark test (5 clips)
  python manager_benchmark.py --test --seed 42 --output_dir ./output/benchmark

  # Run full production (all 16000 clips) - auto-resumes if interrupted
  python manager_benchmark.py --seed 42 --output_dir ./output/dataset_v1

  # Resume an interrupted run (just re-run the same command)
  python manager_benchmark.py --seed 42 --output_dir ./output/dataset_v1

  # Force restart from scratch (ignore existing completed clips)
  python manager_benchmark.py --seed 42 --output_dir ./output/dataset_v1 --force_restart

  # Run specific range of jobs
  python manager_benchmark.py --seed 42 --output_dir ./output/batch1 \\
      --start_job 0 --end_job 4999
        """
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base directory for all output clips"
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Global seed for reproducibility"
    )

    parser.add_argument(
        "--physics_config",
        type=str,
        default=None,
        help="Path to physics config YAML (default: ../config/physics_config.yaml)"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in benchmark/test mode with only 5 clips"
    )

    parser.add_argument(
        "--test_clips",
        type=int,
        default=5,
        help="Number of clips to generate in test mode (default: 5)"
    )

    parser.add_argument(
        "--test_a1_clips",
        type=int,
        default=None,
        help="In --test mode: number of clips sampled from Phase A_1 (overrides distribution if set)"
    )

    parser.add_argument(
        "--test_b_clips",
        type=int,
        default=None,
        help="In --test mode: number of clips sampled from Phase B (overrides distribution if set)"
    )

    parser.add_argument(
        "--test_a2_clips",
        type=int,
        default=None,
        help="In --test mode: number of clips sampled from Phase A_2 (overrides distribution if set)"
    )

    parser.add_argument(
        "--start_job",
        type=int,
        default=0,
        help="Starting job ID for production (default: 0)"
    )

    parser.add_argument(
        "--end_job",
        type=int,
        default=None,
        help="Ending job ID for production (default: derived from config total_clips - 1)"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run without actual rendering (metadata only)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: CPU_COUNT-2 = {get_default_workers()})"
    )

    parser.add_argument(
        "--auto_scale",
        action="store_true",
        help="Enable VRAM-aware auto-scaling (pause submission when GPU memory is high)"
    )

    parser.add_argument(
        "--force_restart",
        action="store_true",
        help="Ignore existing completed jobs and regenerate all (disables resume)"
    )

    parser.add_argument(
        "--no_gif",
        action="store_true",
        help="Skip GIF preview generation (recommended for production runs)"
    )

    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Enable multi-GPU mode: distribute workers across all available GPUs in round-robin fashion"
    )
    

    args = parser.parse_args()

    # Set default workers if not specified
    if args.workers is None:
        args.workers = get_default_workers()

    # Resolve paths
    script_dir = Path(__file__).parent.resolve()
    worker_script = script_dir / "generate_single_clip.py"

    if args.physics_config:
        physics_config = Path(args.physics_config).resolve()
    else:
        physics_config = script_dir.parent / "config" / "physics_config.yaml"

    output_dir = Path(args.output_dir).resolve()

    # Validate paths
    if not worker_script.exists():
        logger.error(f"Worker script not found: {worker_script}")
        sys.exit(1)

    if not physics_config.exists():
        logger.error(f"Physics config not found: {physics_config}")
        sys.exit(1)

    # Set default end_job from config if not specified
    if args.end_job is None:
        config = load_physics_config(physics_config)
        total_clips = get_total_clips_from_config(config)
        if total_clips > 0:
            args.end_job = total_clips - 1
            logger.info(f"Derived end_job={args.end_job} from config total_clips={total_clips}")
        else:
            args.end_job = 16999  # Fallback default
            logger.warning(f"Could not determine total_clips from config, using default end_job={args.end_job}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler for logging
    log_file = output_dir / "generation.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Worker script: {worker_script}")
    logger.info(f"Physics config: {physics_config}")
    logger.info(f"Global seed: {args.seed}")

    # Check GPU
    gpu_available = check_gpu_available()
    if not gpu_available:
        logger.warning("No GPU detected. Rendering will be slower.")
        if not args.dry_run:
            response = input("Continue without GPU? [y/N]: ")
            if response.lower() != 'y':
                logger.info("Aborted by user.")
                sys.exit(0)

    if args.test:
        # Benchmark mode
        logger.info("Running in BENCHMARK mode")

        # Load config to get total job count
        config = load_physics_config(physics_config)
        total_production = get_total_clips_from_config(config)
        if total_production <= 0:
            logger.warning("Could not infer total production jobs from config; using end_job-start_job+1 fallback")
            total_production = args.end_job - args.start_job + 1

        report = run_benchmark(
            worker_script=worker_script,
            output_dir=output_dir,
            seed=args.seed,
            physics_config=physics_config,
            num_test_jobs=args.test_clips,
            total_production_jobs=total_production,
            dry_run=args.dry_run,
            test_phase_a1_clips=args.test_a1_clips,
            test_phase_b_clips=args.test_b_clips,
            test_phase_a2_clips=args.test_a2_clips,
        )

        print_benchmark_report(report, total_production)

        # Save benchmark report
        report_path = output_dir / "benchmark_report.json"
        with open(report_path, "w") as f:
            json.dump({
                "num_jobs": report.num_jobs,
                "successful_jobs": report.successful_jobs,
                "failed_jobs": report.failed_jobs,
                "total_time": report.total_time,
                "avg_time_per_clip": report.avg_time_per_clip,
                "min_time": report.min_time,
                "max_time": report.max_time,
                "estimated_full_run_hours": report.estimated_full_run,
            }, f, indent=2)
        logger.info(f"Benchmark report saved to: {report_path}")

    else:
        # Production mode
        logger.info("Running in PRODUCTION mode")

        run_production(
            worker_script=worker_script,
            output_dir=output_dir,
            seed=args.seed,
            physics_config=physics_config,
            start_job=args.start_job,
            end_job=args.end_job,
            dry_run=args.dry_run,
            max_workers=args.workers,
            auto_scale=args.auto_scale,
            force_restart=args.force_restart,
            no_gif=args.no_gif,
            multi_gpu=args.multi_gpu,
        )


if __name__ == "__main__":
    main()
