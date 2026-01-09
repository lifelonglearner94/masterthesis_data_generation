#!/usr/bin/env python3
"""
utils.py - Shared utilities for Kubric data generation pipeline.

This module contains common utilities used by both the worker script
(generate_single_clip.py) and the orchestrator (manager_benchmark.py).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple

# yaml is imported lazily after ensure_dependencies() is called
yaml = None  # type: ignore

# ============================================================================
# Exit codes (shared between worker and manager)
# ============================================================================
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_VRAM_ERROR = 2  # GPU memory exhaustion - retriable after cooldown


# ============================================================================
# Dependency management
# ============================================================================


def _has_module(import_name: str) -> bool:
    """Check if a module is importable."""
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def _try_apt_install(apt_pkg: str) -> bool:
    """Try to install a package via apt-get."""
    if shutil.which("apt-get") is None:
        return False
    try:
        subprocess.run(
            ["apt-get", "update"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
        subprocess.run(
            ["apt-get", "install", "-y", apt_pkg],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
        return True
    except Exception:
        return False


def _try_pip_install(pip_pkg: str) -> bool:
    """Try to install a package via pip."""
    env = dict(os.environ)
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PIP_DEFAULT_TIMEOUT", "120")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--retries",
        "10",
        "--default-timeout",
        "120",
        pip_pkg,
    ]
    try:
        subprocess.run(cmd, check=True, env=env, timeout=600)
        return True
    except Exception:
        return False


def ensure_dependencies() -> None:
    """
    Ensure minimal runtime dependencies exist.

    This function is used by scripts running inside the Kubric Docker image
    where network access to PyPI can be flaky. Prefers apt packages when
    available and falls back to pip with higher timeouts/retries.
    """
    global yaml

    # pyyaml
    if not _has_module("yaml"):
        print("Dependency missing: pyyaml (yaml). Installing...")
        if not (_try_apt_install("python3-yaml") or _try_pip_install("pyyaml")):
            raise RuntimeError("Failed to install dependency: pyyaml")

    # Import yaml now that it's available
    import yaml as _yaml
    yaml = _yaml

    # numpy
    if not _has_module("numpy"):
        print("Dependency missing: numpy. Installing...")
        if not (_try_apt_install("python3-numpy") or _try_pip_install("numpy")):
            raise RuntimeError("Failed to install dependency: numpy")


# ============================================================================
# GPU utilities
# ============================================================================


def check_gpu_available() -> bool:
    """
    Check if NVIDIA GPU is available via nvidia-smi.

    Returns:
        True if GPU is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_gpu_vram_usage() -> Tuple[float, float, float]:
    """
    Query GPU VRAM usage via nvidia-smi.

    Returns:
        Tuple of (used_mb, total_mb, usage_fraction) or (0, 0, 0) if unavailable
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse "1234, 8192" format
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 2:
                used = float(parts[0])
                total = float(parts[1])
                fraction = used / total if total > 0 else 0
                return used, total, fraction
    except Exception:
        pass
    return 0.0, 0.0, 0.0


# ============================================================================
# Configuration loading
# ============================================================================


def load_physics_config(config_path: Path) -> dict:
    """
    Load physics configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_total_clips_from_config(config: dict) -> int:
    """
    Return total number of clips/jobs implied by the dataset config.

    Supports multiple configuration schemas:
    1. Explicit total_clips field
    2. Phase-based schema (phase_a1, phase_b, phase_a2)

    Args:
        config: Physics configuration dictionary

    Returns:
        Total number of clips, or 0 if cannot be determined
    """
    dataset_cfg = config.get("dataset", {})

    # Preferred schema: explicit total
    if isinstance(dataset_cfg.get("total_clips"), int):
        return int(dataset_cfg["total_clips"])

    # Phase schema: sum the phase clip counts if present
    phase_keys = ["phase_a1", "phase_b", "phase_a2"]
    if all(k in dataset_cfg for k in phase_keys):
        total = 0
        for k in phase_keys:
            phase = dataset_cfg.get(k, {})
            if isinstance(phase.get("clips"), int):
                total += int(phase["clips"])
        if total > 0:
            return total

    # Last resort: infer from ranges if available
    if "phase_a1" in dataset_cfg and "range" in dataset_cfg["phase_a1"]:
        # Use the max end across defined phase ranges
        ends = []
        for k in phase_keys:
            phase = dataset_cfg.get(k, {})
            if isinstance(phase.get("range"), (list, tuple)) and len(phase["range"]) == 2:
                ends.append(int(phase["range"][1]))
        if ends:
            return max(ends) + 1

    return 0
