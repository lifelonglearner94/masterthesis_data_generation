#!/usr/bin/env python3
"""
generate_single_clip.py - Worker Script for Kubric Data Generation

This script generates exactly ONE video clip with associated metadata.
It is designed to be called by the orchestrator (manager_benchmark.py).

Usage:
    python generate_single_clip.py \
        --output_dir /path/to/output \
        --job_id 0 \
        --seed 42 \
        --physics_config /path/to/physics_config.yaml

Author: Generated for Scientific Data Pipeline
"""

# ============================================================================
# CRITICAL: Thread limiting MUST happen BEFORE importing NumPy/BLAS libraries
# This prevents CPU contention when running multiple workers in parallel.
# ============================================================================
import os

def _apply_thread_limits():
    """
    Apply thread limits from environment variables.

    The manager script injects OMP_NUM_THREADS=1, etc. to prevent
    each worker from spawning multiple threads. This function ensures
    these limits are respected even if set after process start.
    """
    # Check if running in parallel mode (manager sets these)
    if os.environ.get("OMP_NUM_THREADS"):
        # Already set by manager, just log it
        pass
    else:
        # Default to conservative threading if not managed
        # This helps when running standalone too
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

_apply_thread_limits()

import sys
from pathlib import Path

# Add project root to path for imports when running from workspace
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import shared utilities and ensure dependencies before other imports
from experiments.scripts.utils import (
    EXIT_SUCCESS,
    EXIT_GENERAL_ERROR,
    EXIT_VRAM_ERROR,
    ensure_dependencies,
    check_gpu_available,
    load_physics_config,
)

ensure_dependencies()

import argparse
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Kubric imports (available inside the Kubric Docker container)
try:
    import kubric as kb
    from kubric.renderer.blender import Blender as KubricRenderer
    from kubric.simulator.pybullet import PyBullet as KubricSimulator
    KUBRIC_AVAILABLE = True
except ImportError:
    KUBRIC_AVAILABLE = False
    print("WARNING: Kubric not available. Running in dry-run mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Error patterns that indicate VRAM exhaustion
VRAM_ERROR_PATTERNS = [
    "out of memory",
    "CUDA out of memory",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "cudaErrorMemoryAllocation",
    "GPU memory",
    "VRAM",
    "cuMemAlloc",
]


def is_vram_error(error_message: str) -> bool:
    """Check if an error message indicates GPU memory exhaustion."""
    error_lower = error_message.lower()
    return any(pattern.lower() in error_lower for pattern in VRAM_ERROR_PATTERNS)


def get_blender_threads() -> int:
    """
    Get the number of threads Blender should use for CPU operations.

    Respects the BLENDER_CPU_THREADS environment variable set by the manager.
    Falls back to 0 (auto) if not set.

    Returns:
        Number of threads, or 0 for auto-detection
    """
    try:
        return int(os.environ.get("BLENDER_CPU_THREADS", "0"))
    except ValueError:
        return 0


def log_thread_configuration() -> None:
    """Log the current thread configuration for debugging."""
    thread_vars = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLENDER_CPU_THREADS",
    ]
    config_str = ", ".join(f"{v}={os.environ.get(v, 'unset')}" for v in thread_vars)
    logger.debug(f"Thread configuration: {config_str}")


def set_deterministic_seed(seed: int) -> None:
    """
    Set seed for all random number generators to ensure reproducibility.

    Args:
        seed: The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)

    # Set Python hash seed for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Set deterministic seed: {seed}")


def determine_friction_category(job_id: int, config: dict) -> str:
    """
    Determine if this job belongs to 'normal' or 'slippery' category.

    Args:
        job_id: The job identifier
        config: Physics configuration dict

    Returns:
        'normal' or 'slippery'
    """
    # Check phase-based configuration first
    if "phase_a1" in config["dataset"]:
        phase_a1 = config["dataset"]["phase_a1"]
        phase_b = config["dataset"]["phase_b"]
        phase_a2 = config["dataset"]["phase_a2"]

        if phase_a1["range"][0] <= job_id <= phase_a1["range"][1]:
            return phase_a1["friction_mode"]
        elif phase_b["range"][0] <= job_id <= phase_b["range"][1]:
            return phase_b["friction_mode"]
        elif phase_a2["range"][0] <= job_id <= phase_a2["range"][1]:
            return phase_a2["friction_mode"]
        else:
            logger.warning(f"Job ID {job_id} outside defined ranges, defaulting to 'normal'")
            return "normal"

    # No phase configuration found
    raise ValueError(
        f"Missing phase configuration in config. Expected 'phase_a1', 'phase_b', 'phase_a2' in dataset config."
    )


def determine_mass_mode(job_id: int, config: dict) -> str:
    """
    Determine if this job uses 'random' or 'fixed' mass.

    Args:
        job_id: The job identifier
        config: Physics configuration dict

    Returns:
        'random' or 'fixed'
    """
    # Check phase-based configuration
    if "phase_a1" in config["dataset"]:
        phase_a1 = config["dataset"]["phase_a1"]
        phase_b = config["dataset"]["phase_b"]
        phase_a2 = config["dataset"]["phase_a2"]

        if phase_a1["range"][0] <= job_id <= phase_a1["range"][1]:
            return phase_a1["mass_mode"]
        elif phase_b["range"][0] <= job_id <= phase_b["range"][1]:
            return phase_b["mass_mode"]
        elif phase_a2["range"][0] <= job_id <= phase_a2["range"][1]:
            return phase_a2["mass_mode"]
        else:
            logger.warning(f"Job ID {job_id} outside defined ranges, defaulting to 'random'")
            return "random"

    # No phase configuration found
    raise ValueError(
        f"Missing phase configuration in config. Expected 'phase_a1', 'phase_b', 'phase_a2' in dataset config."
    )


def determine_phase(job_id: int, config: dict) -> str:
    """
    Determine which phase this job belongs to.

    Args:
        job_id: The job identifier
        config: Physics configuration dict

    Returns:
        'A_1', 'B', or 'A_2'
    """
    if "phase_a1" in config["dataset"]:
        phase_a1 = config["dataset"]["phase_a1"]
        phase_b = config["dataset"]["phase_b"]
        phase_a2 = config["dataset"]["phase_a2"]

        if phase_a1["range"][0] <= job_id <= phase_a1["range"][1]:
            return "A_1"
        elif phase_b["range"][0] <= job_id <= phase_b["range"][1]:
            return "B"
        elif phase_a2["range"][0] <= job_id <= phase_a2["range"][1]:
            return "A_2"

    return "unknown"


def sample_physics_parameters(config: dict, friction_category: str, mass_mode: str, phase: str) -> dict:
    """
    Sample all physics parameters from the configured ranges.

    Args:
        config: Physics configuration dict
        friction_category: 'normal' or 'slippery'
        mass_mode: 'random' or 'fixed'
        phase: 'A_1', 'B', or 'A_2'

    Returns:
        Dictionary containing all sampled parameters
    """
    # Sample friction based on category
    friction_config = config["friction"][friction_category]
    friction_coefficient = np.random.uniform(
        friction_config["min"],
        friction_config["max"]
    )

    # Sample object properties
    obj_config = config["object"]
    object_size = np.random.uniform(
        obj_config["size_range"][0],
        obj_config["size_range"][1]
    )

    # Sample or use fixed mass based on mass_mode
    if mass_mode == "random":
        object_mass = np.random.uniform(
            obj_config["mass_range"][0],
            obj_config["mass_range"][1]
        )
    else:  # fixed
        object_mass = obj_config.get("mass_fixed", 1.0)

    # Sample force vector
    force_config = config["force"]
    force_magnitude = np.random.uniform(
        force_config["magnitude_min"],
        force_config["magnitude_max"]
    )

    # Random direction in XY plane
    angle = np.random.uniform(0, 2 * np.pi)
    force_x = force_magnitude * np.cos(angle)
    force_y = force_magnitude * np.sin(angle)
    force_z = np.random.uniform(
        force_config["z_component_range"][0],
        force_config["z_component_range"][1]
    )

    # Sample object color (high contrast to floor)
    # Avoid grays, prefer saturated colors
    hue = np.random.uniform(0, 1)
    # Convert HSV to RGB (saturation=0.9, value=0.9 for high contrast)
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    object_color = (r, g, b, 1.0)  # RGBA

    parameters = {
        "phase": phase,
        "friction_coefficient": float(friction_coefficient),
        "friction_category": friction_category,
        "mass_mode": mass_mode,
        "object_size": float(object_size),
        "object_mass": float(object_mass),
        "force_vector": {
            "x": float(force_x),
            "y": float(force_y),
            "z": float(force_z)
        },
        "force_magnitude": float(force_magnitude),
        "object_color_rgba": [float(c) for c in object_color],
    }

    logger.info(f"Sampled parameters: phase={phase}, friction={friction_coefficient:.4f}, "
                f"mass={object_mass:.2f}kg (mode={mass_mode}), force_mag={force_magnitude:.1f}N")

    return parameters


def create_scene(config: dict, parameters: dict) -> "kb.Scene":
    """
    Create and configure the Kubric scene.

    Args:
        config: Physics configuration dict
        parameters: Sampled physics parameters

    Returns:
        Configured Kubric Scene object
    """
    video_config = config["video"]

    # Create scene
    scene = kb.Scene(
        resolution=(video_config["resolution"], video_config["resolution"]),
        frame_start=0,
        frame_end=video_config["num_frames"] - 1,
        frame_rate=video_config["fps"],
        step_rate=video_config["simulation_fps"],
    )

    return scene


def setup_camera(scene: "kb.Scene", config: dict) -> dict:
    """
    Set up the static camera with 45° view.

    Returns:
        Dictionary with camera intrinsics and extrinsics
    """
    cam_config = config["camera"]

    # Calculate camera position for 45° view
    angle_rad = np.radians(cam_config["angle_degrees"])
    distance = cam_config["distance"]

    # Camera looks at origin from 45° elevation
    cam_z = distance * np.sin(angle_rad)
    cam_horizontal = distance * np.cos(angle_rad)

    # Position camera on diagonal for good view
    cam_x = cam_horizontal * 0.707  # cos(45°)
    cam_y = cam_horizontal * 0.707  # sin(45°)

    camera = kb.PerspectiveCamera(
        name="main_camera",
        position=(cam_x, cam_y, cam_z),
        look_at=(0, 0, 0),
        focal_length=cam_config["focal_length"],
        sensor_width=cam_config["sensor_width"],
    )
    scene.add(camera)
    scene.camera = camera

    # Extract camera matrices for metadata
    camera_intrinsics = {
        "focal_length": cam_config["focal_length"],
        "sensor_width": cam_config["sensor_width"],
        "resolution": config["video"]["resolution"],
    }

    camera_extrinsics = {
        "position": [float(cam_x), float(cam_y), float(cam_z)],
        "look_at": [0.0, 0.0, 0.0],
        "angle_degrees": cam_config["angle_degrees"],
    }

    return {
        "intrinsics": camera_intrinsics,
        "extrinsics": camera_extrinsics
    }


def setup_environment(scene: "kb.Scene", config: dict) -> None:
    """Set up lighting and background."""
    env_config = config["environment"]

    # Add ambient light
    scene.ambient_illumination = kb.Color(
        env_config["ambient_light"],
        env_config["ambient_light"],
        env_config["ambient_light"]
    )

    # Add key light
    key_light = kb.DirectionalLight(
        name="key_light",
        position=(3, 3, 5),
        look_at=(0, 0, 0),
        intensity=env_config["key_light_intensity"],
    )
    scene.add(key_light)

    # Add fill light (softer, from opposite side)
    fill_light = kb.DirectionalLight(
        name="fill_light",
        position=(-2, -2, 3),
        look_at=(0, 0, 0),
        intensity=env_config["key_light_intensity"] * 0.5,
    )
    scene.add(fill_light)


def create_floor(scene: "kb.Scene", config: dict, parameters: dict) -> "kb.Cube":
    """Create the physical floor object.

    Note: The *visual* floor texture is applied later (after the Blender renderer
    is initialized), because Kubric's high-level material wrapper is uniform-color by
    default. We intentionally do NOT assign a Kubric material here, as it interferes
    with our custom Blender node-based texture material.
    """
    obj_config = config.get("object", {})
    arena_size = float(obj_config.get("arena_size", 10.0))

    # Create floor plane (make sure it covers the arena).
    floor = kb.Cube(
        name="floor",
        scale=(arena_size, arena_size, 0.1),
        position=(0, 0, -0.05),
        static=True,
    )

    # Set floor friction
    floor.friction = parameters["friction_coefficient"]

    # NOTE: Do NOT assign a Kubric material here!
    # The floor texture will be applied directly via Blender nodes in
    # apply_floor_texture_material() after the renderer is initialized.
    # Assigning a Kubric PrincipledBSDFMaterial here would interfere with
    # our custom texture material.

    scene.add(floor)

    logger.info(f"Created floor with friction={parameters['friction_coefficient']:.4f}")

    return floor


def apply_floor_texture_material(
    renderer: "KubricRenderer",
    floor: "kb.Cube",
    config: dict,
) -> None:
    """Apply an image-based texture to the floor in Blender.

    Kubric's `PrincipledBSDFMaterial` is a uniform material by default. To get a
    visible texture, we build a Blender node graph using `ShaderNodeTexImage`.

    This must run after `KubricRenderer(scene, ...)` is created, because that's when
    the Blender objects exist and `floor.linked_objects[renderer]` is populated.

    Uses 'Generated' coordinates which are automatically normalized to [0,1]
    across the object's bounding box, making them ideal for image textures.
    """
    if not KUBRIC_AVAILABLE:
        return

    try:
        from kubric.safeimport.bpy import bpy  # type: ignore
    except Exception as exc:
        logger.warning(f"Cannot import Blender bpy; floor texture disabled: {exc}")
        return

    env_config = config.get("environment", {})
    texture_path_rel = env_config.get("floor_texture", "")

    if not texture_path_rel:
        logger.warning("No floor_texture configured; using default material")
        return

    # Resolve texture path relative to repo root (script is in experiments/scripts/)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    texture_path = repo_root / texture_path_rel

    if not texture_path.exists():
        logger.error(f"Floor texture not found: {texture_path}")
        return

    try:
        blender_obj = floor.linked_objects[renderer]
    except Exception as exc:
        logger.warning(f"Floor not linked in Blender yet; floor texture disabled: {exc}")
        return

    # Build node-based image texture material
    mat = bpy.data.materials.new("floor_texture")
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    # Clear default nodes
    for node in list(nodes):
        nodes.remove(node)

    # Get tiling configuration
    tiling_config = env_config.get("floor_texture_tiling", {})
    tiling_enabled = tiling_config.get("enabled", False)
    tiling_scale = float(tiling_config.get("scale", 1.0))

    # Create nodes
    out_node = nodes.new(type="ShaderNodeOutputMaterial")
    out_node.location = (400, 0)

    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf_node.location = (100, 0)

    image_node = nodes.new(type="ShaderNodeTexImage")
    image_node.location = (-200, 0)

    # Mapping node for UV scaling/tiling
    mapping_node = nodes.new(type="ShaderNodeMapping")
    mapping_node.location = (-400, 0)

    texcoord_node = nodes.new(type="ShaderNodeTexCoord")
    texcoord_node.location = (-600, 0)

    # Load the image texture
    image_node.image = bpy.data.images.load(str(texture_path))

    # Configure tiling based on config
    if tiling_enabled:
        # REPEAT allows texture to tile seamlessly
        image_node.extension = "REPEAT"
        # Scale determines how many times texture repeats across floor
        mapping_node.inputs["Scale"].default_value = (tiling_scale, tiling_scale, 1.0)
        logger.info(f"Floor texture tiling enabled: scale={tiling_scale}")
    else:
        # EXTEND stretches texture across floor without tiling
        image_node.extension = "EXTEND"
        mapping_node.inputs["Scale"].default_value = (1.0, 1.0, 1.0)

    # Connect nodes: TexCoord -> Mapping -> ImageTexture -> BSDF
    # Using "Generated" coordinates which are normalized to [0,1] over bounding box
    links.new(texcoord_node.outputs["Generated"], mapping_node.inputs["Vector"])
    links.new(mapping_node.outputs["Vector"], image_node.inputs["Vector"])
    links.new(image_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], out_node.inputs["Surface"])

    # Material properties - keep the floor fairly matte
    bsdf_node.inputs["Roughness"].default_value = 0.9
    # Blender 4.0+ uses "Specular IOR Level", older versions use "Specular"
    specular_input = bsdf_node.inputs.get("Specular IOR Level") or bsdf_node.inputs.get("Specular")
    if specular_input:
        specular_input.default_value = 0.2

    # Assign material to the Blender object's mesh data
    # NOTE: We must replace the material in data.materials, not just set active_material
    # Kubric already assigned a PrincipledBSDFMaterial, so we need to overwrite it
    if blender_obj.data.materials:
        blender_obj.data.materials[0] = mat
    else:
        blender_obj.data.materials.append(mat)
    logger.info(f"Applied floor texture: {texture_path}")


def sample_initial_cube_position_xy(config: dict, cube_size: float) -> tuple[float, float]:
    """Sample a random initial (x, y) position for the cube near the arena center.

    Supports either explicit ranges:
      object.initial_position.x_range, object.initial_position.y_range
    or a fraction of the arena size:
      object.initial_position.xy_fraction_of_arena

    The sampled position is clamped to stay inside the arena bounds.
    """
    obj_cfg = config.get("object", {}) or {}
    arena_size = float(obj_cfg.get("arena_size", 10.0))

    # Keep cube fully inside arena
    max_abs_xy = arena_size / 2.0 - cube_size / 2.0
    if max_abs_xy <= 0:
        return 0.0, 0.0

    init_cfg = obj_cfg.get("initial_position", {}) or {}

    x_low = x_high = y_low = y_high = None

    if "x_range" in init_cfg and "y_range" in init_cfg:
        try:
            x_low, x_high = map(float, init_cfg["x_range"])
            y_low, y_high = map(float, init_cfg["y_range"])
        except Exception:
            logger.warning("Invalid object.initial_position x_range/y_range; falling back to defaults")
            x_low = x_high = y_low = y_high = None
    elif "xy_fraction_of_arena" in init_cfg:
        frac = float(init_cfg["xy_fraction_of_arena"])
        spawn_range = arena_size * frac
        x_low, x_high = -spawn_range, spawn_range
        y_low, y_high = -spawn_range, spawn_range

    if x_low is None:
        # Backwards-compatible default: central 30% of arena (range = 0.15 * arena_size)
        spawn_range = arena_size * 0.15
        x_low, x_high = -spawn_range, spawn_range
        y_low, y_high = -spawn_range, spawn_range

    if x_low > x_high:
        x_low, x_high = x_high, x_low
    if y_low > y_high:
        y_low, y_high = y_high, y_low

    # Clamp configured ranges to physical arena bounds
    x_low = max(x_low, -max_abs_xy)
    x_high = min(x_high, max_abs_xy)
    y_low = max(y_low, -max_abs_xy)
    y_high = min(y_high, max_abs_xy)

    if x_low > x_high or y_low > y_high:
        logger.warning(
            "Configured initial position range is outside arena bounds; using full allowed range"
        )
        x_low, x_high = -max_abs_xy, max_abs_xy
        y_low, y_high = -max_abs_xy, max_abs_xy

    start_x = float(np.random.uniform(x_low, x_high))
    start_y = float(np.random.uniform(y_low, y_high))
    return start_x, start_y


def create_dynamic_cube(scene: "kb.Scene", config: dict, parameters: dict) -> "kb.Cube":
    """Create the dynamic cube object."""
    size = parameters["object_size"]
    color = parameters["object_color_rgba"]

    # Random starting position (near center, exactly resting on floor)
    start_x, start_y = sample_initial_cube_position_xy(config, cube_size=size)
    start_z = size / 2  # Exactly resting on floor (no gap = no settling)

    cube = kb.Cube(
        name="dynamic_cube",
        scale=(size, size, size),
        position=(start_x, start_y, start_z),
        velocity=(0, 0, 0),
        static=False,
        mass=parameters["object_mass"],
    )

    cube.friction = parameters["friction_coefficient"]

    cube.material = kb.PrincipledBSDFMaterial(
        name="cube_material",
        color=kb.Color(*color[:3]),
        metallic=0.1,
        roughness=0.7,
    )

    scene.add(cube)

    # Store initial position in parameters
    parameters["initial_position"] = {
        "x": float(start_x),
        "y": float(start_y),
        "z": float(start_z)
    }

    logger.info(f"Created cube: size={size:.3f}, mass={parameters['object_mass']:.2f}kg, "
                f"pos=({start_x:.2f}, {start_y:.2f}, {start_z:.2f})")

    return cube


def apply_force_impulse_during_simulation(
    simulator: "KubricSimulator",
    cube: "kb.Cube",
    config: dict,
    parameters: dict,
    scene: "kb.Scene"
) -> None:
    """
    Run simulation with force impulse applied between frame 1 and 2.

    The key insight is that we need to run the ENTIRE simulation in one go,
    but inject the impulse at the right moment during the simulation loop.

    Kubric's simulator.run() records keyframes internally and transfers them
    to Blender at the end. If we call run() twice, the keyframes from the
    second call overwrite those from the first for overlapping frames, AND
    any PyBullet state changes made between run() calls are not properly
    tracked in the animation dictionary.

    Solution: We manually implement the simulation loop with impulse injection,
    similar to how Kubric's run() method works, but with impulse applied at
    the correct timestep.
    """
    import pybullet as p

    force_vector = parameters["force_vector"]
    mass = parameters["object_mass"]
    apply_frame = config["force"]["apply_at_frame"]

    # Get PyBullet body ID for the cube
    body_id = cube.linked_objects[simulator]
    # physics_client property returns the raw pybullet client ID
    physics_client = simulator.physics_client

    # Simulation parameters
    frame_start = 0
    frame_end = scene.frame_end
    steps_per_frame = scene.step_rate // scene.frame_rate
    total_steps = (frame_end - frame_start + 1) * steps_per_frame

    logger.info(f"Running simulation frames {frame_start}-{frame_end} "
                f"({total_steps} steps, {steps_per_frame} steps/frame)")
    logger.info(f"Impulse will be applied after frame {apply_frame} is recorded")

    # Get all body IDs in the simulation using pybullet directly
    num_bodies = p.getNumBodies(physicsClientId=physics_client)
    obj_idxs = [
        p.getBodyUniqueId(i, physicsClientId=physics_client)
        for i in range(num_bodies)
    ]

    # PRE-SETTLING PHASE: Run physics briefly to let objects settle,
    # then reset velocities to zero. This ensures frame 0 is truly at rest.
    settling_steps = steps_per_frame * 2  # 2 frames worth of settling
    for _ in range(settling_steps):
        p.stepSimulation(physicsClientId=physics_client)

    # Reset all dynamic objects to zero velocity after settling
    for obj_idx in obj_idxs:
        # Check if this is a dynamic body (not static)
        body_info = p.getBodyInfo(obj_idx, physicsClientId=physics_client)
        mass_info = p.getDynamicsInfo(obj_idx, -1, physicsClientId=physics_client)
        if mass_info[0] > 0:  # mass > 0 means dynamic
            p.resetBaseVelocity(
                objectUniqueId=obj_idx,
                linearVelocity=(0, 0, 0),
                angularVelocity=(0, 0, 0),
                physicsClientId=physics_client
            )
    logger.info(f"Pre-settling complete. All dynamic objects reset to zero velocity.")

    # Animation storage (matches Kubric's internal format)
    animation = {
        obj_id: {"position": [], "quaternion": [], "velocity": [], "angular_velocity": []}
        for obj_id in obj_idxs
    }

    impulse_applied = False

    for current_step in range(total_steps):
        # Record keyframe at the start of each frame
        if current_step % steps_per_frame == 0:
            current_frame = current_step // steps_per_frame + frame_start

            for obj_idx in obj_idxs:
                position, quaternion = simulator.get_position_and_rotation(obj_idx)
                velocity, angular_velocity = simulator.get_velocities(obj_idx)

                animation[obj_idx]["position"].append(position)
                animation[obj_idx]["quaternion"].append(quaternion)
                animation[obj_idx]["velocity"].append(velocity)
                animation[obj_idx]["angular_velocity"].append(angular_velocity)

            # Apply impulse right after recording the apply_frame keyframe
            # This means the impulse effect will be visible starting from frame apply_frame+1
            if current_frame == apply_frame and not impulse_applied:
                # Calculate velocity change from impulse: Δv = F/m
                velocity_change = (
                    force_vector["x"] / mass,
                    force_vector["y"] / mass,
                    force_vector["z"] / mass
                )

                # Get current velocity of the cube
                current_vel, current_ang_vel = p.getBaseVelocity(
                    body_id, physicsClientId=physics_client
                )

                # Apply impulse by setting new velocity
                new_velocity = (
                    current_vel[0] + velocity_change[0],
                    current_vel[1] + velocity_change[1],
                    current_vel[2] + velocity_change[2]
                )

                p.resetBaseVelocity(
                    objectUniqueId=body_id,
                    linearVelocity=new_velocity,
                    angularVelocity=current_ang_vel,
                    physicsClientId=physics_client
                )

                logger.info(f"Applied impulse at frame {apply_frame}: "
                           f"force=({force_vector['x']:.2f}, {force_vector['y']:.2f}, {force_vector['z']:.2f}) N, "
                           f"Δv=({velocity_change[0]:.2f}, {velocity_change[1]:.2f}, {velocity_change[2]:.2f}) m/s")

                # Store the actual velocity change for ground truth
                parameters["velocity_change"] = {
                    "x": float(velocity_change[0]),
                    "y": float(velocity_change[1]),
                    "z": float(velocity_change[2])
                }

                impulse_applied = True

        # Step the physics simulation using pybullet directly
        p.stepSimulation(physicsClientId=physics_client)

    # Map body IDs back to Kubric assets
    animation = {
        asset: animation[asset.linked_objects[simulator]]
        for asset in scene.assets
        if asset.linked_objects.get(simulator) in obj_idxs
    }

    # Transfer simulation results to renderer keyframes (same as Kubric's run() does)
    logger.info("Transferring simulation keyframes to renderer...")
    for obj in animation.keys():
        for frame_id in range(frame_end - frame_start + 1):
            obj.position = animation[obj]["position"][frame_id]
            obj.quaternion = animation[obj]["quaternion"][frame_id]
            obj.velocity = animation[obj]["velocity"][frame_id]
            obj.angular_velocity = animation[obj]["angular_velocity"][frame_id]
            obj.keyframe_insert("position", frame_id + frame_start)
            obj.keyframe_insert("quaternion", frame_id + frame_start)
            obj.keyframe_insert("velocity", frame_id + frame_start)
            obj.keyframe_insert("angular_velocity", frame_id + frame_start)

    logger.info(f"Simulation complete. Recorded {frame_end - frame_start + 1} keyframes.")


def setup_output_directories(output_dir: Path) -> dict:
    """Create output directory structure for this clip."""
    directories = {
        "root": output_dir,
        "rgb": output_dir / "rgb",
        # NOTE: flow and segmentation disabled - only metadata needed
        # "flow": output_dir / "flow",
        # "segmentation": output_dir / "segmentation",
    }

    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return directories


def render_scene(
    scene: "kb.Scene",
    renderer: "KubricRenderer",
    directories: dict,
    config: dict,
    skip_gif: bool = False,
) -> dict:
    """
    Render the scene with RGB only (flow and segmentation disabled).

    Args:
        scene: Kubric scene to render
        renderer: Kubric Blender renderer
        directories: Output directory structure
        config: Physics configuration
        skip_gif: If True, skip GIF preview generation (faster for production)

    Returns:
        Dictionary with paths to rendered outputs
    """
    # Render all frames (None means render all frames defined in scene)
    # NOTE: flow and segmentation disabled - only metadata needed
    frames = renderer.render(
        frames=None,  # Render all frames in the scene
        return_layers=["rgba"]  # Removed "forward_flow", "segmentation"
    )

    # Save RGB frames
    rgb_paths = []
    for i, frame in enumerate(frames["rgba"]):
        frame_path = directories["rgb"] / f"frame_{i:05d}.png"
        kb.write_png(frame[..., :3], frame_path)  # RGB only, no alpha
        rgb_paths.append(str(frame_path))

    # Create a quick GIF preview for visual appeal (not for analysis)
    # Skip in production mode to save time
    gif_path = None
    if not skip_gif:
        video_config = config["video"]
        gif_path = directories["root"] / "preview.gif"

        try:
            from PIL import Image as PILImage

            # Load RGB frames as PIL Images
            pil_frames = []
            for rgb_path in rgb_paths:
                pil_frames.append(PILImage.open(rgb_path).convert("RGB"))

            # Calculate frame duration in milliseconds
            frame_duration_ms = int(1000 / video_config['fps'])

            # Save as GIF (first frame with append of rest)
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=frame_duration_ms,
                loop=0  # Loop forever
            )
            logger.info(f"Created GIF preview: {gif_path}")
        except Exception as e:
            logger.warning(f"Could not create GIF preview: {e}")
            gif_path = None
    else:
        logger.debug("Skipping GIF preview generation (--no_gif)")

    logger.info(f"Rendered {len(rgb_paths)} frames")

    return {
        "rgb_frames": rgb_paths,
        "gif_preview": str(gif_path) if gif_path else None,
    }


def save_ground_truth(
    output_dir: Path,
    job_id: int,
    seed: int,
    job_seed: int,
    parameters: dict,
    camera_info: dict,
    config: dict
) -> None:
    """Save complete ground truth metadata as JSON."""
    ground_truth = {
        "job_id": job_id,
        "global_seed": seed,
        "job_seed": job_seed,
        "friction_category": parameters["friction_category"],
        "physics": {
            "friction_coefficient": parameters["friction_coefficient"],
            "mass": parameters["object_mass"],
            "object_size": parameters["object_size"],
        },
        "force": {
            "vector": parameters["force_vector"],
            "magnitude": parameters["force_magnitude"],
            "applied_at_frame": config["force"]["apply_at_frame"],
            "velocity_change": parameters.get("velocity_change", None),
        },
        "initial_conditions": {
            "position": parameters["initial_position"],
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
        "object": {
            "type": "cube",
            "color_rgba": parameters["object_color_rgba"],
        },
        "camera": camera_info,
        "video": {
            "resolution": config["video"]["resolution"],
            "num_frames": config["video"]["num_frames"],
            "fps": config["video"]["fps"],
            "simulation_fps": config["video"]["simulation_fps"],
        },
    }

    output_path = output_dir / "ground_truth.json"
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    logger.info(f"Saved ground truth to: {output_path}")


def generate_clip_dry_run(
    output_dir: Path,
    job_id: int,
    seed: int,
    config: dict
) -> dict:
    """
    Dry-run mode when Kubric is not available.
    Generates parameter files without actual rendering.
    """
    job_seed = seed + job_id
    set_deterministic_seed(job_seed)

    friction_category = determine_friction_category(job_id, config)
    mass_mode = determine_mass_mode(job_id, config)
    phase = determine_phase(job_id, config)
    parameters = sample_physics_parameters(config, friction_category, mass_mode, phase)

    # Add placeholder initial position
    start_x, start_y = sample_initial_cube_position_xy(config, cube_size=parameters["object_size"])
    parameters["initial_position"] = {
        "x": float(start_x),
        "y": float(start_y),
        "z": float(parameters["object_size"] / 2 + 0.01)
    }

    # Create output structure
    directories = setup_output_directories(output_dir)

    # Fake camera info
    camera_info = {
        "intrinsics": {
            "focal_length": config["camera"]["focal_length"],
            "sensor_width": config["camera"]["sensor_width"],
            "resolution": config["video"]["resolution"],
        },
        "extrinsics": {
            "position": [1.75, 1.75, 2.47],
            "look_at": [0.0, 0.0, 0.0],
            "angle_degrees": config["camera"]["angle_degrees"],
        }
    }

    # Save ground truth
    save_ground_truth(
        output_dir, job_id, seed, job_seed,
        parameters, camera_info, config
    )

    logger.info(f"[DRY-RUN] Generated metadata for job {job_id}")

    return {
        "status": "success",
        "mode": "dry_run",
        "job_id": job_id,
        "seed": job_seed,
        "friction_category": friction_category,
        "output_dir": str(output_dir),
    }


def generate_clip(
    output_dir: Path,
    job_id: int,
    seed: int,
    config: dict,
    skip_gif: bool = False,
) -> dict:
    """
    Main function to generate a single clip.

    Args:
        output_dir: Path to save this clip's data
        job_id: Unique identifier for this clip
        seed: Global seed value
        config: Physics configuration dict
        skip_gif: If True, skip GIF preview generation

    Returns:
        Dictionary with generation results
    """
    # Calculate job-specific seed
    job_seed = seed + job_id
    set_deterministic_seed(job_seed)

    # Determine phase, friction category, and mass mode
    friction_category = determine_friction_category(job_id, config)
    mass_mode = determine_mass_mode(job_id, config)
    phase = determine_phase(job_id, config)
    parameters = sample_physics_parameters(config, friction_category, mass_mode, phase)

    # Create output directories
    directories = setup_output_directories(output_dir)

    # Create scene
    scene = create_scene(config, parameters)

    # Setup camera
    camera_info = setup_camera(scene, config)

    # Setup environment (lighting)
    setup_environment(scene, config)

    # Create floor
    floor = create_floor(scene, config, parameters)

    # Create dynamic object
    cube = create_dynamic_cube(scene, config, parameters)

    # IMPORTANT: Setup renderer BEFORE simulator
    # The renderer must be created before simulation so it can observe keyframe changes
    # when keyframe_insert() is called during/after simulation
    renderer = KubricRenderer(
        scene,
        scratch_dir=str(output_dir / ".scratch"),
        adaptive_sampling=False,
        use_denoising=True,
        samples_per_pixel=64,
    )

    # Apply the floor texture (rendering only).
    apply_floor_texture_material(renderer, floor, config)

    # Setup simulator
    simulator = KubricSimulator(scene)

    # Run simulation with force impulse applied between frame 1 and 2
    # This function handles the phased simulation internally with proper keyframe recording
    apply_force_impulse_during_simulation(simulator, cube, config, parameters, scene)

    # Note: simulation is now complete with all keyframes properly transferred to Blender

    # Render all outputs
    render_outputs = render_scene(scene, renderer, directories, config, skip_gif=skip_gif)

    # Save ground truth
    save_ground_truth(
        output_dir, job_id, seed, job_seed,
        parameters, camera_info, config
    )

    # Clean up scratch directory to save space
    scratch_dir = output_dir / ".scratch"
    if scratch_dir.exists():
        try:
            shutil.rmtree(scratch_dir)
            logger.info(f"Cleaned up scratch directory: {scratch_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up scratch directory: {e}")

    logger.info(f"Successfully generated clip {job_id}")

    return {
        "status": "success",
        "mode": "full",
        "job_id": job_id,
        "seed": job_seed,
        "friction_category": friction_category,
        "output_dir": str(output_dir),
        "render_outputs": render_outputs,
    }


def main():
    """Main entry point for the worker script."""
    parser = argparse.ArgumentParser(
        description="Generate a single Kubric video clip with ground truth metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save this specific job's output"
    )

    parser.add_argument(
        "--job_id",
        type=int,
        required=True,
        help="Integer ID for this specific clip (0 to N-1)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Global/master seed for reproducibility"
    )

    parser.add_argument(
        "--physics_config",
        type=str,
        required=True,
        help="Path to YAML file defining physics parameter ranges"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only generate metadata without rendering"
    )

    parser.add_argument(
        "--no_gif",
        action="store_true",
        help="Skip GIF preview generation (faster for production runs)"
    )

    args = parser.parse_args()

    # Validate inputs
    output_dir = Path(args.output_dir)
    config_path = Path(args.physics_config)

    if not config_path.exists():
        logger.error(f"Physics config not found: {config_path}")
        sys.exit(1)

    # Load configuration
    config = load_physics_config(str(config_path))

    # Check GPU availability
    gpu_available = check_gpu_available()
    if not gpu_available:
        logger.warning("GPU not detected. Rendering may be slow.")

    # Generate the clip
    try:
        if args.dry_run or not KUBRIC_AVAILABLE:
            result = generate_clip_dry_run(
                output_dir,
                args.job_id,
                args.seed,
                config
            )
        else:
            result = generate_clip(
                output_dir,
                args.job_id,
                args.seed,
                config,
                skip_gif=args.no_gif,
            )

        # Print result as JSON for the orchestrator to parse
        print(json.dumps(result, indent=2))
        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        error_str = str(e)
        logger.exception(f"Failed to generate clip {args.job_id}")

        # Determine exit code based on error type
        if is_vram_error(error_str):
            exit_code = EXIT_VRAM_ERROR
            logger.error("VRAM error detected - job may be retried")
        else:
            exit_code = EXIT_GENERAL_ERROR

        error_result = {
            "status": "error",
            "job_id": args.job_id,
            "error": error_str,
            "output_dir": str(output_dir),
            "exit_code": exit_code,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
