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

import subprocess
import sys
import shutil

# Auto-install missing dependencies
def _ensure_dependencies():
    """Install required packages if missing."""
    required = ["pyyaml", "numpy"]
    for pkg in required:
        try:
            __import__("yaml" if pkg == "pyyaml" else pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                stdout=subprocess.DEVNULL
            )

def _ensure_ffmpeg():
    """Install ffmpeg if not available (required for MP4 video creation)."""
    if shutil.which("ffmpeg") is None:
        print("Installing ffmpeg for video creation...")
        try:
            subprocess.check_call(
                ["apt-get", "update"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            subprocess.check_call(
                ["apt-get", "install", "-y", "ffmpeg"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("ffmpeg installed successfully.")
        except Exception as e:
            print(f"WARNING: Could not install ffmpeg: {e}")
            print("Video creation will be skipped.")

_ensure_dependencies()
_ensure_ffmpeg()

import argparse
import json
import logging
import os
import random
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


def check_gpu_available() -> bool:
    """Check if NVIDIA GPU is available via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip()
            logger.info(f"GPU detected: {gpu_name}")
            return True
        else:
            logger.warning("nvidia-smi returned non-zero exit code")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"GPU check failed: {e}")
        return False


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


def load_physics_config(config_path: str) -> dict:
    """Load physics configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded physics config from: {config_path}")
    return config


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

    # Fallback to legacy range-based logic
    normal_range = config["dataset"]["normal_range"]
    slippery_range = config["dataset"]["slippery_range"]

    if normal_range[0] <= job_id <= normal_range[1]:
        return "normal"
    elif slippery_range[0] <= job_id <= slippery_range[1]:
        return "slippery"
    else:
        # Default to normal for out-of-range job IDs
        logger.warning(f"Job ID {job_id} outside defined ranges, defaulting to 'normal'")
        return "normal"


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

    # Legacy: always random mass
    return "random"


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

    # Sample restitution
    rest_config = config["restitution"]
    restitution = np.random.uniform(rest_config["min"], rest_config["max"])

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

    # Sample object color (high contrast to checkerboard)
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
        "restitution": float(restitution),
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


def create_checkerboard_floor(scene: "kb.Scene", config: dict, parameters: dict) -> None:
    """Create a checkerboard textured floor."""
    # Create floor plane
    floor = kb.Cube(
        name="floor",
        scale=(5, 5, 0.1),
        position=(0, 0, -0.05),
        static=True,
    )

    # Set floor friction
    floor.friction = parameters["friction_coefficient"]
    floor.restitution = parameters["restitution"]

    # Apply checkerboard material (Kubric handles this via Blender materials)
    floor.material = kb.PrincipledBSDFMaterial(
        name="floor_material",
        color=kb.Color(0.8, 0.8, 0.8),  # Base color, checkerboard applied in Blender
    )

    scene.add(floor)

    logger.info(f"Created floor with friction={parameters['friction_coefficient']:.4f}")


def create_walls(scene: "kb.Scene", config: dict) -> None:
    """Create arena walls to contain the cube."""
    wall_config = config["walls"]

    if not wall_config["enabled"]:
        return

    arena_size = wall_config["arena_size"]
    thickness = wall_config["thickness"]
    height = wall_config["height"]

    wall_positions = [
        ("wall_north", (0, arena_size/2, height/2), (arena_size, thickness, height)),
        ("wall_south", (0, -arena_size/2, height/2), (arena_size, thickness, height)),
        ("wall_east", (arena_size/2, 0, height/2), (thickness, arena_size, height)),
        ("wall_west", (-arena_size/2, 0, height/2), (thickness, arena_size, height)),
    ]

    for name, position, scale in wall_positions:
        wall = kb.Cube(
            name=name,
            position=position,
            scale=scale,
            static=True,
        )
        wall.friction = wall_config["friction"]
        wall.restitution = wall_config["restitution"]
        wall.material = kb.PrincipledBSDFMaterial(
            name=f"{name}_material",
            color=kb.Color(0.3, 0.3, 0.35),
        )
        scene.add(wall)

    logger.info("Created arena walls")


def create_dynamic_cube(scene: "kb.Scene", config: dict, parameters: dict) -> "kb.Cube":
    """Create the dynamic cube object."""
    size = parameters["object_size"]
    color = parameters["object_color_rgba"]

    # Random starting position (centered, exactly resting on floor)
    # Use a smaller spawn area relative to the arena size
    arena_size = config["walls"].get("arena_size", 2.5)
    spawn_range = arena_size * 0.15  # Spawn in central 30% of arena
    start_x = np.random.uniform(-spawn_range, spawn_range)
    start_y = np.random.uniform(-spawn_range, spawn_range)
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
    cube.restitution = parameters["restitution"]

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
        "flow": output_dir / "flow",
        "segmentation": output_dir / "segmentation",
    }

    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return directories


def render_scene(
    scene: "kb.Scene",
    renderer: "KubricRenderer",
    directories: dict,
    config: dict
) -> dict:
    """
    Render the scene with RGB, optical flow, and segmentation.

    Returns:
        Dictionary with paths to rendered outputs
    """
    # Render all frames (None means render all frames defined in scene)
    frames = renderer.render(
        frames=None,  # Render all frames in the scene
        return_layers=["rgba", "forward_flow", "segmentation"]
    )

    # Save RGB frames
    rgb_paths = []
    for i, frame in enumerate(frames["rgba"]):
        frame_path = directories["rgb"] / f"frame_{i:05d}.png"
        kb.write_png(frame[..., :3], frame_path)  # RGB only, no alpha
        rgb_paths.append(str(frame_path))

    # Save optical flow as numpy arrays (preserves float precision and 2-channel data)
    # Flow has shape (H, W, 2) for x and y components
    flow_paths = []
    for i, flow in enumerate(frames["forward_flow"]):
        flow_path = directories["flow"] / f"flow_{i:05d}.npy"
        np.save(flow_path, flow)
        flow_paths.append(str(flow_path))

    # Save segmentation masks
    seg_paths = []
    for i, seg in enumerate(frames["segmentation"]):
        seg_path = directories["segmentation"] / f"seg_{i:05d}.png"
        kb.write_png(seg, seg_path)
        seg_paths.append(str(seg_path))

    # Create video using ffmpeg (kb.write_video doesn't exist)
    video_config = config["video"]
    video_path = directories["root"] / "video.mp4"

    try:
        import subprocess
        import tempfile

        # Create a temporary file list for ffmpeg
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for rgb_path in rgb_paths:
                f.write(f"file '{rgb_path}'\n")
                f.write(f"duration {1.0 / video_config['fps']}\n")
            file_list = f.name

        # Use ffmpeg to create video with All-Intra encoding (no inter-frame compression)
        # This provides maximum quality and frame-accurate seeking
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', file_list,
            '-c:v', 'libx264',
            '-preset', 'slow',           # Better compression efficiency
            '-crf', '0',                  # Lossless quality
            '-g', '1',                    # GOP size = 1 (All-Intra: every frame is a keyframe)
            '-pix_fmt', 'yuv444p',        # Full chroma resolution (no subsampling)
            '-r', str(video_config['fps']),
            str(video_path)
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        os.unlink(file_list)
        logger.info(f"Created video: {video_path}")
    except Exception as e:
        logger.warning(f"Could not create video: {e}")
        video_path = None

    logger.info(f"Rendered {len(rgb_paths)} frames")

    return {
        "rgb_frames": rgb_paths,
        "flow_frames": flow_paths,
        "segmentation_frames": seg_paths,
        "video": str(video_path) if video_path else None,
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
            "restitution": parameters["restitution"],
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
    parameters["initial_position"] = {
        "x": float(np.random.uniform(-0.5, 0.5)),
        "y": float(np.random.uniform(-0.5, 0.5)),
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
    config: dict
) -> dict:
    """
    Main function to generate a single clip.

    Args:
        output_dir: Path to save this clip's data
        job_id: Unique identifier for this clip
        seed: Global seed value
        config: Physics configuration dict

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

    # Create floor and walls
    create_checkerboard_floor(scene, config, parameters)
    create_walls(scene, config)

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

    # Setup simulator
    simulator = KubricSimulator(scene)

    # Run simulation with force impulse applied between frame 1 and 2
    # This function handles the phased simulation internally with proper keyframe recording
    apply_force_impulse_during_simulation(simulator, cube, config, parameters, scene)

    # Note: simulation is now complete with all keyframes properly transferred to Blender

    # Render all outputs
    render_outputs = render_scene(scene, renderer, directories, config)

    # Save ground truth
    save_ground_truth(
        output_dir, job_id, seed, job_seed,
        parameters, camera_info, config
    )

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
                config
            )

        # Print result as JSON for the orchestrator to parse
        print(json.dumps(result, indent=2))
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Failed to generate clip {args.job_id}")
        error_result = {
            "status": "error",
            "job_id": args.job_id,
            "error": str(e),
            "output_dir": str(output_dir),
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
