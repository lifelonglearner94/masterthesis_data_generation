#!/usr/bin/env python3
"""
test_extreme_physics.py - Test Script for Extreme Physics Settings

This script generates ONE video clip with EXTREME physics settings for testing:
- Minimum friction (from config: slippery.min = 0.17)
- Maximum force (from config: force.magnitude_max = 4.5)

Uses the same flow as generate_single_clip.py but hardcodes extreme parameters.

Usage:
    python experiments/scripts/test_extreme_physics.py \
        --output_dir experiments/output/extreme_test \
        --seed 42 \
        --physics_config experiments/config/physics_config.yaml

Author: Test script for physics tuning
"""

# ============================================================================
# CRITICAL: Thread limiting MUST happen BEFORE importing NumPy/BLAS libraries
# ============================================================================
import os

def _apply_thread_limits():
    """Apply thread limits from environment variables."""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

_apply_thread_limits()

import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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
import colorsys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Kubric imports
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


def set_deterministic_seed(seed: int) -> None:
    """Set seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set deterministic seed: {seed}")


def sample_extreme_physics_parameters(config: dict) -> dict:
    """
    Sample physics parameters with EXTREME settings for testing:
    - Minimum friction (slippery.min from config)
    - Maximum force (force.magnitude_max from config)
    - Fixed mass (1.0 kg)
    """
    # Use minimum friction from slippery category
    friction_min = config["friction"]["slippery"]["min"]
    friction_coefficient = friction_min

    # Use maximum force
    force_magnitude = config["force"]["magnitude_max"]

    # Use OOD mass for extreme test (testing slippery conditions)
    object_mass = config["object"].get("mass_ood", 1.2)

    # Random object size (keep this random for variety)
    obj_config = config["object"]
    object_size = np.random.uniform(
        obj_config["size_range"][0],
        obj_config["size_range"][1]
    )

    # Random direction in XY plane (keep random as requested)
    angle = np.random.uniform(0, 2 * np.pi)
    force_x = force_magnitude * np.cos(angle)
    force_y = force_magnitude * np.sin(angle)
    force_z = 0.0

    # Random color (high saturation)
    hue = np.random.uniform(0, 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    object_color = (r, g, b, 1.0)

    parameters = {
        "phase": "EXTREME_TEST",
        "friction_coefficient": float(friction_coefficient),
        "friction_category": "slippery",
        "mass_mode": "ood",
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

    # Print extreme settings prominently
    print("\n" + "="*60)
    print("EXTREME PHYSICS TEST PARAMETERS")
    print("="*60)
    print(f"  Friction coefficient: {friction_coefficient:.4f} (MINIMUM from config)")
    print(f"  Force magnitude:      {force_magnitude:.2f} N (MAXIMUM from config)")
    print(f"  Force direction:      ({force_x:.2f}, {force_y:.2f}) N")
    print(f"  Object mass:          {object_mass:.2f} kg (fixed)")
    print(f"  Object size:          {object_size:.3f} m")
    print("="*60 + "\n")

    logger.info(f"EXTREME params: friction={friction_coefficient:.4f}, "
                f"mass={object_mass:.2f}kg, force_mag={force_magnitude:.1f}N")

    return parameters


def create_scene(config: dict, parameters: dict) -> "kb.Scene":
    """Create and configure the Kubric scene."""
    video_config = config["video"]

    scene = kb.Scene(
        resolution=(video_config["resolution"], video_config["resolution"]),
        frame_start=0,
        frame_end=video_config["num_frames"] - 1,
        frame_rate=video_config["fps"],
        step_rate=video_config["simulation_fps"],
    )

    return scene


def setup_camera(scene: "kb.Scene", config: dict) -> dict:
    """Set up the static camera with 45° view."""
    cam_config = config["camera"]

    angle_rad = np.radians(cam_config["angle_degrees"])
    distance = cam_config["distance"]

    cam_z = distance * np.sin(angle_rad)
    cam_horizontal = distance * np.cos(angle_rad)
    cam_x = cam_horizontal * 0.707
    cam_y = cam_horizontal * 0.707

    camera = kb.PerspectiveCamera(
        name="main_camera",
        position=(cam_x, cam_y, cam_z),
        look_at=(0, 0, 0),
        focal_length=cam_config["focal_length"],
        sensor_width=cam_config["sensor_width"],
    )
    scene.add(camera)
    scene.camera = camera

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

    scene.ambient_illumination = kb.Color(
        env_config["ambient_light"],
        env_config["ambient_light"],
        env_config["ambient_light"]
    )

    key_light = kb.DirectionalLight(
        name="key_light",
        position=(3, 3, 5),
        look_at=(0, 0, 0),
        intensity=env_config["key_light_intensity"],
    )
    scene.add(key_light)

    fill_light = kb.DirectionalLight(
        name="fill_light",
        position=(-2, -2, 3),
        look_at=(0, 0, 0),
        intensity=env_config["key_light_intensity"] * 0.5,
    )
    scene.add(fill_light)


def create_floor(scene: "kb.Scene", config: dict, parameters: dict) -> "kb.Cube":
    """Create the physical floor object."""
    obj_config = config.get("object", {})
    arena_size = float(obj_config.get("arena_size", 10.0))

    floor = kb.Cube(
        name="floor",
        scale=(arena_size, arena_size, 0.1),
        position=(0, 0, -0.05),
        static=True,
    )

    floor.friction = parameters["friction_coefficient"]
    scene.add(floor)

    logger.info(f"Created floor with friction={parameters['friction_coefficient']:.4f}")

    return floor


def apply_floor_texture_material(
    renderer: "KubricRenderer",
    floor: "kb.Cube",
    config: dict,
) -> None:
    """Apply an image-based texture to the floor in Blender."""
    if not KUBRIC_AVAILABLE:
        return

    try:
        from kubric.safeimport.bpy import bpy
    except Exception as exc:
        logger.warning(f"Cannot import Blender bpy; floor texture disabled: {exc}")
        return

    env_config = config.get("environment", {})
    texture_path_rel = env_config.get("floor_texture", "")

    if not texture_path_rel:
        logger.warning("No floor_texture configured; using default material")
        return

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

    mat = bpy.data.materials.new("floor_texture")
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    for node in list(nodes):
        nodes.remove(node)

    tiling_config = env_config.get("floor_texture_tiling", {})
    tiling_enabled = tiling_config.get("enabled", False)
    tiling_scale = float(tiling_config.get("scale", 1.0))

    out_node = nodes.new(type="ShaderNodeOutputMaterial")
    out_node.location = (400, 0)

    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf_node.location = (100, 0)

    image_node = nodes.new(type="ShaderNodeTexImage")
    image_node.location = (-200, 0)

    mapping_node = nodes.new(type="ShaderNodeMapping")
    mapping_node.location = (-400, 0)

    texcoord_node = nodes.new(type="ShaderNodeTexCoord")
    texcoord_node.location = (-600, 0)

    image_node.image = bpy.data.images.load(str(texture_path))

    if tiling_enabled:
        image_node.extension = "REPEAT"
        mapping_node.inputs["Scale"].default_value = (tiling_scale, tiling_scale, 1.0)
        logger.info(f"Floor texture tiling enabled: scale={tiling_scale}")
    else:
        image_node.extension = "EXTEND"
        mapping_node.inputs["Scale"].default_value = (1.0, 1.0, 1.0)

    links.new(texcoord_node.outputs["Generated"], mapping_node.inputs["Vector"])
    links.new(mapping_node.outputs["Vector"], image_node.inputs["Vector"])
    links.new(image_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], out_node.inputs["Surface"])

    bsdf_node.inputs["Roughness"].default_value = 0.9
    specular_input = bsdf_node.inputs.get("Specular IOR Level") or bsdf_node.inputs.get("Specular")
    if specular_input:
        specular_input.default_value = 0.2

    if blender_obj.data.materials:
        blender_obj.data.materials[0] = mat
    else:
        blender_obj.data.materials.append(mat)
    logger.info(f"Applied floor texture: {texture_path}")


def sample_initial_cube_position_xy(config: dict, cube_size: float) -> tuple[float, float]:
    """Sample a random initial (x, y) position for the cube."""
    obj_cfg = config.get("object", {}) or {}
    arena_size = float(obj_cfg.get("arena_size", 10.0))

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
            x_low = x_high = y_low = y_high = None
    elif "xy_fraction_of_arena" in init_cfg:
        frac = float(init_cfg["xy_fraction_of_arena"])
        spawn_range = arena_size * frac
        x_low, x_high = -spawn_range, spawn_range
        y_low, y_high = -spawn_range, spawn_range

    if x_low is None:
        spawn_range = arena_size * 0.15
        x_low, x_high = -spawn_range, spawn_range
        y_low, y_high = -spawn_range, spawn_range

    if x_low > x_high:
        x_low, x_high = x_high, x_low
    if y_low > y_high:
        y_low, y_high = y_high, y_low

    x_low = max(x_low, -max_abs_xy)
    x_high = min(x_high, max_abs_xy)
    y_low = max(y_low, -max_abs_xy)
    y_high = min(y_high, max_abs_xy)

    if x_low > x_high or y_low > y_high:
        x_low, x_high = -max_abs_xy, max_abs_xy
        y_low, y_high = -max_abs_xy, max_abs_xy

    start_x = float(np.random.uniform(x_low, x_high))
    start_y = float(np.random.uniform(y_low, y_high))
    return start_x, start_y


def create_dynamic_cube(scene: "kb.Scene", config: dict, parameters: dict) -> "kb.Cube":
    """Create the dynamic cube object."""
    size = parameters["object_size"]
    color = parameters["object_color_rgba"]

    start_x, start_y = sample_initial_cube_position_xy(config, cube_size=size)
    start_z = size / 2

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
) -> np.ndarray:
    """
    Run simulation with force impulse applied between frame 1 and 2.
    Returns cube positions (x, y) for each frame.
    """
    import pybullet as p

    force_vector = parameters["force_vector"]
    mass = parameters["object_mass"]
    apply_frame = config["force"]["apply_at_frame"]

    body_id = cube.linked_objects[simulator]
    physics_client = simulator.physics_client

    frame_start = 0
    frame_end = scene.frame_end
    steps_per_frame = scene.step_rate // scene.frame_rate
    total_steps = (frame_end - frame_start + 1) * steps_per_frame

    logger.info(f"Running simulation frames {frame_start}-{frame_end} "
                f"({total_steps} steps, {steps_per_frame} steps/frame)")

    num_bodies = p.getNumBodies(physicsClientId=physics_client)
    obj_idxs = [
        p.getBodyUniqueId(i, physicsClientId=physics_client)
        for i in range(num_bodies)
    ]

    # Pre-settling phase
    settling_steps = steps_per_frame * 2
    for _ in range(settling_steps):
        p.stepSimulation(physicsClientId=physics_client)

    for obj_idx in obj_idxs:
        mass_info = p.getDynamicsInfo(obj_idx, -1, physicsClientId=physics_client)
        if mass_info[0] > 0:
            p.resetBaseVelocity(
                objectUniqueId=obj_idx,
                linearVelocity=(0, 0, 0),
                angularVelocity=(0, 0, 0),
                physicsClientId=physics_client
            )
    logger.info(f"Pre-settling complete.")

    animation = {
        obj_id: {"position": [], "quaternion": [], "velocity": [], "angular_velocity": []}
        for obj_id in obj_idxs
    }

    impulse_applied = False

    for current_step in range(total_steps):
        if current_step % steps_per_frame == 0:
            current_frame = current_step // steps_per_frame + frame_start

            for obj_idx in obj_idxs:
                position, quaternion = simulator.get_position_and_rotation(obj_idx)
                velocity, angular_velocity = simulator.get_velocities(obj_idx)

                animation[obj_idx]["position"].append(position)
                animation[obj_idx]["quaternion"].append(quaternion)
                animation[obj_idx]["velocity"].append(velocity)
                animation[obj_idx]["angular_velocity"].append(angular_velocity)

            if current_frame == apply_frame and not impulse_applied:
                velocity_change = (
                    force_vector["x"] / mass,
                    force_vector["y"] / mass,
                    force_vector["z"] / mass
                )

                current_vel, current_ang_vel = p.getBaseVelocity(
                    body_id, physicsClientId=physics_client
                )

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
                           f"force=({force_vector['x']:.2f}, {force_vector['y']:.2f}) N, "
                           f"Δv=({velocity_change[0]:.2f}, {velocity_change[1]:.2f}) m/s")

                parameters["velocity_change"] = {
                    "x": float(velocity_change[0]),
                    "y": float(velocity_change[1]),
                    "z": float(velocity_change[2])
                }

                impulse_applied = True

        p.stepSimulation(physicsClientId=physics_client)

    animation = {
        asset: animation[asset.linked_objects[simulator]]
        for asset in scene.assets
        if asset.linked_objects.get(simulator) in obj_idxs
    }

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

    cube_positions = animation[cube]["position"]
    cube_positions_xy = np.array([[pos[0], pos[1]] for pos in cube_positions])

    return cube_positions_xy


def save_states_and_actions(
    actions_states_dir: Path,
    cube_positions_xy: np.ndarray,
    parameters: dict,
    config: dict
) -> None:
    """Save ground_truth_states.npy and actions.npy for this clip."""
    num_frames = config["video"]["num_frames"]
    apply_frame = config["force"]["apply_at_frame"]
    force_vector = parameters["force_vector"]

    states = cube_positions_xy

    actions = np.zeros((num_frames, 2), dtype=np.float32)
    actions[apply_frame, 0] = force_vector["x"]
    actions[apply_frame, 1] = force_vector["y"]

    np.save(actions_states_dir / "ground_truth_states.npy", states.astype(np.float32))
    np.save(actions_states_dir / "actions.npy", actions)

    logger.info(f"Saved states {states.shape} and actions {actions.shape}")


def setup_output_directories(output_dir: Path) -> dict:
    """Create output directory structure."""
    directories = {
        "root": output_dir,
        "rgb": output_dir / "rgb",
        "actions_states": output_dir / "actions_states",
    }

    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return directories


def render_scene(
    scene: "kb.Scene",
    renderer: "KubricRenderer",
    directories: dict,
    config: dict,
) -> dict:
    """Render the scene (RGB only, no GIF)."""
    frames = renderer.render(
        frames=None,
        return_layers=["rgba"]
    )

    rgb_paths = []
    for i, frame in enumerate(frames["rgba"]):
        frame_path = directories["rgb"] / f"frame_{i:05d}.png"
        kb.write_png(frame[..., :3], frame_path)
        rgb_paths.append(str(frame_path))

    logger.info(f"Rendered {len(rgb_paths)} frames")

    return {"rgb_frames": rgb_paths, "gif_preview": None}


def save_ground_truth(
    output_dir: Path,
    seed: int,
    parameters: dict,
    camera_info: dict,
    config: dict
) -> None:
    """Save complete ground truth metadata as JSON."""
    ground_truth = {
        "job_id": "extreme_test",
        "global_seed": seed,
        "job_seed": seed,
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
        "test_mode": "EXTREME_PHYSICS",
        "description": "Min friction + Max force test",
    }

    output_path = output_dir / "ground_truth.json"
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    logger.info(f"Saved ground truth to: {output_path}")


def generate_extreme_clip(
    output_dir: Path,
    seed: int,
    config: dict,
) -> dict:
    """Main function to generate a single extreme physics test clip."""
    set_deterministic_seed(seed)

    # Sample EXTREME physics parameters
    parameters = sample_extreme_physics_parameters(config)

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

    # Setup renderer BEFORE simulator
    renderer = KubricRenderer(
        scene,
        scratch_dir=str(output_dir / ".scratch"),
        adaptive_sampling=False,
        use_denoising=True,
        samples_per_pixel=64,
    )

    # Apply floor texture
    apply_floor_texture_material(renderer, floor, config)

    # Setup simulator
    simulator = KubricSimulator(scene)

    # Run simulation with force impulse
    cube_positions_xy = apply_force_impulse_during_simulation(
        simulator, cube, config, parameters, scene
    )

    # Save states and actions
    save_states_and_actions(
        directories["actions_states"],
        cube_positions_xy,
        parameters,
        config
    )

    # Render
    render_outputs = render_scene(scene, renderer, directories, config)

    # Save ground truth
    save_ground_truth(output_dir, seed, parameters, camera_info, config)

    # Clean up scratch directory
    scratch_dir = output_dir / ".scratch"
    if scratch_dir.exists():
        try:
            shutil.rmtree(scratch_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up scratch: {e}")

    print("\n" + "="*60)
    print("EXTREME PHYSICS TEST COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Check {output_dir}/rgb/ for rendered frames")
    print(f"Check {output_dir}/ground_truth.json for parameters")
    print("="*60 + "\n")

    return {
        "status": "success",
        "mode": "extreme_test",
        "seed": seed,
        "friction": parameters["friction_coefficient"],
        "force_magnitude": parameters["force_magnitude"],
        "output_dir": str(output_dir),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate a single clip with EXTREME physics (min friction, max force).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save output"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility"
    )

    parser.add_argument(
        "--physics_config",
        type=str,
        required=True,
        help="Path to physics_config.yaml"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config_path = Path(args.physics_config)

    if not config_path.exists():
        logger.error(f"Physics config not found: {config_path}")
        sys.exit(1)

    config = load_physics_config(str(config_path))

    gpu_available = check_gpu_available()
    if not gpu_available:
        logger.warning("GPU not detected. Rendering may be slow.")

    if not KUBRIC_AVAILABLE:
        logger.error("Kubric not available. Run this inside the Kubric Docker container.")
        sys.exit(1)

    try:
        result = generate_extreme_clip(output_dir, args.seed, config)
        print(json.dumps(result, indent=2))
        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        logger.exception(f"Failed to generate extreme test clip")
        sys.exit(EXIT_GENERAL_ERROR)


if __name__ == "__main__":
    main()
