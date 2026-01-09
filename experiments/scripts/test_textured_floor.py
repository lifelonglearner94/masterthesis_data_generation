#!/usr/bin/env python3
"""
Textured Grid Floor Demo for Kubric.

This script demonstrates how to apply a custom image texture to a floor
using direct Blender node manipulation within Kubric.

Usage (inside Kubric Docker container):
    python experiments/scripts/test_textured_floor.py
"""

import os
from pathlib import Path

import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.simulator.pybullet import PyBullet as KubricSimulator
from kubric.safeimport.bpy import bpy  # type: ignore


def apply_texture_to_floor(
    kubric_obj: kb.Cube,
    renderer: KubricRenderer,
    texture_path: str,
    tiling_scale: float = 20.0,
) -> None:
    """
    Apply an image texture to a Kubric floor object using Blender nodes.

    This function bridges Kubric and Blender by accessing the underlying
    Blender object and constructing a custom shader node graph for tiled
    texture mapping.

    Args:
        kubric_obj: The Kubric Cube object representing the floor.
        renderer: The KubricRenderer instance (needed to access linked Blender objects).
        texture_path: Absolute path to the texture image file.
        tiling_scale: How many times the texture tiles across each axis.
                      E.g., 20.0 means 20 tiles across a 20-unit floor (1 tile per unit).
    """
    # Step 1: Access the underlying Blender object
    blender_obj = kubric_obj.linked_objects[renderer]

    # Step 2: Create a new material with nodes enabled
    mat = bpy.data.materials.new(name="floor_textured_material")
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    # Step 3: Clear default nodes for a clean slate
    nodes.clear()

    # Step 4: Create shader nodes
    # Texture Coordinate node - provides UV coordinates
    node_texcoord = nodes.new(type="ShaderNodeTexCoord")
    node_texcoord.location = (-800, 300)

    # Mapping node - scales/transforms UVs for tiling
    node_mapping = nodes.new(type="ShaderNodeMapping")
    node_mapping.location = (-600, 300)
    # Set the tiling scale (how many times texture repeats)
    node_mapping.inputs["Scale"].default_value = (tiling_scale, tiling_scale, 1.0)

    # Image Texture node - loads and samples the texture
    node_texture = nodes.new(type="ShaderNodeTexImage")
    node_texture.location = (-300, 300)
    # Load the image
    img = bpy.data.images.load(texture_path)
    node_texture.image = img
    # CRITICAL: Set extension to REPEAT for seamless tiling
    node_texture.extension = "REPEAT"

    # Principled BSDF node - physically-based shader
    node_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    node_bsdf.location = (100, 300)
    # High roughness to minimize reflections
    node_bsdf.inputs["Roughness"].default_value = 0.9
    node_bsdf.inputs["Metallic"].default_value = 0.0

    # Material Output node
    node_output = nodes.new(type="ShaderNodeOutputMaterial")
    node_output.location = (400, 300)

    # Step 5: Connect the nodes
    # TexCoord[Generated] -> Mapping[Vector]
    # Using "Generated" coordinates which are auto-normalized to object bounds
    links.new(node_texcoord.outputs["Generated"], node_mapping.inputs["Vector"])

    # Mapping[Vector] -> TexImage[Vector]
    links.new(node_mapping.outputs["Vector"], node_texture.inputs["Vector"])

    # TexImage[Color] -> PrincipledBSDF[Base Color]
    links.new(node_texture.outputs["Color"], node_bsdf.inputs["Base Color"])

    # PrincipledBSDF[BSDF] -> Output[Surface]
    links.new(node_bsdf.outputs["BSDF"], node_output.inputs["Surface"])

    # Step 6: Assign material to the Blender object
    if blender_obj.data.materials:
        blender_obj.data.materials[0] = mat
    else:
        blender_obj.data.materials.append(mat)

    print(f"[INFO] Applied texture '{texture_path}' to floor with tiling scale {tiling_scale}")


def setup_scene() -> kb.Scene:
    """Create and configure the Kubric scene."""
    scene = kb.Scene(
        resolution=(512, 512),
        frame_start=0,
        frame_end=1,  # Single frame for static render
        frame_rate=30,
    )
    return scene


def setup_lighting(scene: kb.Scene) -> None:
    """Add directional and ambient lighting to the scene."""
    # Directional light (key light) - simulates sun
    directional_light = kb.DirectionalLight(
        name="key_light",
        position=(5, -5, 10),
        look_at=(0, 0, 0),
        intensity=1.5,
        color=kb.Color(1.0, 1.0, 1.0),
    )
    scene.add(directional_light)

    # Ambient light for fill
    scene.ambient_illumination = kb.Color(0.3, 0.3, 0.3)

    print("[INFO] Lighting configured: DirectionalLight + Ambient")


def setup_camera(scene: kb.Scene) -> None:
    """Configure a perspective camera looking down at the floor."""
    camera = kb.PerspectiveCamera(
        name="main_camera",
        position=(8, -8, 12),
        look_at=(0, 0, 0),
        focal_length=35,
    )
    scene.camera = camera
    print(f"[INFO] Camera positioned at {camera.position}")


def create_floor(scene: kb.Scene) -> kb.Cube:
    """
    Create the floor as a flat cube.

    The floor uses scale (10, 10, 0.1) which creates a 20x20 unit area
    (Kubric cubes are 2 units per side at scale 1).
    """
    floor = kb.Cube(
        name="floor",
        scale=(10, 10, 0.1),  # 20x20 area, 0.2 thick
        position=(0, 0, -0.1),  # Slightly below origin
        static=True,  # Fixed in place for physics
    )
    floor.friction = 0.5
    floor.restitution = 0.5
    scene.add(floor)
    print(f"[INFO] Floor created: scale={floor.scale}, position={floor.position}")
    return floor


def create_reference_object(scene: kb.Scene) -> kb.Sphere:
    """Add a red sphere on the floor for visual scale reference."""
    sphere = kb.Sphere(
        name="reference_sphere",
        scale=(0.5, 0.5, 0.5),  # 1 unit diameter
        position=(2, 2, 0.5),  # Sitting on the floor
        static=False,
        mass=1.0,
    )
    sphere.material = kb.PrincipledBSDFMaterial(
        name="red_material",
        color=kb.Color(0.8, 0.1, 0.1),  # Red
        roughness=0.5,
        metallic=0.1,
    )
    sphere.friction = 0.5
    sphere.restitution = 0.3
    scene.add(sphere)
    print(f"[INFO] Reference sphere created at {sphere.position}")
    return sphere


def main():
    """Main execution function."""
    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    texture_path = repo_root / "generate_grid_image" / "black_grid.png"
    output_dir = repo_root / "experiments" / "output" / "test_floor_texture"

    # Verify texture file exists
    if not texture_path.exists():
        raise FileNotFoundError(
            f"Texture file not found: {texture_path}\n"
            f"Run: python generate_grid_image/generate_black_grid_png.py --out {texture_path}"
        )

    print(f"[INFO] Texture path: {texture_path}")
    print(f"[INFO] Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Initialize scene
    scene = setup_scene()
    print(f"[INFO] Scene created: resolution={scene.resolution}")

    # Step 2: Setup camera
    setup_camera(scene)

    # Step 3: Setup lighting
    setup_lighting(scene)

    # Step 4: Create floor
    floor = create_floor(scene)

    # Step 5: Create reference object
    reference = create_reference_object(scene)

    # Step 6: Initialize renderer (MUST be before texture application)
    renderer = KubricRenderer(
        scene,
        scratch_dir=str(output_dir / ".scratch"),
        adaptive_sampling=False,
        use_denoising=True,
        samples_per_pixel=64,
    )
    print("[INFO] Renderer initialized")

    # Step 7: Apply texture to floor (AFTER renderer initialization)
    # Tiling scale of 20.0 means 20 tiles across the 20-unit floor = 1 tile per unit
    apply_texture_to_floor(
        kubric_obj=floor,
        renderer=renderer,
        texture_path=str(texture_path),
        tiling_scale=20.0,
    )

    # Step 8: Initialize simulator
    simulator = KubricSimulator(scene)
    print("[INFO] Simulator initialized")

    # Step 9: Run physics simulation (brief, just to settle objects)
    simulator.run(frame_start=0, frame_end=1)
    print("[INFO] Physics simulation complete")

    # Step 10: Render the frame
    print("[INFO] Rendering frame...")
    frames_dict = renderer.render()

    # Step 11: Save the rendered image
    kb.write_png(
        frames_dict["rgba"][0],
        str(output_dir / "textured_floor_render.png"),
    )
    print(f"[INFO] Render saved to: {output_dir / 'textured_floor_render.png'}")

    # Also save depth for debugging if needed (normalize to [0,1] range)
    import numpy as np
    depth = frames_dict["depth"][0]
    # Filter out infinite values and normalize
    finite_depth = depth[np.isfinite(depth)]
    if len(finite_depth) > 0:
        depth_min, depth_max = finite_depth.min(), finite_depth.max()
        depth_normalized = np.clip((depth - depth_min) / (depth_max - depth_min + 1e-8), 0, 1)
        kb.write_png(
            depth_normalized,
            str(output_dir / "textured_floor_depth.png"),
        )
        print(f"[INFO] Depth saved to: {output_dir / 'textured_floor_depth.png'}")
    else:
        print("[WARNING] Could not save depth - no finite values")

    print("[SUCCESS] Textured floor demo complete!")


if __name__ == "__main__":
    main()
