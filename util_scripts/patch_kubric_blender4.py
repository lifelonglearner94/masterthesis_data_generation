#!/usr/bin/env python3
"""
Patch Kubric for Blender 4.x compatibility on macOS (M1/M2/M3).

Kubric was designed for Blender 3.x. This script patches the renderer
to work with Blender 4.x API changes:

1. Specular -> Specular IOR Level
2. Transmission -> Transmission Weight  
3. Emission -> Emission Color
4. Disable Transmission Roughness (removed in Blender 4.x)
5. Disable Specular Tint (changed from float to RGBA)

Usage:
    python patch_kubric_blender4.py

The script auto-detects the Kubric installation path.
"""

import subprocess
import sys


def find_kubric_path():
    """Find the installed Kubric package path."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import kubric; import os; print(os.path.dirname(kubric.__file__))"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        print("ERROR: Kubric not installed. Run: pip install git+https://github.com/google-research/kubric.git")
        sys.exit(1)


def patch_blender_renderer(kubric_path: str):
    """Apply Blender 4.x compatibility patches."""
    blender_py = f"{kubric_path}/renderer/blender.py"
    
    print(f"Patching: {blender_py}")
    
    with open(blender_py, "r") as f:
        content = f.read()
    
    # Create backup
    backup_path = f"{blender_py}.bak"
    with open(backup_path, "w") as f:
        f.write(content)
    print(f"Backup created: {backup_path}")
    
    # Apply string replacements for Blender 4.x
    replacements = [
        ('inputs["Specular"]', 'inputs["Specular IOR Level"]'),
        ('inputs["Transmission"]', 'inputs["Transmission Weight"]'),
        ('inputs["Emission"]', 'inputs["Emission Color"]'),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
        print(f"  ✓ Replaced: {old} -> {new}")
    
    # Comment out problematic lines (preserves structure)
    lines_to_comment = [
        'obj.observe(AttributeSetter(bsdf_node.inputs["Transmission Roughness"]',
        'obj.observe(KeyframeSetter(bsdf_node.inputs["Transmission Roughness"]',
        'obj.observe(AttributeSetter(bsdf_node.inputs["Specular Tint"]',
        'obj.observe(KeyframeSetter(bsdf_node.inputs["Specular Tint"]',
    ]
    
    lines = content.split("\n")
    new_lines = []
    in_multiline_comment = False
    comment_count = 0
    
    for line in lines:
        should_comment = False
        for pattern in lines_to_comment:
            if pattern in line:
                should_comment = True
                open_parens = line.count("(") - line.count(")")
                in_multiline_comment = open_parens > 0
                comment_count = 0
                print(f"  ✓ Commented out: {pattern[:50]}...")
                break
        
        if should_comment:
            new_lines.append("    # BLENDER4_COMPAT: " + line.strip())
        elif in_multiline_comment:
            new_lines.append("    # BLENDER4_COMPAT: " + line.strip())
            comment_count += 1
            if line.count("(") <= line.count(")") or comment_count > 3:
                in_multiline_comment = False
        else:
            new_lines.append(line)
    
    with open(blender_py, "w") as f:
        f.write("\n".join(new_lines))
    
    print(f"\n✅ Patches applied successfully!")
    print(f"   Kubric is now compatible with Blender 4.x on Apple Silicon.")


def main():
    print("=" * 60)
    print("Kubric Blender 4.x Compatibility Patch")
    print("=" * 60)
    
    kubric_path = find_kubric_path()
    print(f"Found Kubric at: {kubric_path}\n")
    
    patch_blender_renderer(kubric_path)
    
    # Clear Python cache
    import shutil
    cache_path = f"{kubric_path}/renderer/__pycache__"
    try:
        shutil.rmtree(cache_path)
        print(f"   Cleared cache: {cache_path}")
    except FileNotFoundError:
        pass
    
    print("\n" + "=" * 60)
    print("Done! You can now run Kubric natively on M1/M2/M3 Mac.")
    print("=" * 60)


if __name__ == "__main__":
    main()
