import os
import shutil
import glob

# distinct paths relative to project root
SOURCE_REL_PATH = "experiments/output/friction_dataset_v2"
DEST_REL_PATH = "experiments/output/friction_dataset_v2_rgb_and_ground_truth"

def move_items():
    # Use current working directory as base
    base_dir = os.getcwd()
    
    source_dir = os.path.join(base_dir, SOURCE_REL_PATH)
    dest_dir = os.path.join(base_dir, DEST_REL_PATH)

    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return

    # Ensure destination directory exists
    if not os.path.exists(dest_dir):
        print(f"Creating destination directory: {dest_dir}")
        os.makedirs(dest_dir, exist_ok=True)

    # Pattern for clip folders
    search_pattern = os.path.join(source_dir, "clip_*")
    clip_path_list = glob.glob(search_pattern)
    
    # Filter only for directories
    clip_path_list = [p for p in clip_path_list if os.path.isdir(p)]
    
    clip_path_list.sort()
    
    print(f"Found {len(clip_path_list)} clip folders to process.")

    moved_count = 0
    
    for src_clip_path in clip_path_list:
        clip_name = os.path.basename(src_clip_path)
        dest_clip_path = os.path.join(dest_dir, clip_name)

        # 1. Create destination clip folder
        os.makedirs(dest_clip_path, exist_ok=True)

        items_moved = False

        # 2. Move rgb folder
        src_rgb = os.path.join(src_clip_path, "rgb")
        dest_rgb = os.path.join(dest_clip_path, "rgb")
        
        if os.path.exists(src_rgb):
            if os.path.exists(dest_rgb):
                print(f"Skipping rgb move for {clip_name}: Destination 'rgb' already exists.")
            else:
                shutil.move(src_rgb, dest_rgb)
                items_moved = True

        # 3. Move ground_truth.json
        src_gt = os.path.join(src_clip_path, "ground_truth.json")
        dest_gt = os.path.join(dest_clip_path, "ground_truth.json")

        if os.path.exists(src_gt):
            if os.path.exists(dest_gt):
                 print(f"Skipping ground_truth.json move for {clip_name}: Destination file already exists.")
            else:
                shutil.move(src_gt, dest_gt)
                items_moved = True

        if items_moved:
            moved_count += 1
            if moved_count % 1000 == 0:
                print(f"Processed {moved_count} folders...")

    print(f"Finished. Moved content for {moved_count} clips.")

if __name__ == "__main__":
    move_items()
