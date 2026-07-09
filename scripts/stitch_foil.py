import os
import glob
import re
import json
from PIL import Image

def main():
    # 1. Load config.json
    config_path = "config.json"
    if not os.path.exists(config_path):
        # If run from scripts/ directory
        config_path = os.path.join("..", "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        print("config.json not found. Using default paths.")
        config = {
            "folder_path": "/Users/julesloisance/Desktop/StageHelsinki/MoEDAL_Data_Apr2025/",
            "element": "O1_L8_ME18_UD"
        }

    element = config.get("element", "O1_L8_ME18_UD")
    folder_path = config.get("folder_path", "/Users/julesloisance/Desktop/StageHelsinki/MoEDAL_Data_Apr2025/")
    
    mosaics_dir = os.path.join(folder_path, element, "Mosaics")
    if not os.path.exists(mosaics_dir):
        print(f"Error: Mosaics folder not found at {mosaics_dir}")
        return

    # 2. Find all mosaics and determine the grid size (max_X, max_Y)
    pattern = re.compile(r"Mosaic_(\d+)_(\d+)\.png")
    mosaics = []
    max_x = 0
    max_y = 0

    for file in os.listdir(mosaics_dir):
        match = pattern.match(file)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            mosaics.append((x, y, file))
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

    if not mosaics:
        print("No mosaics found matching the pattern Mosaic_X_Y.png.")
        return

    print(f"Found {len(mosaics)} mosaics.")
    print(f"Detected Grid size: Rows (X) up to {max_x} | Columns (Y) up to {max_y}")

    # 3. Determine the size of a single mosaic image
    sample_file = os.path.join(mosaics_dir, mosaics[0][2])
    with Image.open(sample_file) as img:
        orig_w, orig_h = img.size
    print(f"Original mosaic dimension: {orig_w}x{orig_h} pixels")

    # To avoid out-of-memory errors and excessive file sizes, we resize the mosaics.
    # We target a reasonable scale factor (e.g. 0.1) which yields a detailed yet light final image.
    scale = 0.1
    scaled_w = int(orig_w * scale)
    scaled_h = int(orig_h * scale)
    print(f"Scaled mosaic dimension (scale={scale}): {scaled_w}x{scaled_h} pixels")

    # Final image dimensions
    final_w = max_y * scaled_w
    final_h = max_x * scaled_h
    print(f"Final stitched image dimension: {final_w}x{final_h} pixels")

    # Create the blank canvas
    canvas = Image.new("RGB", (final_w, final_h), (255, 255, 255))

    # 4. Stitch mosaics together
    # Grid mapping rules:
    # - Row position: X goes from 1 (bottom) to max_x (top).
    #   So the row index on the canvas (from top = 0 to bottom = max_x - 1) is: row_idx = max_x - X.
    # - Col position: Y goes from 1 (right) to max_y (left).
    #   So the column index on the canvas (from left = 0 to right = max_y - 1) is: col_idx = max_y - Y.
    
    print("Stitching mosaics...")
    for x, y, filename in mosaics:
        filepath = os.path.join(mosaics_dir, filename)
        row_idx = max_x - x
        col_idx = max_y - y
        
        pos_x = col_idx * scaled_w
        pos_y = row_idx * scaled_h
        
        try:
            with Image.open(filepath) as img:
                resized = img.resize((scaled_w, scaled_h), Image.LANCZOS)
                canvas.paste(resized, (pos_x, pos_y))
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 5. Save the result in report_figures
    # Ensure report_figures directory exists
    output_dir = "report_figures"
    if not os.path.exists(output_dir):
        output_dir = os.path.join("..", "report_figures")
        if not os.path.exists(output_dir):
            output_dir = "report_figures"
            os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "full_foil_mosaic.png")
    print(f"Saving stitched image to {output_path}...")
    canvas.save(output_path, "PNG")
    print("✅ Stitching complete!")

if __name__ == "__main__":
    main()
