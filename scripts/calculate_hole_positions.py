import json
import os
import re

def calculate_global_coords():
    # 1. Load config
    config_path = "/Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    element = config.get("element", "O1_L8_ME18_UD")
    folder_path = config.get("folder_path", "/Users/julesloisance/Desktop/StageHelsinki/MoEDAL_Data_Apr2025/")
    
    # Overlap parameters
    C_x = config.get("crop_width_X", 667)
    C_y = config.get("crop_height_Y", 323.5)
    
    # Raw image dimensions
    w_raw = 3840
    h_raw = 2748
    
    # Usable grid ranges (from build_mosaics.py)
    i_min, i_max = 34, 62
    j_min, j_max = 34, 55
    
    num_rows = (i_max - i_min + 1)
    num_cols = (j_max - j_min + 1)
    
    num_row_blocks = num_rows // 3  # 9 blocks
    num_col_blocks = num_cols // 3  # 7 blocks
    
    new_row_max = i_min + (num_row_blocks * 3) - 1 # 60
    new_column_max = j_min + (num_col_blocks * 3) - 1 # 54
    
    # Dimension of a single cropped image
    w_crop = w_raw - C_x # 3840 - 667 = 3173
    h_crop = h_raw - C_y # 2748 - 323.5 = 2424.5
    
    # Dimension of a single 3x3 mosaic
    w_mosaic = 3 * w_crop # 9519
    h_mosaic = 3 * int(round(h_crop)) # 3 * 2425 = 7275 or 7272 depending on rounding
    # Let's check: min_height = 2748, min_height - crop_height_Y = 2748 - 323.5 = 2424.5
    # int(round(2424.5)) = 2424 in Python 3 (banker's rounding) or 2425?
    # Let's verify by calculating int(round(2424.5))
    h_single_crop = int(round(h_raw - C_y)) # 2425 or 2424
    h_mosaic = 3 * h_single_crop
    
    # Canvas dimensions (unscaled)
    max_x = num_row_blocks # 9
    max_y = num_col_blocks # 7
    canvas_w = max_y * w_mosaic # 7 * w_mosaic
    canvas_h = max_x * h_mosaic # 9 * h_mosaic
    
    # Load reference holes from cache
    cache_path = os.path.join(folder_path, element, "reference_holes.json")
    with open(cache_path, "r") as f:
        holes = json.load(f)
        
    print("Coordinates of reference holes in the full-resolution stitched image (scale = 1.0):")
    print("-" * 110)
    print(f"{'Image Name':<20} | {'Raw X':<6} | {'Raw Y':<6} | {'Mosaic (X, Y)':<13} | {'Global X (px)':<15} | {'Global Y (px)':<15} | {'Stitched Map X (scale 0.1)':<25} | {'Stitched Map Y (scale 0.1)':<25}")
    print("-" * 110)
    
    for name, hole in sorted(holes.items()):
        # Parse image row i and column j
        match = re.search(r"MoEDAL-(\d{3})-(\d{3})\.png", name)
        if not match:
            continue
        i = int(match.group(1))
        j = int(match.group(2))
        
        # Check if inside the usable 3x3 grid range
        if not (i_min <= i <= new_row_max and j_min <= j <= new_column_max):
            continue
            
        cx, cy = hole["x"], hole["y"]
        
        # Filter duplicates in crop zones
        if cx < C_x or cy > h_raw - C_y:
            continue
            
        # 1. Determine which 3x3 mosaic block this image belongs to
        block_i = (i - i_min) // 3
        block_j = (j - j_min) // 3
        
        # X and Y labels of the mosaic (1-indexed)
        X_mosaic = block_i + 1
        Y_mosaic = block_j + 1
        
        # 2. Local coordinates inside the 3x3 mosaic
        # Local position in the 3x3 grid:
        r = (i - i_min) % 3
        c = (j - j_min) % 3
        
        # Coordinates of the cropped image inside the raw image
        local_x_in_img = cx - C_x
        local_y_in_img = cy
        
        # Paste rules in build_mosaics.py:
        # Row images assembly: row_images[r] stacked bottom to top
        #   r=2 is top row of block (pasted at y=0)
        #   r=1 is middle row of block (pasted at y=h_single_crop)
        #   r=0 is bottom row of block (pasted at y=2*h_single_crop)
        local_y_in_mosaic = (2 - r) * h_single_crop + local_y_in_img
        
        # Column images assembly: img3, img2, img1 pasted right to left
        #   c=2 is leftmost col of block (pasted at x=0)
        #   c=1 is middle col of block (pasted at x=w_crop)
        #   c=0 is rightmost col of block (pasted at x=2*w_crop)
        local_x_in_mosaic = (2 - c) * w_crop + local_x_in_img
        
        # 3. Global coordinates in the unscaled canvas
        # Stitch rules in stitch_foil.py:
        #   row_idx = max_x - X_mosaic
        #   col_idx = max_y - Y_mosaic
        row_idx = max_x - X_mosaic
        col_idx = max_y - Y_mosaic
        
        global_x = col_idx * w_mosaic + local_x_in_mosaic
        global_y = row_idx * h_mosaic + local_y_in_mosaic
        
        # Scale 0.1 coordinates (rounded to nearest pixel)
        map_x = int(round(global_x * 0.1))
        map_y = int(round(global_y * 0.1))
        
        print(f"{name:<20} | {cx:<6} | {cy:<6} | Mosaic_{X_mosaic}_{Y_mosaic:<6} | {global_x:<15.1f} | {global_y:<15.1f} | {map_x:<25} | {map_y:<25}")

if __name__ == "__main__":
    calculate_global_coords()
