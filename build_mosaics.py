import os
import glob
import re
from PIL import Image
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import json

def main():
    parser = argparse.ArgumentParser(description="Build 3x3 Mosaics from MoEDAL grid images")
    parser.add_argument("--dir", default="/Users/julesloisance/Desktop/StageHelsinki/MoEDAL_Data_Apr2025/O1_L8_ME18_UD", help="Target directory containing the images")
    parser.add_argument("--draw-craters", action="store_true", help="Draw red circles around interesting craters on the mosaics")
    args = parser.parse_args()
    
    target_dir = args.dir
    save_folder = os.path.join(target_dir, "Mosaics")
    os.makedirs(save_folder, exist_ok=True)
    
    config = {}
    config_path = "config.json"
    if args.draw_craters:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            print(f"⚠️ Warning: {config_path} not found. Using default crater parameters.")
    
    print(f"Target directory: {target_dir}")
    print(f"Output folder: {save_folder}")
    
    # 1. Analyze Grid
    pattern = re.compile(r'MoEDAL-(\d{3})-(\d{3})\.png')
    i_values = []
    j_values = []
    
    for filename in os.listdir(target_dir):
        match = pattern.match(filename)
        if match:
            i_values.append(int(match.group(1)))
            j_values.append(int(match.group(2)))
            
    if not i_values or not j_values:
        print("⚠️ No matching files found in the directory.")
        return
        
    i_min, i_max = min(i_values), max(i_values)
    j_min, j_max = min(j_values), max(j_values)
    
    print(f"Grid detected: Rows (i) from {i_min} to {i_max} | Columns (j) from {j_min} to {j_max}")
    
    # Trim grid to multiple of 3
    num_rows = (i_max - i_min + 1)
    num_cols = (j_max - j_min + 1)
    
    num_row_blocks = num_rows // 3
    num_col_blocks = num_cols // 3
    
    new_row_max = i_min + (num_row_blocks * 3) - 1
    new_column_max = j_min + (num_col_blocks * 3) - 1
    
    print(f"Usable grid for 3x3 mosaics: Rows (i) {i_min} to {new_row_max} | Cols (j) {j_min} to {new_column_max}")
    print(f"Total 3x3 Mosaics to build: {num_row_blocks * num_col_blocks}")
    
    # Crop constants from empirical validation (Code.ipynb)
    crop_width_X = 655   # will be cropped from the left
    crop_height_Y = 295  # will be cropped from the bottom
    step = 3
    
    # Find the minimum height across the usable dataset to resize images identically
    # using img.size is RAM efficient as it only loads the header
    print("\nScanning image dimensions to find minimum height...")
    min_height = float('inf')
    for i in range(i_min, new_row_max + 1):
        for j in range(j_min, new_column_max + 1):
            path = os.path.join(target_dir, f"MoEDAL-{i:03}-{j:03}.png")
            if os.path.exists(path):
                with Image.open(path) as img:
                    if img.size[1] < min_height:
                        min_height = img.size[1]
                        
    if min_height == float('inf'):
        print("⚠️ No images could be opened.")
        return
        
    print(f"Minimum height determined for resizing: {min_height} pixels")
    
    # 2. Build Mosaics block by block
    total_blocks = num_row_blocks * num_col_blocks
    
    # Clean previous mosaics if they exist
    for file in glob.glob(os.path.join(save_folder, "Mosaic*.png")):
        os.remove(file)
        
    with tqdm(total=total_blocks, desc="Building Mosaics", unit="mosaic") as pbar:
        for block_i in range(num_row_blocks):
            for block_j in range(num_col_blocks):
                i_start = i_min + (block_i * 3)
                j_start = j_min + (block_j * 3)
                
                # Load, resize and crop the 9 images for this block
                images_3x3 = [] # 2D list: 3 rows, 3 cols
                for r in range(3):
                    row_imgs = []
                    for c in range(3):
                        curr_i = i_start + r
                        curr_j = j_start + c
                        path = os.path.join(target_dir, f"MoEDAL-{curr_i:03}-{curr_j:03}.png")
                        
                        if os.path.exists(path):
                            if args.draw_craters:
                                img = draw_interesting_craters(path, config)
                                if img is None:
                                    img = Image.open(path)
                            else:
                                img = Image.open(path)
                                
                            # Resize to min_height while maintaining aspect ratio
                            new_width = int(img.size[0] * min_height / img.size[1])
                            resized_img = img.resize((new_width, min_height), Image.LANCZOS)
                            
                            # Systematic cropping
                            # PIL crop tuple: (left, upper, right, lower)
                            cropped_img = resized_img.crop((crop_width_X, 0, resized_img.size[0], min_height - crop_height_Y))
                            row_imgs.append(cropped_img)
                            img.close()
                        else:
                            row_imgs.append(None)
                    images_3x3.append(row_imgs)
                
                # Assemble Horizontal Rows
                row_images = [None, None, None]
                for r in range(3):
                    img1 = images_3x3[r][0] # j
                    img2 = images_3x3[r][1] # j+1
                    img3 = images_3x3[r][2] # j+2
                    
                    if not all([img1, img2, img3]):
                        continue
                        
                    target_height = img1.size[1]
                    total_width = img1.size[0] + img2.size[0] + img3.size[0]
                    
                    new_row = Image.new('RGB', (total_width, target_height), (255, 255, 255))
                    
                    # Order from right to left as per scanner orientation (img3, img2, img1)
                    images_to_paste = [img3, img2, img1]
                    x_offset = 0
                    for img in images_to_paste:
                        new_row.paste(img, (x_offset, 0))
                        x_offset += img.size[0]
                        
                    row_images[r] = new_row
                    
                if None in row_images:
                    print(f"⚠️ Skipping mosaic {block_i+1}_{block_j+1} due to missing images.")
                    pbar.update(1)
                    continue
                    
                # Assemble Vertical Mosaic
                total_height = sum(row.size[1] for row in row_images)
                final_width = max(row.size[0] for row in row_images)
                final_image = Image.new('RGB', (final_width, total_height), (255, 255, 255))
                
                # Stack rows from bottom to top
                y_offsets = [0, row_images[0].size[1], row_images[0].size[1] + row_images[1].size[1]]
                y_offsets.reverse()
                
                for k in range(2, -1, -1):
                    final_image.paste(row_images[k], (0, y_offsets[k]))
                    
                # Save
                filename = f"Mosaic_{block_i+1}_{block_j+1}.png"
                final_image.save(os.path.join(save_folder, filename))
                
                # Cleanup RAM
                for r in range(3):
                    for img in images_3x3[r]:
                        if img:
                            img.close()
                            
                pbar.update(1)
                
    print(f"\n✅ All {total_blocks} mosaics have been successfully built and saved in {save_folder}")

def draw_interesting_craters(img_path, config):
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_color is None:
        return None
        
    green_channel = img_color[:, :, 1]
    blurred = cv2.GaussianBlur(green_channel, (5, 5), 0)
    
    lower = int(max(0, np.median(blurred) * 0.66))
    upper = int(max(0, np.median(blurred) * 1.33))
    edges = cv2.Canny(blurred, threshold1=lower, threshold2=upper)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = config.get("min_area", 20)
    max_area = config.get("max_area", 500)
    min_intensity = config.get("min_intensity", 0)
    max_intensity = config.get("max_intensity", 130)
    
    for cnt in contours:
        if len(cnt) < 5: continue
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area): continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.4: continue
        
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (w, h), angle = ellipse
        
        # Make the drawing ellipse significantly larger for visibility
        enlarged_ellipse = ((cx, cy), (w + 100, h + 100), angle)
        
        # intensity check (must use the original true ellipse for pixel extraction)
        mask = np.zeros_like(green_channel, dtype=np.uint8)
        cv2.ellipse(mask, ellipse, (255,), thickness=-1)
        masked_pixels = green_channel[mask == 255]
        mean_intensity = np.mean(masked_pixels) if masked_pixels.size > 0 else 0
        
        if min_intensity <= mean_intensity <= max_intensity:
            # Draw enlarged ellipse in RED (BGR: 0, 0, 255) with a very thick line
            cv2.ellipse(img_color, enlarged_ellipse, (0, 0, 255), 10)
            
    # Convert back to PIL Image (OpenCV uses BGR, PIL uses RGB)
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

if __name__ == "__main__":
    main()
