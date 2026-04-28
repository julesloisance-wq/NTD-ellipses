import os
import re
import glob
from PIL import Image
from tqdm import tqdm

def process_and_build_mosaics(config):
    folder_path = config["folder_path"]
    element = config["element"]
    save_folder = os.path.join(config["save_folder"], element)
    os.makedirs(save_folder, exist_ok=True)
    
    crop_width_X = config["crop_width_X"]
    crop_height_Y = config["crop_height_Y"]
    step = config["step"]
    num_columns = config["num_columns"]
    num_rows = config["num_rows"]

    # Regular expression to extract i and j values from filenames
    pattern = re.compile(r'MoEDAL-(\d{3})-(\d{3})\.png') 

    # Lists to store grid coordinates
    i_values = []
    j_values = []

    total_width = num_columns * step
    total_height = num_rows * step

    # Loop through all files in the folder to find the grid boundaries
    for filename in os.listdir(folder_path + element):
        match = pattern.match(filename)
        if match:
            i_values.append(int(match.group(1)))
            j_values.append(int(match.group(2)))
            
    # Compute min and max if matching files were found
    if i_values and j_values:
        i_min, i_max = min(i_values), max(i_values)
        j_min, j_max = min(j_values), max(j_values)
    else:
        return save_folder
        
    i_center = int((i_min + i_max) / 2)
    j_center = int((j_min + j_max) / 2)

    # Define the mosaic borders properly centered
    i_min = i_center - (total_height // 2)
    i_max = i_min + total_height - 1 

    j_min = j_center - (total_width // 2)
    j_max = j_min + total_width - 1  

    num_row_operations = (i_max + 1 - i_min) // step
    num_column_operations = (j_max + 1 - j_min) // step

    new_row_min = i_min
    new_column_min = j_min
    new_row_max = i_min + num_row_operations * step - 1
    new_column_max = j_min + num_column_operations * step - 1

    # Fast pass: determine the minimum height without loading full pixel data into RAM
    heights = []
    for img_path in glob.glob(f"{folder_path}{element}/MoEDAL-*.png"):
        with Image.open(img_path) as img:
            heights.append(img.size[1])
    new_height = min(heights) if heights else 1000

    # Delete old mosaics and associated files
    for file in glob.glob(os.path.join(save_folder, "Mosaic*.*")):
        os.remove(file)

    print("\nGenerating mosaics in memory-efficient batches...")
    
    # Batch processing: load, resize, and assemble one mosaic at a time
    for i in tqdm(range(new_row_min, new_row_max + 1, step), desc="Mosaic rows"):
        for j in range(new_column_min, new_column_max + 1, step):
            row_images = [[] for _ in range(step)]
            
            # Build the sub-image matrix for the current mosaic
            for k in range(step):
                for l in range(step):
                    img_path = f"{folder_path}{element}/MoEDAL-{i+k:03}-{j+l:03}.png"
                    
                    if os.path.exists(img_path):
                        with Image.open(img_path) as img:
                            # Resize to the minimum height while keeping aspect ratio
                            new_width = int(img.size[0] * new_height / img.size[1])
                            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                            
                            # Systematic cropping at the bottom and left edges
                            resized_img = resized_img.crop((0, 0, resized_img.size[0], new_height - crop_height_Y))
                            resized_img = resized_img.crop((crop_width_X, 0, resized_img.size[0], resized_img.size[1]))
                            
                            row_images[k].append(resized_img)
            
            # Assemble the current mosaic if all required sub-images are valid
            if all(len(row) == step for row in row_images):
                assembled_rows = []
                
                # Assemble horizontally
                for k in range(step):
                    target_row_height = row_images[k][0].size[1]
                    total_row_width = sum(img.size[0] for img in row_images[k])
                    
                    new_row = Image.new('RGB', (total_row_width, target_row_height), (255, 255, 255))
                    
                    x_offset = 0
                    # Paste images in reverse order to match original logic
                    for img in reversed(row_images[k]):
                        new_row.paste(img, (x_offset, 0))
                        x_offset += img.size[0]
                        
                    assembled_rows.append(new_row)
                    
                # Assemble vertically
                total_final_height = sum(row.size[1] for row in assembled_rows)
                final_width = max(row.size[0] for row in assembled_rows)
                final_image = Image.new('RGB', (final_width, total_final_height), (255, 255, 255))
                
                # Calculate Y offsets from bottom to top
                y_offsets = [0, assembled_rows[0].size[1], assembled_rows[0].size[1] + assembled_rows[1].size[1]]
                y_offsets = y_offsets[::-1] 
                
                for k in range(2, -1, -1):
                    final_image.paste(assembled_rows[k], (0, y_offsets[k]))
                    
                # Save image immediately to free system memory for the next iteration
                filename = f"Mosaic_{int((i-new_row_min)/step)+1}_{int((j-new_column_min)/step)+1}.png"
                final_image.save(os.path.join(save_folder, filename))
                
    return save_folder