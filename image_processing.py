import os
import re

def get_grid_metadata(config):
    """
    Scans the raw images folder to determine the boundaries 
    of the acquisition grid (min and max coordinates).
    """
    folder_path = config["folder_path"]
    element = config["element"]
    save_folder = os.path.join(config["save_folder"], element)
    
    # Create the save folder if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # Regular expression to extract i and j indices from filenames
    pattern = re.compile(r'MoEDAL-(\d{3})-(\d{3})\.png') 

    i_values = []
    j_values = []

    # Iterate through all files in the target directory
    target_dir = os.path.join(folder_path, element)
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    for filename in os.listdir(target_dir):
        match = pattern.match(filename)
        if match:
            i_values.append(int(match.group(1)))
            j_values.append(int(match.group(2)))
            
    # Check if any valid images were found in the folder
    if not i_values or not j_values:
        raise ValueError(f"No images matching pattern 'MoEDAL-xxx-yyy.png' found in {target_dir}")

    # Calculate the grid boundaries
    i_min, i_max = min(i_values), max(i_values)
    j_min, j_max = min(j_values), max(j_values)
    
    print(f"Grid detected: Rows (i) from {i_min} to {i_max} | Columns (j) from {j_min} to {j_max}")

    # Pack all relevant metadata into a dictionary
    metadata = {
        "i_min": i_min,
        "i_max": i_max,
        "j_min": j_min,
        "j_max": j_max,
        "save_folder": save_folder,
        "target_dir": target_dir
    }

    return metadata