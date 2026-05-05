import json
import glob
import re
import cv2
import numpy as np
import os
import concurrent.futures

from image_processing import get_grid_metadata
from ellipse_detection import get_reference_center, analyze_ellipses
from data_export import export_json, export_histogram, export_angle_histogram_from_bins, export_highlighted_image, export_valid_ellipses_histogram, export_global_heatmap

def process_single_image(image_path, config, ref_x0, ref_y0, i_ref, j_ref, img_width, img_height, save_folder):
    """
    Processes a single raw image and exports its specific files. 
    Extracted as an independent function to enable multiprocessing.
    """

    ellipses_data, ellipse_histogram, dominant_angle = analyze_ellipses(image_path, config, ref_x0, ref_y0, i_ref, j_ref, img_width, img_height)
    
    # Filter to retain only 'red' category for JSON export and area histogram
    red_ellipses = [e for e in ellipses_data if e["category"] == "red"]
    
    base_name = os.path.basename(image_path).replace(".png", "")
    
    # Execute individual exports for this specific image
    export_json(red_ellipses, f"{base_name}_ellipses_red.json", save_folder)
    export_angle_histogram_from_bins(ellipse_histogram, config["element"], base_name, save_folder)
    
    export_highlighted_image(image_path, ellipses_data, base_name, save_folder)
    
    return base_name, red_ellipses

def main():
    print("Starting the raw image processing...")
    
    # Load configuration parameters
    with open('config.json', 'r') as f:
        config = json.load(f)

    # 1. METADATA: Retrieve grid dimensions without assembling images
    print("Scanning directory for grid metadata...")
    metadata = get_grid_metadata(config)
    save_folder = metadata["save_folder"]
    target_dir = metadata["target_dir"]

    # 2. REFERENCE POINT SELECTION
    print("\nAction required: Please select the reference image and click on the center of the hole.")
    print(f"Target directory: {target_dir}")

    # Loop until a valid existing raw file is provided by the user
    while True:
        ref_image_name = input("Enter the name of the reference image (ex: MoEDAL-001-002.png) : ")
        ref_image_path = os.path.join(target_dir, ref_image_name)
        
        if os.path.exists(ref_image_path):
            break
        else:
            print(f"❌ The file '{ref_image_name}' is not found. Please try again.")

    # Retrieve the reference center coordinates from the user click
    ref_x0, ref_y0 = get_reference_center(ref_image_path)
    print(f"Reference point selected at coordinates: x={ref_x0}, y={ref_y0}")
 
    # Extract i and j indices from the reference image name
    match_ref = re.search(r"MoEDAL-(\d{3})-(\d{3})\.png", ref_image_name)
    if match_ref:
        i_ref = int(match_ref.group(1))
        j_ref = int(match_ref.group(2))
    else:
        raise ValueError("Reference image name does not match expected pattern 'MoEDAL-xxx-yyy.png'.")

    # Retrieve physical dimensions from the raw reference image
    ref_img = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
    img_height, img_width = ref_img.shape[:2]
    print(f"Raw image dimensions detected: {img_width}x{img_height} pixels")

    # 3. PARALLEL DETECTION ON ALL RAW IMAGES
    image_files = glob.glob(os.path.join(target_dir, "MoEDAL-*.png"))
    global_red_ellipses_data = []
    image_ellipse_counts = {}
    
    print("\nAnalyzing raw images in parallel across CPU cores...")
    
    # Use ProcessPoolExecutor to distribute the workload
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_image, path, config, ref_x0, ref_y0, i_ref, j_ref, img_width, img_height, save_folder)
            for path in image_files
        ]
        
        # Collect results dynamically
        for future in concurrent.futures.as_completed(futures):
            try:
                base_name, red_ellipses = future.result()
                global_red_ellipses_data.extend(red_ellipses)
                image_ellipse_counts[base_name] = len(red_ellipses)
            except Exception as exc:
                print(f"An error occurred during analysis: {exc}")

    # 4. GLOBAL EXPORTS
    print("Generating global area histogram...")
    export_histogram(global_red_ellipses_data, config["element"], save_folder)
    
    print("Generating image distribution histogram...")
    angle_tolerance = config.get("angle_tolerance", 5)
    export_valid_ellipses_histogram(
        image_ellipse_counts, 
        config["element"], 
        save_folder, 
        config["min_intensity"], 
        config["max_intensity"], 
        angle_tolerance
    )

    print("Generating global spatial heatmap...")
    export_global_heatmap(global_red_ellipses_data, config["element"], save_folder)

    print("Processing completed successfully.")

if __name__ == "__main__":
    main()