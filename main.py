import json
import glob
import re
import cv2
import numpy as np
import os
import concurrent.futures
from image_processing import process_and_build_mosaics
from ellipse_detection import get_reference_center, analyze_ellipses
from data_export import export_json, export_histogram, export_angle_histogram_from_bins, export_highlighted_mosaic, export_mosaics_histogram, export_global_heatmap

def process_single_mosaic(mosaic_path, config, ref_x0, ref_y0, i_ref, j_ref, mosaic_width, mosaic_height, save_folder):
    """
    Processes a single mosaic and exports its specific files. 
    Extracted as an independent function to enable multiprocessing.
    """
    ellipses_data, ellipse_histogram, dominant_angle = analyze_ellipses(mosaic_path, config, ref_x0, ref_y0, i_ref, j_ref, mosaic_width, mosaic_height)
    
    # Filter to retain only 'red' category for JSON export and area histogram
    red_ellipses = [e for e in ellipses_data if e["category"] == "red"]
    
    base_name = os.path.basename(mosaic_path).replace(".png", "")
    
    # Execute individual exports for this mosaic
    export_json(red_ellipses, f"{base_name}_ellipses_red.json", save_folder)
    export_angle_histogram_from_bins(ellipse_histogram, config["element"], base_name, save_folder)
    export_highlighted_mosaic(mosaic_path, ellipses_data, base_name, save_folder)
    
    return base_name, red_ellipses

def main():
    print("Starting the processing...")
    
    # Load configuration parameters
    with open('config.json', 'r') as f:
        config = json.load(f)

    # 1. PROCESSING: Build mosaics using memory-efficient batches
    save_folder = process_and_build_mosaics(config)
    print("Mosaics created and saved in:", save_folder)

    # 2. REFERENCE POINT SELECTION
    print("\nAction required: Please select the reference mosaic and click on the center of the hole.")
    print(save_folder)

    # Loop until a valid existing file is provided by the user
    while True:
        ref_mosaic_name = input("Enter the name of the reference mosaic (ex: Mosaic_2_1.png) : ")
        ref_mosaic_path = os.path.join(save_folder, ref_mosaic_name)
        
        if os.path.exists(ref_mosaic_path):
            break
        else:
            print(f"❌ The file '{ref_mosaic_name}' is not found. Please try again.")

    # Retrieve the reference center coordinates from the user click
    ref_x0, ref_y0 = get_reference_center(ref_mosaic_path)
    print(f"Reference point selected at coordinates: x={ref_x0}, y={ref_y0}")
 
    # Extract mosaic indices from the reference mosaic name to calculate global coordinates later
    match_ref = re.search(r"Mosaic_(\d+)_(\d+)", ref_mosaic_name)
    if match_ref:
        i_ref = int(match_ref.group(1))
        j_ref = int(match_ref.group(2))
    else:
        raise ValueError("Reference mosaic name does not match expected pattern 'Mosaic_i_j.png'.")

    # Retrieve mosaic dimensions from the reference image to calculate global coordinates later
    ref_img = cv2.imread(ref_mosaic_path, cv2.IMREAD_GRAYSCALE)
    mosaic_height, mosaic_width = ref_img.shape[:2]
    print(f"Mosaic dimensions detected: {mosaic_width}x{mosaic_height} pixels")

    # 3. PARALLEL DETECTION ON ALL MOSAICS
    mosaic_files = glob.glob(os.path.join(save_folder, "Mosaic_*.png"))
    global_red_ellipses_data = []
    mosaic_ellipse_counts = {}
    
    print("\nAnalyzing mosaics in parallel across CPU cores...")
    
    # Use ProcessPoolExecutor to distribute the workload
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks to the executor
        futures = [
            executor.submit(process_single_mosaic, path, config, ref_x0, ref_y0, i_ref, j_ref, mosaic_width, mosaic_height, save_folder)
            for path in mosaic_files
        ]
        
        # Collect results dynamically as each process completes
        for future in concurrent.futures.as_completed(futures):
            try:
                base_name, red_ellipses = future.result()
                global_red_ellipses_data.extend(red_ellipses)
                mosaic_ellipse_counts[base_name] = len(red_ellipses)
            except Exception as exc:
                print(f"An error occurred during analysis: {exc}")

    # 4. GLOBAL EXPORTS
    print("Generating global area histogram...")
    export_histogram(global_red_ellipses_data, config["element"], save_folder)
    
    print("Generating mosaics distribution histogram...")
    angle_tolerance = config.get("angle_tolerance", 5)
    export_mosaics_histogram(
        mosaic_ellipse_counts, 
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