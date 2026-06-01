import json
import glob
import re
import cv2
import numpy as np
import os
import concurrent.futures
import math

from image_processing import get_grid_metadata
from ellipse_detection import detect_reference_holes, analyze_ellipses
from data_export import export_json, export_histogram, export_angle_histogram_from_bins, export_highlighted_image, export_valid_ellipses_histogram, export_global_heatmap, export_csv, export_mosaic_index
from update_parent_mosaics import build_mosaic_mapping, patch_json

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def select_reference_image_ui(image_dir, reference_holes):
    img_names = sorted(list(reference_holes.keys()))
    if not img_names:
        return None
        
    current_idx = 0
    selected_image = None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)
    
    # Text annotation for status
    txt = fig.text(0.5, 0.95, '', ha='center', va='top', fontsize=12, fontweight='bold')
    
    def draw_current():
        ax.clear()
        img_name = img_names[current_idx]
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            # Convert BGR to RGB for matplotlib
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hole = reference_holes[img_name]
            
            # Draw the detected hole in Red
            cv2.circle(img, (hole['x'], hole['y']), hole['radius'], (255, 0, 0), 10)
            ax.imshow(img)
            
            # Update title
            txt.set_text(f"Image {current_idx+1}/{len(img_names)}: {img_name}\nHole: r={hole['radius']} at ({hole['x']},{hole['y']})")
        ax.axis('off')
        fig.canvas.draw_idle()
        
    def next_img(event):
        nonlocal current_idx
        current_idx = (current_idx + 1) % len(img_names)
        draw_current()
        
    def prev_img(event):
        nonlocal current_idx
        current_idx = (current_idx - 1) % len(img_names)
        draw_current()
        
    def select_img(event):
        nonlocal selected_image
        selected_image = img_names[current_idx]
        plt.close(fig)
        
    axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
    axselect = plt.axes([0.45, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
    
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(next_img)
    
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(prev_img)
    
    bselect = Button(axselect, 'Select')
    bselect.on_clicked(select_img)
    
    draw_current()
    plt.show()
    
    return selected_image

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
    #!export_json(red_ellipses, f"{base_name}_ellipses_red.json", save_folder)
    #!export_angle_histogram_from_bins(ellipse_histogram, config["element"], base_name, save_folder)
    #!export_highlighted_image(image_path, ellipses_data, base_name, save_folder)
    
    return base_name, red_ellipses

def main():
    print("Starting the raw image processing...")
    
    # Load configuration parameters
    with open('config.json', 'r') as f:
        config = json.load(f)

    # 1. METADATA: Retrieve grid dimensions without assembling images
    print("Scanning directory for grid metadata...")
    metadata = get_grid_metadata(config)
    save_folder = os.path.join(config["save_folder"], config["element"])
    target_dir = os.path.join(config["folder_path"], config["element"])
    element = config["element"]
  
    # Create the save folder if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # 2. REFERENCE POINT DETECTION
    print("\nStarting automated reference hole detection...")
    reference_holes = detect_reference_holes(target_dir, config)
    
    if not reference_holes:
        raise ValueError("No reference holes found in the directory! Please check the parameters or images.")
        
    print(f"\nFound {len(reference_holes)} reference hole(s). Opening UI window to choose...")
    
    ref_image_name = select_reference_image_ui(target_dir, reference_holes)
    
    if not ref_image_name:
        print("Selection cancelled or failed. Exiting.")
        return

    # Retrieve the automatically detected coordinates
    ref_x0 = reference_holes[ref_image_name]["x"]
    ref_y0 = reference_holes[ref_image_name]["y"]
    ref_image_path = os.path.join(target_dir, ref_image_name)
    print(f"Reference point automatically selected at coordinates: x={ref_x0}, y={ref_y0}")
 
    # Extract i and j indices from the reference image name
    match_ref = re.search(r"MoEDAL-(\d{3})-(\d{3})\.png", ref_image_name)
    if match_ref:
        j_ref = int(match_ref.group(1))
        i_ref = int(match_ref.group(2))
    else:
        raise ValueError("Reference image name does not match expected pattern 'MoEDAL-xxx-yyy.png'.")

    # Retrieve physical dimensions from the raw reference image
    ref_img = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
    img_height, img_width = ref_img.shape[:2]
    print(f"Raw image dimensions detected: {img_width}x{img_height} pixels")

    # 3. AUTOMATIC SECOND REFERENCE POINT (R2) DETECTION
    print("\nAutomatically searching for the second reference hole (R2)...")
    max_dist_sq = 0.0
    r2_name = None
    r2_x_cart = 0.0
    r2_y_cart = 0.0

    for name, hole in reference_holes.items():
        if name == ref_image_name:
            continue # Skip R1 (the origin)
            
        # 1. Extract indices of the current hole's image
        match = re.search(r"MoEDAL-(\d{3})-(\d{3})\.png", name)
        j_curr = int(match.group(1))
        i_curr = int(match.group(2))
        
        # 2. Calculate raw global coordinates (Image coordinate system: Y pointing down)
        gx_raw = (j_ref - j_curr) * img_width + hole["x"] - ref_x0
        gy_raw = (i_ref - i_curr) * img_height + hole["y"] - ref_y0
        
        # 3. Convert to Standard Cartesian coordinate system (Y pointing up) for Profilometer trigonometry
        x_cart = gx_raw
        y_cart = -gy_raw
        
        # 4. Calculate squared distance to find the farthest hole
        dist_sq = x_cart**2 + y_cart**2
        
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            r2_name = name
            r2_x_cart = x_cart
            r2_y_cart = y_cart

    if not r2_name:
        raise ValueError("Error: Could not find a second reference hole (R2) on the foil.")

    # 5. Calculate theoretical angle and physical distance
    theta_code_rad = math.atan2(r2_y_cart, r2_x_cart)
    pixel_resolution = config.get("pixel_resolution", 1.75)
    dist_theoretical_um = math.sqrt(max_dist_sq) * pixel_resolution

    print(f"-> R2 automatically selected: {r2_name}")
    print(f"-> Theoretical foil angle (theta_code): {math.degrees(theta_code_rad):.3f}°")
    print(f"-> Theoretical distance R1-R2: {dist_theoretical_um / 1000:.3f} mm")
    
    # 6. Structure metadata for JSON export
    reference_system = {
        "r1_image": ref_image_name,
        "r2_image": r2_name,
        "theta_code_radians": float(theta_code_rad),
        "distance_R1_R2_um": float(dist_theoretical_um)
    }   

    # 4. PARALLEL DETECTION ON ALL RAW IMAGES
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

    # 5. GLOBAL EXPORTS
    print("Generating global JSON export for all red ellipses...")
    export_json(global_red_ellipses_data, element, save_folder, reference_system)

    print("Filling parent_mosaic fields...")
    json_path = os.path.join(save_folder, f"all_data_{element}.json")
    mapping = build_mosaic_mapping(target_dir)
    patch_json(json_path, mapping)

    print("Generating CSV export...")
    export_csv(json_path, element, save_folder)

    print("Generating mosaic index...")
    export_mosaic_index(json_path, element, save_folder)

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