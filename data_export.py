import csv
import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import json

def export_json(data, element, save_folder, reference_system=None):
    """Exports a list of dictionaries to JSON format."""
    master_json = { 
        "element_name" : element,
        "images" : {}
        }
    
    # Inject the profilometric coordinate system metadata at the root
    if reference_system:
        master_json["reference_system"] = reference_system
        
    # Fill in the JSON with clear parent_mosaic structure
    for ellipse in data:
        img_name = ellipse["image_source"]
        if img_name not in master_json["images"]:
            master_json["images"][img_name] = {
                "parent_mosaic": "tbd",
                "ellipses": []
            }
        ellipse_wo_img_name = {k: v for k, v in ellipse.items() if k != "image_source"} # Remove redundant image name from each ellipse entry
        master_json["images"][img_name]["ellipses"].append(ellipse_wo_img_name) 

    def extract_coords(filename):
        try:
            parts = filename.replace(".png", "").split("-")
            return int(parts[1]), int(parts[2])
        except Exception:  
            return 0, 0
            
    # Sort the dictionary keys to ensure the X-axis is logically ordered
    master_json["images"] = dict(sorted(master_json["images"].items(), key=lambda item: extract_coords(item[0])))

    filepath = os.path.join(save_folder, f"all_data_{element}.json")
    with open(filepath, 'w') as f:
        json.dump(master_json, f, indent=4)

    print(f"Base de données JSON globale exportée : {filepath}")

def export_angle_histogram_from_bins(ellipse_histogram, element_name, base_name, save_folder):
    """Generates the angle histogram using 5-degree bins for a single image."""
    angle_bins = np.arange(0, 360, 5)
    
    plt.figure(figsize=(12, 6))
    plt.bar(angle_bins, ellipse_histogram, width=5, align='edge', color='skyblue', edgecolor='black')
    plt.xlabel("Ellipse angle (°)")
    plt.ylabel("Number of relevant ellipses")
    plt.title(f"Histogram of ellipses by angle interval – {element_name} ({base_name})")
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    filename = f"angle_histogram_{base_name}.png"
    plt.savefig(os.path.join(save_folder, filename))
    plt.close()

def export_highlighted_image(image_path, ellipses_data, base_name, save_folder):
    """
    Draws highlighted circles on the raw image to visually verify detections.
    Converts physical micrometers back to local pixels for drawing.
    """
    if not ellipses_data:
        return

    # Load the raw image
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_height = img_color.shape[0]
    
    # We need the config to know the pixel resolution for unit conversion
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        pixel_res = config.get("pixel_resolution", 1.75)
    except FileNotFoundError:
        pixel_res = 1.75 # Fallback default

    for e in ellipses_data:
        # Re-invert Y locally for OpenCV drawing (which expects origin at top-left)
        cv_y = int(img_height - e["local_y"])
        cv_x = int(e["local_x"])
        center = (cv_x, cv_y)
        
        # Convert axes from µm back to pixels for drawing
        major_px = e["major_axis_um"] / pixel_res
        minor_px = e["minor_axis_um"] / pixel_res
        
        # Green circle for all geometrically valid ellipses
        radius_green = int(max(major_px, minor_px) * 3)
        cv2.circle(img_color, center, radius_green, (0, 200, 0), 2)  # Thickness reduced to 2 for raw images
        
        # Red circle on top if intensity confirms it's a deep crater
        if e["category"] == "red":
            radius_red = int((major_px + minor_px) * 5)
            cv2.circle(img_color, center, radius_red, (0, 0, 255), 2)

    filename = f"{base_name}_highlighted.png"
    cv2.imwrite(os.path.join(save_folder, filename), img_color)

def export_histogram(all_ellipses_data, element, save_folder):
    """Generates the area histogram for relevant ellipses in square micrometers."""
    if not all_ellipses_data:
        return

    # Extract the area directly in µm² (already calculated by the detection logic)
    ellipse_areas = [ellipse["area_um2"] for ellipse in all_ellipses_data]

    plt.figure(figsize=(12, 6))
    plt.hist(ellipse_areas, bins=30, color='mediumseagreen', edgecolor='black')
    plt.xlabel("Ellipse area (µm²)") # Updated unit
    plt.ylabel("Number of ellipses")
    plt.title(f"Histogram of areas of relevant ellipses - {element}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    filename = f"histo_areas_{element}.png"
    plt.savefig(os.path.join(save_folder, filename))
    plt.close()

def export_valid_ellipses_histogram(image_counts, element_name, save_folder, min_int, max_int, angle_tol):
    """Generates a bar chart showing the number of valid ellipses per raw image."""
    if not image_counts:
        return

    # Sort the dictionary keys to ensure the X-axis is logically ordered
    sorted_names = sorted(image_counts.keys())
    counts = [image_counts[name] for name in sorted_names]

    plt.figure(figsize=(14, 6))
    plt.bar(sorted_names, counts, color='skyblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Raw Image Name") # Updated label
    plt.ylabel("Number of valid ellipses (Red)")
    
    # Explicit title demonstrating the dual filtering
    title = (f"Histogram of valid ellipses per image ({element_name})\n"
             f"Filtered by Intensity ∈ [{min_int}, {max_int}] & Optimal Angle ±{angle_tol}°")
    plt.title(title)
    
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)

    filename = f"Histogram_ellipses_valid_{element_name}.png"
    plt.savefig(os.path.join(save_folder, filename))
    plt.close()

def export_global_heatmap(all_ellipses_data, element_name, save_folder):
    """Generates a 2D spatial density heatmap of all detected craters."""
    if not all_ellipses_data:
        return

    # Extract the new global coordinates in micrometers
    x_coords = [e["x_um"] for e in all_ellipses_data]
    y_coords = [e["y_um"] for e in all_ellipses_data]

    plt.figure(figsize=(10, 8))
    
    # Create the 2D histogram (heatmap)
    # bins=50 divides the space into a 50x50 grid
    # cmap='inferno' is a standard scientific color map for density visualization
    h = plt.hist2d(x_coords, y_coords, bins=50, cmap='inferno')
    plt.colorbar(h[3], label='Number of craters per sector')
    
    plt.xlabel("Global X Position (µm)")
    plt.ylabel("Global Y Position (µm)")
    plt.title(f"Global Density Heatmap of Craters - {element_name}")
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    filename = f"Heatmap_Density_{element_name}.png"
    plt.savefig(os.path.join(save_folder, filename))
    plt.close()

def export_csv(json_path, element, save_folder):
    """
    Reads the patched JSON and writes a flat CSV table: one row per ellipse.
    Columns: image, parent_mosaic, then all ellipse fields.
    Designed to be opened directly in Excel, LibreOffice or loaded with pandas.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    fieldnames = [
        'image', 'parent_mosaic',
        'local_x', 'local_y', 'x_um', 'y_um',
        'minor_axis_um', 'major_axis_um', 'area_um2',
        'angle', 'mean_intensity', 'circularity', 'category'
    ]

    filepath = os.path.join(save_folder, f"all_data_{element}.csv")
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for img_name, img_data in data['images'].items():
            parent_mosaic = img_data.get('parent_mosaic', '')
            for ellipse in img_data['ellipses']:
                writer.writerow({
                    'image': img_name,
                    'parent_mosaic': parent_mosaic,
                    **ellipse
                })

    print(f"CSV exporté : {filepath}")

def export_mosaic_index(json_path, element, save_folder):
    """
    Reads the patched JSON and writes a lightweight companion JSON:
        { "Mosaic_1_1.png": ["MoEDAL-034-035.png", ...], ... }
    Allows instant lookup of which images belong to a given mosaic.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    index = {}
    for img_name, img_data in data['images'].items():
        mosaic = img_data.get('parent_mosaic', 'no_mosaic')
        index.setdefault(mosaic, []).append(img_name)

    # Sort images within each mosaic, then sort mosaics alphabetically
    index = {k: sorted(v) for k, v in sorted(index.items())}

    filepath = os.path.join(save_folder, f"index_mosaics_{element}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=4)

    print(f"Index des mosaïques exporté : {filepath}")