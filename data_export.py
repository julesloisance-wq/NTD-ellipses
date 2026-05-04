import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import json

def export_json(data, filename, save_folder):
    """Exports a list of dictionaries to JSON format."""
    filepath = os.path.join(save_folder, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def export_angle_histogram_from_bins(ellipse_histogram, element_name, base_name, save_folder):
    """Generates the angle histogram using 5-degree bins."""
    angle_bins = np.arange(0, 360, 5)
    
    plt.figure(figsize=(12, 6))
    plt.bar(angle_bins, ellipse_histogram, width=5, align='edge', color='skyblue', edgecolor='black')
    plt.xlabel("Ellipse angle (°)")
    plt.ylabel("Number of relevant ellipses")
    plt.title(f"Histogram of ellipses by angle interval – {element_name}")
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    filename = f"angle_histogram_{base_name}.png"
    plt.savefig(os.path.join(save_folder, filename))
    plt.close()

def export_highlighted_mosaic(image_path, ellipses_data, base_name, save_folder):
    """Draws highlighted circles replicating the notebook's exact dimensions."""
    if not ellipses_data:
        return

    # OpenCV loads images in BGR format
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    for e in ellipses_data:
        original_y = img_color.shape[0] - e["y_local"]
        center = (int(e["x_local"]), int(original_y))
        
        # Notebook logic: Green circle for all valid angles
        radius_green = int(max(e["major_axis"], e["minor_axis"]) * 3)
        cv2.circle(img_color, center, radius_green, (0, 200, 0), 10)  # Green in BGR
        
        # Notebook logic: Red circle on top if intensity is also correct
        if e["category"] == "red":
            radius_red = int((e["major_axis"] + e["minor_axis"]) * 5)
            cv2.circle(img_color, center, radius_red, (0, 0, 255), 10)  # Red in BGR

    filename = f"{base_name}_highlighted.png"
    cv2.imwrite(os.path.join(save_folder, filename), img_color)

def export_histogram(all_ellipses_data, element, save_folder):
    """Generates the area histogram for relevant ellipses."""
    if not all_ellipses_data:
        return

    ellipse_areas = []
    for ellipse in all_ellipses_data:
        area = math.pi * (ellipse["major_axis"] / 2.0) * (ellipse["minor_axis"] / 2.0)
        ellipse_areas.append(area)

    plt.figure(figsize=(12, 6))
    plt.hist(ellipse_areas, bins=30, color='mediumseagreen', edgecolor='black')
    plt.xlabel("Ellipse area (pixels²)")
    plt.ylabel("Number of ellipses")
    plt.title(f"Histogram of areas of relevant ellipses - {element}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    filename = f"histo_areas_{element}.png"
    plt.savefig(os.path.join(save_folder, filename))
    plt.close()

def export_mosaics_histogram(mosaic_counts, element_name, save_folder, min_int, max_int, angle_tol):
    """Generates a bar chart showing the number of valid ellipses (angle + intensity) per mosaic."""
    if not mosaic_counts:
        return

    # Sort the dictionary keys to ensure the X-axis is logically ordered
    sorted_names = sorted(mosaic_counts.keys())
    counts = [mosaic_counts[name] for name in sorted_names]

    plt.figure(figsize=(14, 6))
    plt.bar(sorted_names, counts, color='skyblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Mosaic name")
    plt.ylabel("Number of valid ellipses (Red)")
    
    # Explicit title demonstrating the dual filtering
    title = (f"Histogram of valid ellipses per mosaic ({element_name})\n"
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

    # Extract the global coordinates of all valid craters
    # Using .get() to safely handle both "x_global_um" and the older "x_global_µm" keys
    x_coords = [e.get("x_global_um", e.get("x_global_µm", 0)) for e in all_ellipses_data]
    y_coords = [e.get("y_global_um", e.get("y_global_µm", 0)) for e in all_ellipses_data]

    plt.figure(figsize=(10, 8))
    
    # Create the 2D histogram (heatmap)
    # bins=50 divides the space into a 50x50 grid
    # cmap='inferno' is a standard scientific color map for density visualization
    h = plt.hist2d(x_coords, y_coords, bins=50, cmap='inferno')
    plt.colorbar(h[3], label='Number of craters per sector')
    
    plt.xlabel("Global X Position (µm)")
    plt.ylabel("Global Y Position (µm)")
    plt.title(f"Global Density Heatmap of Craters - {element_name}")
    
    # Invert Y-axis to match standard image coordinate systems (0,0 at top-left)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    
    filename = f"Heatmap_Density_{element_name}.png"
    plt.savefig(os.path.join(save_folder, filename))
    plt.close()