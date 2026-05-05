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
    
    # Note: plt.gca().invert_yaxis() was removed here because we already inverted 
    # the local Y axis in ellipse_detection.py to match a standard Cartesian plane.
    
    plt.tight_layout()
    
    filename = f"Heatmap_Density_{element_name}.png"
    plt.savefig(os.path.join(save_folder, filename))
    plt.close()