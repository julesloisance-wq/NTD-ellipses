import cv2
import numpy as np
import re
import math
import os

def get_reference_center(image_path):
    """
    Opens the reference image and allows the user to click on the reference hole.
    Returns the (x, y) coordinates of the click.
    """
    # Load image in color to display it nicely, or grayscale. We use color here.
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load reference image: {image_path}")

    ref_point = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ref_point.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Reference Image", img)

    cv2.imshow("Reference Image", img)
    cv2.setMouseCallback("Reference Image", click_event)
    
    print("Please click on the center of the reference hole, then press any key.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not ref_point:
        raise ValueError("No reference point was selected.")
        
    return ref_point[0]

def analyze_ellipses(image_path, config, ref_x0, ref_y0, i_ref, j_ref, img_width, img_height):
    """
    Detects and analyzes ellipses in a single raw image using the Green Channel,
    Canny Edge Detection, and Morphological filtering.
    """
    # 1. Extract grid coordinates (i, j) from the filename
    filename = os.path.basename(image_path)
    match = re.search(r"MoEDAL-(\d{3})-(\d{3})\.png", filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match expected pattern.")
    
    i_current = int(match.group(1))
    j_current = int(match.group(2))

    # 2. Load image and extract the Green Channel
    # OpenCV loads images in BGR format. Index 1 is Green.
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    green_channel = img_color[:, :, 1]

    # 3. Apply Gaussian Blur to reduce high-frequency noise
    blurred = cv2.GaussianBlur(green_channel, (5, 5), 0)

    # 4. Canny Edge Detection
    # Detects sharp changes in intensity (edges) instead of global thresholds
    # Thresholds can be adjusted in config later (e.g., 30, 100)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    # 5. Find contours on the detected edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses_data = []
    
    # Histogram setup (72 bins of 5 degrees)
    num_bins = int(360 / 5)
    ellipse_histogram = np.zeros(num_bins, dtype=int)
    
    pixel_resolution = config.get("pixel_resolution", 1.75) # Default 1.75 µm/px
    min_area = config.get("min_area", 20) # Minimum area in pixels to consider (to filter out noise)
    max_area = config.get("max_area", 500) # Maximum area in pixels to consider (to filter out large artifacts)

    for cnt in contours:
        # An ellipse needs at least 5 points to be mathematically fitted
        if len(cnt) < 5:
            continue

        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue

        # 6. Morphological Filter: Circularity
        # Perfect circle = 1.0. A long scratch will be close to 0.1
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        # Reject long traces/scratches (e.g., circularity < 0.4)
        if circularity < 0.4:
            continue

        # 7. Fit Ellipse
        ellipse = cv2.fitEllipse(cnt)
        (local_x, local_y), (minor_axis, major_axis), angle = ellipse
        local_y = img_height - local_y  # Invert Y 
    
        # 8. Check internal intensity to categorize (using the green channel)
        mask = np.zeros_like(green_channel, dtype=np.uint8)
        cv2.ellipse(mask, ellipse, (255,), thickness=-1)
        masked_pixels = green_channel[mask == 255]
        mean_intensity = np.mean(masked_pixels) if masked_pixels.size > 0 else 0

        # Categorize
        if config["min_intensity"] <= mean_intensity <= config["max_intensity"]:
            category = "red"
            # Add to histogram
            bin_index = int(angle // 5) % num_bins
            ellipse_histogram[bin_index] += 1
        else:
            continue # Too bright, ignore

        # 9. Spatial Geometry: Convert local pixels to Global Micrometers
        # We calculate how many pixels away this image is from the reference image
        # Assuming i = columns (X-axis) and j = rows (Y-axis) based on standard grid 
        # with bottom right as (j=0,i=0) and top left as (x_global=0,y_global=0)
        global_x_pixels = (j_ref - j_current) * img_width + local_x - ref_x0
        global_y_pixels = (i_ref - i_current) * img_height + local_y - ref_y0

        # Convert to micrometers
        global_x_um = global_x_pixels * pixel_resolution
        global_y_um = global_y_pixels * pixel_resolution

        ellipses_data.append({
            "local_x": float(local_x),
            "local_y": float(local_y),
            "x_um": float(global_x_um),
            "y_um": float(global_y_um),
            "minor_axis_um": float(minor_axis * pixel_resolution),
            "major_axis_um": float(major_axis * pixel_resolution),
            "area_um2": float(area * (pixel_resolution ** 2)),
            "angle": float(angle),
            "mean_intensity": float(mean_intensity),
            "circularity": float(circularity),
            "category": category,
            "image_source": filename
        })

    # Find the dominant angle for this specific raw image
    dominant_angle = float(np.argmax(ellipse_histogram) * 5) if np.sum(ellipse_histogram) > 0 else 0.0

    return ellipses_data, ellipse_histogram, dominant_angle