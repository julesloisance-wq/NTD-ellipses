import cv2
import numpy as np
import os
import glob
import json
import re

def detect_reference_holes(target_dir, config):
    """
    Scans the target directory for reference holes using cv2.HoughCircles.
    Caches the results in reference_holes.json to avoid rescanning.
    Returns a dictionary of found holes: { 'filename': {'x': cx, 'y': cy, 'radius': r} }
    """
    cache_path = os.path.join(target_dir, "reference_holes.json")
    if os.path.exists(cache_path):
        print(f"Loading reference holes from cache: {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)

    print("Scanning images for reference holes (this may take a minute)...")
    reference_holes = {}
    
    # We look at all png files
    image_files = sorted(glob.glob(os.path.join(target_dir, "MoEDAL-*.png")))
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
            
        # 1. Threshold to find bright regions (plastic is ~215, background light is > 230)
        # We use 190 to separate the bright plastic/holes from the black jagged border (< 100)
        _, thresh = cv2.threshold(img_gray, 190, 255, cv2.THRESH_BINARY)
        
        # 2. Find all contours of these bright regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = img_gray.shape
        best_circle = None
        best_score = -1.0
        best_mean = 0.0

        #! use intensity filters to filter out bolts --> clear lighting in the center of ref holes, darker in bolts
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter out tiny noise and the massive contour of the whole plastic sheet
            if 5000 < area < (h * w * 0.5):
                # Extract points strictly inside the image to avoid fitting the straight image borders
                curve_points = []
                for p in cnt:
                    px, py = p[0]
                    if px > 5 and px < w - 5 and py > 5 and py < h - 5:
                        curve_points.append(p)
                
                # Fit an ellipse to the jagged curved boundary
                if len(curve_points) >= 5:
                    curve_points = np.array(curve_points)
                    (cx_f, cy_f), (MA, ma), angle = cv2.fitEllipse(curve_points)
                    
                    cx_f, cy_f = (cx_f, cy_f)
                    r_avg = (MA + ma) / 4
                    
                    # --- HOUGH REFINEMENT FOR EXTREME PRECISION ---
                    # fitEllipse shifts the center if the tearing is asymmetric (e.g., top missing).
                    # To find the true physical center of the intact drilled edge, we use Hough on the pure boundary.
                    mask = np.zeros_like(thresh)
                    cv2.drawContours(mask, [cnt], -1, 255, 1)
                    
                    circles = cv2.HoughCircles(
                        mask, 
                        cv2.HOUGH_GRADIENT, 
                        dp=1, 
                        minDist=100, 
                        param1=50, 
                        param2=15, # Low threshold since the boundary is 1-pixel clean
                        minRadius=int(max(50, r_avg * 0.7)), 
                        maxRadius=int(r_avg * 1.3)
                    )
                    
                    cx, cy, r = int(cx_f), int(cy_f), int(r_avg) + 20
                    
                    if circles is not None:
                        circles = np.round(circles[0, :]).astype("int")
                        # Pick the Hough circle closest to the robust fitEllipse center
                        best_dist = float('inf')
                        for (hx, hy, hr) in circles:
                            dist = (hx - cx_f)**2 + (hy - cy_f)**2
                            if dist < best_dist and dist < 10000: # Must be within ~100 pixels of approximate center
                                best_dist = dist
                                cx, cy = hx, hy
                                r = hr
                                
                    # We still add ~10 pixels to the Hough radius to visually align with the outer dark boundary
                    r = r + 10
                    
                    # Ensure the radius is within reasonable bounds for a reference hole (small or giant)
                    if 300 <= r <= 500:
                        # ── RIGOROUS VISUAL VALIDATION (ANGULAR SPREAD) ─────────────────────
                        
                        y_min, y_max = max(0, cy - r - 40), min(h, cy + r + 40)
                        x_min, x_max = max(0, cx - r - 40), min(w, cx + r + 40)
                        roi_gray = img_gray[y_min:y_max, x_min:x_max]
                        
                        if roi_gray.size == 0: continue
                        
                        local_cy = cy - y_min
                        local_cx = cx - x_min
                        ys, xs = np.ogrid[:roi_gray.shape[0], :roi_gray.shape[1]]
                        d2 = (xs - local_cx)**2 + (ys - local_cy)**2
                        
                        # Check interior brightness
                        interior_mask = d2 < (r - 20)**2
                        mean_interior = float(np.mean(roi_gray[interior_mask])) if np.any(interior_mask) else 0.0
                        if mean_interior <= 180: continue
                        
                        # Validate the angular spread of the dark torn plastic ring
                        annulus_mask = (d2 >= (r - 10)**2) & (d2 <= (r + 30)**2)
                        dark_pixels_mask = (roi_gray < 100) & annulus_mask
                        
                        dark_ys, dark_xs = np.where(dark_pixels_mask)
                        angular_spread = 0
                        
                        if len(dark_xs) > 0:
                            angles = np.degrees(np.arctan2(dark_ys - local_cy, dark_xs - local_cx))
                            angles = (angles + 360) % 360
                            unique_degrees = np.unique(angles.astype(int))
                            angular_spread = len(unique_degrees)
                            
                        # A true reference hole has a jagged black border covering at least 60 degrees
                        if angular_spread > 60:
                            # We keep the one with the thickest/most complete black border
                            if angular_spread > best_score:
                                best_score = angular_spread
                                best_circle = (cx, cy, r)
                                best_mean = mean_interior
        
        if best_circle is not None:
            cx, cy, r = best_circle
            reference_holes[img_name] = {"x": int(cx), "y": int(cy), "radius": int(r)}
            print(f"  -> Valid reference hole found in {img_name} at ({cx}, {cy}) r={r} [Interior={best_mean:.1f}, Spread={best_score}°]")

    # Save to cache
    with open(cache_path, 'w') as f:
        json.dump(reference_holes, f, indent=4)
        
    return reference_holes

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
    # Thresholds are calculated based on the median of the pixel intensities
    # to adapt to varying lighting conditions across images.
    lower = int(max(0, np.median(blurred) * 0.66))
    upper = int(max(0, np.median(blurred) * 1.33))
    edges = cv2.Canny(blurred, threshold1=lower, threshold2=upper)

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
        # Reject long traces/scratches. Lowered to 0.25 to accept slightly
        # asymmetric holes (teardrop shapes) while still rejecting scratches.
        if circularity < 0.25:
            continue

        # 7. Fit Ellipse
        ellipse = cv2.fitEllipse(cnt)
        (local_x, local_y), (minor_axis, major_axis), angle = ellipse
    
        # 8. Check internal intensity AND dark-ring / bright-center morphology
        # We expect: dark annular border (the etched track wall) + brighter interior
        (ex, ey), (eMA, ema), eangle = ellipse
        ex_i, ey_i = int(ex), int(ey)
        r_inner = max(1, int(min(eMA, ema) / 2 * 0.6))   # 60% of minor semi-axis
        r_outer = max(2, int(min(eMA, ema) / 2 * 1.4))   # 140% of minor semi-axis

        # Build interior and annular ring masks using simple distance from center
        h_img, w_img = green_channel.shape
        ys_grid, xs_grid = np.ogrid[:h_img, :w_img]
        dist2 = (xs_grid - ex_i) ** 2 + (ys_grid - ey_i) ** 2
        interior_mask = dist2 < r_inner ** 2
        ring_mask     = (dist2 >= r_inner ** 2) & (dist2 < r_outer ** 2)

        interior_pixels = green_channel[interior_mask]
        ring_pixels     = green_channel[ring_mask]

        mean_interior = float(np.mean(interior_pixels)) if interior_pixels.size > 0 else 0.0
        mean_ring     = float(np.mean(ring_pixels))     if ring_pixels.size     > 0 else 0.0

        # Full-ellipse mean for the original intensity gate
        mask = np.zeros_like(green_channel, dtype=np.uint8)
        cv2.ellipse(mask, ellipse, (255,), thickness=-1)
        masked_pixels = green_channel[mask == 255]
        mean_intensity = np.mean(masked_pixels) if masked_pixels.size > 0 else 0

        # MORPHOLOGICAL FILTER — disabled (too aggressive, rejects valid craters)
        # Rejects contours where the interior is darker than the surrounding ring.
        # Uncomment to re-enable the "noir dehors, blanc dedans" signature check.
        # if mean_interior < mean_ring - 5:
        #     continue


        # Categorize
        if config["min_intensity"] <= mean_intensity <= config["max_intensity"]:
            category = "red"
            # Add to histogram
            bin_index = int(angle // 5) % num_bins
            ellipse_histogram[bin_index] += 1
        else:
            continue # Intensity gate

        # 9. Spatial Geometry: Convert local pixels to Global Micrometers
        # We calculate how many pixels away this image is from the reference image
        # Assuming i = columns (X-axis) and j = rows (Y-axis) based on standard grid 
        # with bottom right as (j=0,i=0) and top left as (x_global=0,y_global=0)
        step_x = img_width - config.get("crop_width_X", 655)
        step_y = img_height - config.get("crop_height_Y", 295)

        global_x_pixels = (j_ref - j_current) * step_x + local_x - ref_x0
        global_y_pixels = (i_ref - i_current) * step_y + local_y - ref_y0

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