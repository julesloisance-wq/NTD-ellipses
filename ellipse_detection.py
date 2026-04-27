import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os


def get_reference_center(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"❌ Image introuvable : {image_path}")

    print(f"Calibration sur la mosaïque : {image_path}")
    clicked_coords = []

    def on_click_center(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            clicked_coords.append((x, y))
            ax.plot(x, y, 'rx', markersize=10)
            fig.canvas.draw()
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, cmap='gray')
    cid = fig.canvas.mpl_connect('button_press_event', on_click_center)
    plt.title("Cliquez pour marquer le centre du trou")
    plt.show()

    if not clicked_coords:
        raise ValueError("❌ Aucun point sélectionné.")

    x0, y0 = clicked_coords[0]
    y0 = img.shape[0] - y0
    return x0, y0


def analyze_ellipses(image_path, config, ref_x0, ref_y0):
    min_intensity = config["min_intensity"]
    max_intensity = config["max_intensity"]
    green_threshold = 170

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Preprocessing identical to the original notebook
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 1. Angular histogram calculation
    num_bins = 360 // 5
    ellipse_histogram = np.zeros(num_bins, dtype=int)
    
    valid_contours_data = []
    
    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (major, minor), angle = ellipse
            
            if math.isnan(major) or math.isnan(minor):
                continue

            # Mean intensity calculation
            mask = np.zeros_like(img, dtype=np.uint8)
            cv2.ellipse(mask, ellipse, (255,), thickness=-1)
            masked_pixels = img[mask == 255]
            mean_intensity = np.mean(masked_pixels) if masked_pixels.size > 0 else 0
            
            valid_contours_data.append({
                "cx": cx, "cy": cy, "major": major, "minor": minor, 
                "angle": angle, "intensity": mean_intensity
            })

            # Intensity filtering for histogram construction
            if min_intensity <= mean_intensity <= max_intensity:
                bin_index = int(angle // 5) % num_bins
                ellipse_histogram[bin_index] += 1

    # Dominant angle extraction
    dominant_bin_index = int(np.argmax(ellipse_histogram))
    dominant_angle = dominant_bin_index * 5
    print(f"🧭 Dominant angle for {os.path.basename(image_path)}: {dominant_angle}°")

    # 2. Filtering and classification
    ellipses_data = []
    
    for e in valid_contours_data:
        # Exact angle condition from the notebook
        if dominant_angle <= e["angle"] <= dominant_angle + 5:
            if e["intensity"] < green_threshold:
                
                category = "green"
                if min_intensity <= e["intensity"] <= max_intensity:
                    category = "red"
                
                inverted_cy = img.shape[0] - e["cy"]
                x_relatif = e["cx"] - ref_x0
                y_relatif = inverted_cy - ref_y0
                
                ellipses_data.append({
                    "category": category,
                    "x": round(e["cx"], 2),
                    "y": round(inverted_cy, 2),
                    "intensity": round(e["intensity"], 2),
                    "major_axis": round(e["major"], 2),
                    "minor_axis": round(e["minor"], 2),
                    "angle": round(e["angle"], 2),
                    "x_relative": round(x_relatif, 2),
                    "y_relative": round(y_relatif, 2)
                })

    return ellipses_data, ellipse_histogram, dominant_angle