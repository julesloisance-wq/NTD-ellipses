"""
Profilometer Recalibration - Interactive Tool
==============================================
Process:
1. Display the target raw image (e.g. MoEDAL-057-035.png)
2. The user clicks on a feature of interest (e.g. a scratch) to get its local pixel coordinates
3. The script computes the global pixel coordinates (in the full stitched image)
4. Applies the rigid transformation (rotation + translation + scaling) to the profilometer coordinate system
5. Outputs the profilometer coordinates (in µm) of the selected feature

Usage: python scripts/profilometer_recalibration.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import math
import os

# ===========================================================================
# PARAMETERS: MODIFY AS NEEDED
# ===========================================================================

# "Real" coordinates of the reference holes in the profilometer frame (µm)
# These values come from the physical measurements provided by the user
R1_PROF = {"x": 21805, "y": -21794}   # MoEDAL-042-040.png
R2_PROF = {"x": -1637,  "y": 38563}   # MoEDAL-057-045.png

# Coordinates of the reference holes in the global stitched image (pixels)
R1_GLOBAL_PX = {"x": 44951, "y": 45010}   # MoEDAL-042-040.png
R2_GLOBAL_PX = {"x": 31438, "y": 8822}    # MoEDAL-057-045.png

# Target image to analyse
TARGET_IMAGE = "MoEDAL-057-035.png"
IMAGE_DIR = "/Users/julesloisance/Desktop/StageHelsinki/MoEDAL_Data_Apr2025/O1_L8_ME18_UD"

# Config file path (pixel resolution and overlap parameters)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

# ===========================================================================

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def local_to_global_px(local_x, local_y, img_name, config):
    """
    Converts local pixel coordinates within a raw image into global pixel
    coordinates in the full stitched canvas (at scale 1.0).
    """
    import re
    match = re.search(r"MoEDAL-(\d{3})-(\d{3})\.png", img_name)
    i, j = int(match.group(1)), int(match.group(2))

    C_x = config.get("crop_width_X", 667)
    C_y = config.get("crop_height_Y", 323.5)
    w_raw = 3840
    h_raw = 2748

    i_min, j_min = 34, 34
    h_single_crop = int(round(h_raw - C_y))
    w_crop = int(w_raw - C_x)
    w_mosaic = 3 * w_crop
    h_mosaic = 3 * h_single_crop

    num_row_blocks = 9   # (62-34+1)//3 = 9
    num_col_blocks = 7   # (55-34+1)//3 = 7
    max_x = num_row_blocks
    max_y = num_col_blocks

    block_i = (i - i_min) // 3
    block_j = (j - j_min) // 3
    r = (i - i_min) % 3
    c = (j - j_min) % 3

    X_mosaic = block_i + 1
    Y_mosaic = block_j + 1

    # Local coordinates within the mosaic tile
    # Columns are assembled right-to-left (c=2 is leftmost)
    local_x_in_img = local_x - C_x
    local_x_in_mosaic = (2 - c) * w_crop + local_x_in_img

    # Rows are assembled bottom-to-top (r=2 is the top row)
    local_y_in_mosaic = (2 - r) * h_single_crop + local_y

    # Global coordinates in the full stitched canvas
    row_idx = max_x - X_mosaic
    col_idx = max_y - Y_mosaic

    global_x = col_idx * w_mosaic + local_x_in_mosaic
    global_y = row_idx * h_mosaic + local_y_in_mosaic

    return global_x, global_y

def compute_transform(r1_global_px, r2_global_px, r1_prof, r2_prof, pixel_resolution):
    """
    Computes the rotation angle theta between the global pixel frame
    (stitched image) and the profilometer frame (µm).

    The rotation is derived purely from the direction of the R1->R2 vector
    in both coordinate systems (no scaling involved here).

    Returns:
        theta (float): rotation angle in radians
    """
    # R1->R2 vector in Cartesian pixel space (Y flipped: pixel Y points down)
    dx_pix = r2_global_px["x"] - r1_global_px["x"]
    dy_pix = -(r2_global_px["y"] - r1_global_px["y"])  # flip Y to match Cartesian convention
    angle_pix = math.atan2(dy_pix, dx_pix)

    # R1->R2 vector in the profilometer frame
    dx_prof = r2_prof["x"] - r1_prof["x"]
    dy_prof = r2_prof["y"] - r1_prof["y"]
    angle_prof = math.atan2(dy_prof, dx_prof)

    theta = angle_prof - angle_pix

    return theta

def pixel_to_profilometer(global_x, global_y, r1_global_px, r1_prof, theta, pixel_resolution):
    """
    Transforms global pixel coordinates into profilometer coordinates (µm).

    Pipeline:
        1. Center on R1 in pixel space
        2. Flip Y axis (pixel Y points down, profilometer Y points up)
        3. Scale by pixel_resolution (µm/px)
        4. Apply rotation theta
        5. Translate to R1's known profilometer position
    """
    # Step 1 & 2: Center on R1 and flip Y
    dx = global_x - r1_global_px["x"]
    dy = -(global_y - r1_global_px["y"])  # flip Y

    # Step 3: Convert to µm
    dx_um = dx * pixel_resolution
    dy_um = dy * pixel_resolution

    # Step 4: Apply rotation
    x_rot = dx_um * math.cos(theta) - dy_um * math.sin(theta)
    y_rot = dx_um * math.sin(theta) + dy_um * math.cos(theta)

    # Step 5: Translate to R1's profilometer position
    x_prof = x_rot + r1_prof["x"]
    y_prof = y_rot + r1_prof["y"]

    return x_prof, y_prof

def main():
    config = load_config()
    pixel_resolution = config.get("pixel_resolution", 1.6752328595)

    # Compute the rotation angle from the two reference holes
    theta = compute_transform(R1_GLOBAL_PX, R2_GLOBAL_PX, R1_PROF, R2_PROF, pixel_resolution)
    print(f"Rotation angle theta = {math.degrees(theta):.4f} deg")
    print(f"Pixel resolution used: {pixel_resolution} µm/px")

    # Sanity check: R2 should be recovered correctly from its pixel coordinates
    x_r2_check, y_r2_check = pixel_to_profilometer(
        R2_GLOBAL_PX["x"], R2_GLOBAL_PX["y"],
        R1_GLOBAL_PX, R1_PROF, theta, pixel_resolution
    )
    print(f"\nSanity check on R2:")
    print(f"  Computed : X = {x_r2_check:.1f} µm, Y = {y_r2_check:.1f} µm")
    print(f"  Expected : X = {R2_PROF['x']} µm, Y = {R2_PROF['y']} µm")
    print(f"  Residual : ΔX = {x_r2_check - R2_PROF['x']:.1f} µm, ΔY = {y_r2_check - R2_PROF['y']:.1f} µm")

    # Load target image
    img_path = os.path.join(IMAGE_DIR, TARGET_IMAGE)
    if not os.path.exists(img_path):
        print(f"\nError: image not found at {img_path}")
        return

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h_raw, w_raw = img.shape[:2]

    # List to store clicked points
    clicked_points = []

    print(f"\n=== INTERACTIVE MODE ===")
    print(f"Image: {TARGET_IMAGE} ({w_raw}x{h_raw} px)")
    print(f"Left-click to select a point of interest (e.g. the scratch).")
    print(f"Right-click to close and print the summary.\n")

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(img)
    ax.set_title(f"{TARGET_IMAGE} — Left-click to select a point. Right-click to finish.", fontsize=12)
    ax.set_xlabel("X local (px)")
    ax.set_ylabel("Y local (px)")

    def on_click(event):
        if event.inaxes != ax:
            return

        if event.button == 1:  # Left click -> add a point
            lx, ly = int(event.xdata), int(event.ydata)
            clicked_points.append((lx, ly))

            # Compute global and profilometer coordinates
            gx, gy = local_to_global_px(lx, ly, TARGET_IMAGE, config)
            xp, yp = pixel_to_profilometer(gx, gy, R1_GLOBAL_PX, R1_PROF, theta, pixel_resolution)

            ax.plot(lx, ly, 'r+', markersize=15, markeredgewidth=2)
            ax.annotate(
                f"  ({xp:.0f}, {yp:.0f}) µm",
                (lx, ly), color='red', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
            fig.canvas.draw_idle()

            print(f"Selected point:")
            print(f"  Local (raw image)  : X = {lx} px, Y = {ly} px")
            print(f"  Global (stitched)  : X = {gx:.0f} px, Y = {gy:.0f} px")
            print(f"  *** Profilometer   : X = {xp:.1f} µm, Y = {yp:.1f} µm ***")
            print()

        elif event.button == 3:  # Right click -> close
            plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()

    print(f"\n=== SUMMARY OF SELECTED POINTS ===")
    print(f"{'#':<4} {'Local X':<10} {'Local Y':<10} {'Prof X (µm)':<15} {'Prof Y (µm)':<15}")
    print("-" * 55)
    for k, (lx, ly) in enumerate(clicked_points):
        gx, gy = local_to_global_px(lx, ly, TARGET_IMAGE, config)
        xp, yp = pixel_to_profilometer(gx, gy, R1_GLOBAL_PX, R1_PROF, theta, pixel_resolution)
        print(f"{k+1:<4} {lx:<10} {ly:<10} {xp:<15.1f} {yp:<15.1f}")

if __name__ == "__main__":
    main()
