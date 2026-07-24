"""
Profilometer Targeting Tool
============================
Main scientific tool for targeting detected craters on the profilometer stage.

Prerequisites:
  - Run main.py at least once to generate the JSON database for the current foil sheet.
  - Have the raw scanner images available (folder_path / element in config.json).

Pipeline (pixel → profilometer µm):
  Local pixel (raw image)
      ↓  local_to_global_px()   [accounts for overlap via crop_width_X / crop_height_Y]
  Global pixel (full stitched canvas)
      ↓  pixel_to_profilometer()
        1. Centre on R1_GLOBAL_PX
        2. Flip Y  (image y↓ → physical y↑)
        3. Scale by pixel_resolution (µm/px)
        4. Rotate by θ  (= θ_machine − θ_code)
        5. Translate to R1 motor position
  Profilometer coordinates (µm)

Usage:
  python profilo_target.py
"""

import cv2
import re
import math
import json
import os
import matplotlib.pyplot as plt


# ===========================================================================
# PARAMETERS — MODIFY AS NEEDED FOR EACH SESSION
# ===========================================================================

# Motor coordinates of the reference holes in the profilometer frame (µm).
# Read these values directly from the profilometer stage display after
# centering the objective on each reference hole.
R1_MOTOR = {"x": 21805, "y": -21794}   # e.g. MoEDAL-042-040.png
R2_MOTOR = {"x": -1637,  "y":  38563}   # e.g. MoEDAL-057-045.png

# ===========================================================================

# ---------------------------------------------------------------------------
# COORDINATE TRANSFORMATION FUNCTIONS
# (exact copies from scripts/calibration/profilometer_recalibration.py)
# ---------------------------------------------------------------------------

def local_to_global_px(local_x, local_y, img_name, config):
    """
    Converts local pixel coordinates within a raw image into global pixel
    coordinates in the full stitched canvas (at scale 1.0).

    local_y must be in OpenCV convention (y from top).
    """
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

    # Columns assembled right-to-left (c=2 is leftmost)
    local_x_in_img = local_x - C_x
    local_x_in_mosaic = (2 - c) * w_crop + local_x_in_img

    # Rows assembled bottom-to-top (r=2 is the top row)
    local_y_in_mosaic = (2 - r) * h_single_crop + local_y

    row_idx = max_x - X_mosaic
    col_idx = max_y - Y_mosaic

    global_x = col_idx * w_mosaic + local_x_in_mosaic
    global_y = row_idx * h_mosaic + local_y_in_mosaic

    return global_x, global_y


def pixel_to_profilometer(global_x, global_y, r1_global_px, r1_prof, theta, pixel_resolution):
    """
    Transforms global pixel coordinates into profilometer coordinates (µm).

    Pipeline:
        1. Centre on R1 in pixel space
        2. Flip Y (pixel Y points down, profilometer Y points up)
        3. Scale by pixel_resolution (µm/px)
        4. Apply rotation theta
        5. Translate to R1 motor position
    """
    dx = global_x - r1_global_px["x"]
    dy = -(global_y - r1_global_px["y"])  # flip Y

    dx_um = dx * pixel_resolution
    dy_um = dy * pixel_resolution

    x_rot = dx_um * math.cos(theta) - dy_um * math.sin(theta)
    y_rot = dx_um * math.sin(theta) + dy_um * math.cos(theta)

    x_prof = x_rot + r1_prof["x"]
    y_prof = y_rot + r1_prof["y"]

    return x_prof, y_prof


# ---------------------------------------------------------------------------
# IMAGE NAME HELPER
# ---------------------------------------------------------------------------

def parse_image_name(user_input):
    """
    Accepts short formats like "57 35", "57-35", "57,35" or full name
    "MoEDAL-057-035.png" and always returns the canonical filename.
    """
    user_input = user_input.strip()

    # Already full name
    if re.match(r"MoEDAL-\d{3}-\d{3}\.png", user_input):
        return user_input

    # Two numbers separated by space, dash or comma
    m = re.match(r"(\d+)[\s\-,]+(\d+)", user_input)
    if m:
        i = int(m.group(1))
        j = int(m.group(2))
        return f"MoEDAL-{i:03d}-{j:03d}.png"

    return None


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # --- Load config and JSON database ---
    with open("config.json", "r") as f:
        config = json.load(f)

    element      = config["element"]
    save_folder  = config["save_folder"]
    image_dir    = os.path.join(config["folder_path"], element)
    json_path    = os.path.join(save_folder, element, f"all_data_{element}.json")

    if not os.path.exists(json_path):
        print(f"Error: data file not found at {json_path}")
        print("Make sure you have run main.py at least once to generate the JSON.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    ref_sys = data.get("reference_system", {})
    try:
        r1_image         = ref_sys["r1_image"]
        r1_local_px_x    = ref_sys["r1_local_px_x"]
        r1_local_px_y    = ref_sys["r1_local_px_y"]   # OpenCV convention (y from top) — pass directly
        theta_code       = ref_sys["theta_code_radians"]
        pixel_resolution = config.get("pixel_resolution", 1.6752)
    except KeyError as e:
        print(f"Error: missing field in reference_system: {e}")
        print("Re-run main.py to regenerate the JSON with the updated reference system.")
        return

    # --- Compute R1 global pixel position from stored local coords ---
    # r1_local_px_y is already in OpenCV convention (y from top): pass directly.
    r1_gx, r1_gy = local_to_global_px(r1_local_px_x, r1_local_px_y, r1_image, config)
    R1_GLOBAL_PX = {"x": r1_gx, "y": r1_gy}

    print("=== PROFILOMETER TARGETING SYSTEM ===")
    print(f"Sheet            : {element}")
    print(f"R1 image         : {r1_image}")
    print(f"R1 global px     : ({r1_gx:.0f}, {r1_gy:.0f})")
    print(f"θ_code           : {math.degrees(theta_code):.3f}°")
    print(f"Pixel resolution : {pixel_resolution} µm/px\n")

    # --- Motor coordinates from hardcoded PARAMETERS section ---
    R1_PROF = R1_MOTOR
    x_m2, y_m2 = R2_MOTOR["x"], R2_MOTOR["y"]

    # theta_machine: angle of R1→R2 vector in the physical (Cartesian, y-up) motor frame
    theta_machine = math.atan2(y_m2 - R1_PROF["y"], x_m2 - R1_PROF["x"])
    # delta_theta: rotation to apply when transforming pixels → profilometer coords
    delta_theta   = theta_machine - theta_code

    print(f"[INFO] R1 motor  : ({R1_PROF['x']}, {R1_PROF['y']}) µm")
    print(f"[INFO] R2 motor  : ({x_m2}, {y_m2}) µm")
    print(f"[INFO] θ_machine = {math.degrees(theta_machine):.3f}°  (R1→R2 in motor frame)")
    print(f"[INFO] θ_code    = {math.degrees(theta_code):.3f}°  (R1→R2 in pixel frame)")
    print(f"[INFO] δθ (foil tilt) = {math.degrees(delta_theta):.3f}°\n")

    # --- Target image ---
    raw_input_img   = input("Enter target image (e.g. '57 35' or '57-35' or 'MoEDAL-057-035.png'): ")
    target_img_name = parse_image_name(raw_input_img)
    if target_img_name is None:
        print("Could not parse image name. Use format '57 35' or 'MoEDAL-057-035.png'.")
        return

    images_dict = data.get("images", {})
    if target_img_name not in images_dict:
        print(f"Error: '{target_img_name}' has no detected craters in the JSON.")
        return

    ellipses_in_image = images_dict[target_img_name].get("ellipses", [])
    if not ellipses_in_image:
        print("No craters found in this image.")
        return

    # --- Load raw image ---
    img_path = os.path.join(image_dir, target_img_name)
    if not os.path.exists(img_path):
        print(f"Error: image not found at {img_path}")
        return

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_raw, w_raw = img_rgb.shape[:2]

    print(f"\nImage loaded: {target_img_name} ({w_raw}×{h_raw} px)")
    print(f"{len(ellipses_in_image)} crater(s) in the database for this image.")
    print("Left-click on or near a crater → profilometer coordinates printed.")
    print("Right-click → close window.\n")

    # --- Interactive matplotlib window ---
    snap_mode = [False]   # mutable so the closure can modify it — default: FREE mode

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(img_rgb)

    def _update_title():
        mode = "SNAP (→ nearest crater)" if snap_mode[0] else "FREE (→ click position)"
        ax.set_title(
            f"{target_img_name}  |  Mode: {mode}  —  Press S to toggle  |  Left-click. Right-click to finish.",
            fontsize=10
        )

    _update_title()
    ax.set_xlabel("X local (px)")
    ax.set_ylabel("Y local (px)")

    # Draw detected craters as cyan circles.
    # local_x is in image x convention (left→right).
    # local_y is stored in Cartesian convention (y-up from bottom) → convert to display (y-down).
    import math as _math
    for e in ellipses_in_image:
        lx_disp = e["local_x"]
        ly_disp = h_raw - e["local_y"]   # Cartesian y-up → image y-down for matplotlib
        radius  = _math.sqrt(max(e.get("area_px", 100), 1) / _math.pi)
        circle  = plt.Circle((lx_disp, ly_disp), radius, color="cyan", fill=False, linewidth=1.2)
        ax.add_patch(circle)

    # Lock axes: prevent out-of-frame coordinates from zooming out the view
    ax.set_xlim(0, w_raw)
    ax.set_ylim(h_raw, 0)   # y increases downward in image display space

    final_result = {}

    def on_key(event):
        if event.key in ('s', 'S'):
            snap_mode[0] = not snap_mode[0]
            _update_title()
            fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return

        if event.button == 1:   # left click
            click_x = event.xdata
            click_y = event.ydata   # display coords (y from top, OpenCV convention)

            if snap_mode[0] and ellipses_in_image:
                # ── SNAP MODE: find nearest crater in database ──────────────────
                best   = None
                best_d = float("inf")
                for e in ellipses_in_image:
                    lx_disp = e["local_x"]
                    ly_disp = h_raw - e["local_y"]
                    d = math.hypot(click_x - lx_disp, click_y - ly_disp)
                    if d < best_d:
                        best_d = d
                        best   = e

                # local_x: image convention → pass directly
                # local_y: Cartesian y-up → OpenCV y-from-top for local_to_global_px
                local_x_t = best["local_x"]
                local_y_t = h_raw - best["local_y"]
                label = f"[SNAP] nearest crater  area={best.get('area_um2','?'):.1f} µm²  dist={best_d:.0f} px"
                meta  = best
            else:
                # ── FREE MODE: use click position directly ───────────────────────
                local_x_t = int(click_x)
                local_y_t = int(click_y)   # already y-from-top in display coords
                label = "[FREE] click position"
                meta  = None

            gx, gy = local_to_global_px(local_x_t, local_y_t, target_img_name, config)
            xp, yp = pixel_to_profilometer(gx, gy, R1_GLOBAL_PX, R1_PROF, delta_theta, pixel_resolution)

            # Cross always at the click position
            ax.plot(click_x, click_y, "r+", markersize=16, markeredgewidth=2)
            ax.annotate(
                f"  ({xp:.0f}, {yp:.0f}) µm",
                (click_x, click_y), color="red", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
            )
            fig.canvas.draw_idle()

            print(label)
            print(f"  Click pos (display)  : X = {int(click_x)} px,  Y = {int(click_y)} px")
            if snap_mode[0] and meta:
                print(f"  Crater pos (display) : X = {local_x_t} px,  Y = {local_y_t} px")
            print(f"  Global (stitched)    : X = {gx:.0f} px, Y = {gy:.0f} px")
            print(f"  *** Profilometer     : X = {xp:.1f} µm,  Y = {yp:.1f} µm ***\n")

            final_result["x"] = xp
            final_result["y"] = yp
            final_result["meta"] = meta

        elif event.button == 3:   # right click → close
            plt.close()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()

    if final_result:
        print("=== FINAL TARGETING RESULT ===")
        print(f"  → MOVE PROFILOMETER TO:")
        print(f"     X = {final_result['x']:.1f} µm")
        print(f"     Y = {final_result['y']:.1f} µm")
    else:
        print("No crater selected.")


if __name__ == "__main__":
    main()