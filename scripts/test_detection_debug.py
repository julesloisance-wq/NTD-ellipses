"""
test_detection_debug.py
-----------------------
Runs ellipse detection on a sample of images from the CENTER of the scan grid
(avoiding border images that contain the plastic edge and reflections).

Color code on the output image:
  GREEN  = accepted (detected ellipse)
  RED    = rejected: area out of range
  ORANGE = rejected: circularity too low
  CYAN   = rejected: ring validation (interior not brighter than border)
  PURPLE = rejected: intensity out of range

Run from the project root:
    .venv/bin/python scripts/test_detection_debug.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import json
import glob
import re

# ── Load config ────────────────────────────────────────────────────────────────
with open("config.json", "r") as f:
    config = json.load(f)

TARGET_DIR   = os.path.join(config["folder_path"], config["element"])
OUTPUT_DIR   = "scripts/debug_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_AREA              = config.get("min_area",       20)
MAX_AREA              = config.get("max_area",      2000)
MIN_INTENSITY         = config.get("min_intensity",    0)
MAX_INTENSITY         = config.get("max_intensity",  200)
CIRCULARITY_THRESHOLD = 0.25

# ── Select images closest to the center of the grid ───────────────────────────
all_images = sorted(glob.glob(os.path.join(TARGET_DIR, "MoEDAL-*.png")))

coords = []
for f in all_images:
    m = re.search(r"MoEDAL-(\d{3})-(\d{3})\.png", os.path.basename(f))
    if m:
        coords.append((int(m.group(1)), int(m.group(2)), f))

js  = sorted(set(c[0] for c in coords))
is_ = sorted(set(c[1] for c in coords))
j_mid = js[len(js) // 2]
i_mid = is_[len(is_) // 2]

# Sort by Manhattan distance to center and pick the 10 closest
center_sorted = sorted(coords, key=lambda c: abs(c[0] - j_mid) + abs(c[1] - i_mid))
sample = [c[2] for c in center_sorted[:10]]

print(f"Grid: j={js[0]}–{js[-1]}, i={is_[0]}–{is_[-1]}")
print(f"Center: j={j_mid}, i={i_mid}")
print(f"Processing {len(sample)} center images → saving to {OUTPUT_DIR}/\n")

# ── Counters ──────────────────────────────────────────────────────────────────
stats = {
    "accepted":           0,
    "rejected_area":      0,
    "rejected_circ":      0,
    "rejected_ring":      0,
    "rejected_intensity": 0,
    "rejected_points":    0,
}

for img_path in sample:
    img_name  = os.path.basename(img_path)
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_color is None:
        print(f"  WARNING: could not load {img_name}")
        continue

    green_channel = img_color[:, :, 1]

    # ── Pipeline (same as ellipse_detection.py) ────────────────────────────
    blurred = cv2.GaussianBlur(green_channel, (5, 5), 0)
    med     = np.median(blurred)
    lower   = int(max(0, med * 0.66))
    upper   = int(max(0, med * 1.33))
    edges   = cv2.Canny(blurred, lower, upper)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis = img_color.copy()
    accepted_count = 0
    h_img, w_img = green_channel.shape

    for cnt in contours:
        # Gate 1: min points
        if len(cnt) < 5:
            stats["rejected_points"] += 1
            continue

        area = cv2.contourArea(cnt)

        # Gate 2: area
        if not (MIN_AREA <= area <= MAX_AREA):
            cv2.drawContours(vis, [cnt], -1, (0, 0, 200), 1)   # RED
            stats["rejected_area"] += 1
            continue

        # Gate 3: circularity
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity < CIRCULARITY_THRESHOLD:
            cv2.drawContours(vis, [cnt], -1, (0, 128, 255), 1)  # ORANGE
            stats["rejected_circ"] += 1
            continue

        # Gate 4: dark-ring / bright-center  ("noir dehors, blanc dedans")
        ellipse = cv2.fitEllipse(cnt)
        (ex, ey), (eMA, ema), _ = ellipse
        ex_i, ey_i = int(ex), int(ey)
        r_inner = max(1, int(min(eMA, ema) / 2 * 0.6))
        r_outer = max(2, int(min(eMA, ema) / 2 * 1.4))

        ys_g, xs_g = np.ogrid[:h_img, :w_img]
        dist2    = (xs_g - ex_i) ** 2 + (ys_g - ey_i) ** 2
        int_pix  = green_channel[dist2 < r_inner ** 2]
        ring_pix = green_channel[(dist2 >= r_inner ** 2) & (dist2 < r_outer ** 2)]
        mean_int_c  = float(np.mean(int_pix))  if int_pix.size  > 0 else 0.0
        mean_ring_c = float(np.mean(ring_pix)) if ring_pix.size > 0 else 0.0

        if mean_int_c < mean_ring_c - 5:
            cv2.drawContours(vis, [cnt], -1, (255, 255, 0), 1)  # CYAN
            stats["rejected_ring"] += 1
            continue

        # Gate 5: global intensity
        mask   = np.zeros_like(green_channel, dtype=np.uint8)
        cv2.ellipse(mask, ellipse, (255,), thickness=-1)
        pixels = green_channel[mask == 255]
        mean_i = float(np.mean(pixels)) if pixels.size > 0 else 0.0

        if not (MIN_INTENSITY <= mean_i <= MAX_INTENSITY):
            cv2.drawContours(vis, [cnt], -1, (200, 0, 200), 1)  # PURPLE
            cx_e, cy_e = int(ellipse[0][0]), int(ellipse[0][1])
            cv2.putText(vis, f"I={mean_i:.0f}", (cx_e, cy_e),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 200), 1)
            stats["rejected_intensity"] += 1
            continue

        # ── ACCEPTED ──────────────────────────────────────────────────────
        cv2.ellipse(vis, ellipse, (0, 255, 0), 2)
        cx_e, cy_e = int(ellipse[0][0]), int(ellipse[0][1])
        cv2.putText(vis,
                    f"I={mean_i:.0f} C={circularity:.2f} A={area:.0f}",
                    (cx_e + 4, cy_e),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 220, 0), 1)
        stats["accepted"] += 1
        accepted_count += 1

    # Legend
    legend = [
        ("GREEN  = accepted",       (0,   200,   0)),
        ("RED    = area filtered",   (0,     0, 200)),
        ("ORANGE = circ filtered",   (0,   128, 255)),
        ("CYAN   = ring filtered",   (255, 255,   0)),
        ("PURPLE = intensity filt",  (200,   0, 200)),
    ]
    for k, (label, col) in enumerate(legend):
        cv2.putText(vis, label, (10, 20 + k * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    out_path = os.path.join(OUTPUT_DIR, img_name.replace(".png", "_debug.png"))
    cv2.imwrite(out_path, vis)
    print(f"  {img_name}: {accepted_count} accepted → {out_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────")
for k, v in stats.items():
    print(f"  {k:25s}: {v}")
print("─────────────────────────────────────────────────────────────────")
