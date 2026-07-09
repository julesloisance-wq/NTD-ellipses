#!/usr/bin/env python3
"""
debug_clahe.py
==============
2-panel comparison on Mosaic_6_4:
  LEFT  — current pipeline (no CLAHE, no morpho filter)
  RIGHT — with CLAHE applied to the green channel before Canny

CLAHE (Contrast Limited Adaptive Histogram Equalization):
  Splits the image into small tiles, equalizes the histogram locally in each
  tile (to compensate for uneven illumination / vignetting), then clips the
  gain to avoid noise amplification.

Run from the project root:
    .venv/bin/python scripts/debug_clahe.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_ROOT, "config.json")) as f:
    config = json.load(f)

TARGET_MOSAIC = "Mosaic_6_4.png"
ELEMENT       = config["element"]
DATA_DIR      = os.path.join(config["folder_path"], ELEMENT)
SAVE_DIR      = os.path.join(config["save_folder"], ELEMENT)
INDEX_PATH    = os.path.join(SAVE_DIR, f"index_mosaics_{ELEMENT}.json")

MIN_AREA = config.get("min_area", 20)
MAX_AREA = config.get("max_area", 2000)
MIN_INT  = config.get("min_intensity", 0)
MAX_INT  = config.get("max_intensity", 200)

SCALE = 0.30   # downscale factor for display
COLS  = 3      # images per mosaic row

# CLAHE parameters
CLAHE_CLIP   = 2.0        # contrast clip limit (higher = more enhancement)
CLAHE_TILE   = (8, 8)     # tile grid size in pixels

# ─── LOAD IMAGE LIST ────────────────────────────────────────────────────────
with open(INDEX_PATH) as f:
    index = json.load(f)

image_names = sorted(index[TARGET_MOSAIC])
print(f"Mosaic {TARGET_MOSAIC}: {len(image_names)} images")

clahe_obj = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)

# ─── DETECTION (morpho filter DISABLED, CLAHE toggled) ──────────────────────
def detect(img_path, use_clahe=False):
    """Returns list of (cx, cy, eMA, ema) in original pixel coords."""
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_color is None:
        return []

    green = img_color[:, :, 1]

    if use_clahe:
        green = clahe_obj.apply(green)   # ← equalize before blurring

    blurred     = cv2.GaussianBlur(green, (5, 5), 0)
    med         = np.median(blurred)
    edges       = cv2.Canny(blurred, int(med * 0.66), int(med * 1.33))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0 or (4 * np.pi * area / perim ** 2) < 0.25:
            continue

        ellipse = cv2.fitEllipse(cnt)
        (ex, ey), (eMA, ema), _ = ellipse

        # Morphological filter — DISABLED (commented out)
        # if mean_interior < mean_ring - 5: continue

        # Intensity gate (on ORIGINAL green channel, not the equalized one)
        orig_green = img_color[:, :, 1]
        mask = np.zeros(orig_green.shape, dtype=np.uint8)
        cv2.ellipse(mask, ellipse, (255,), thickness=-1)
        px   = orig_green[mask == 255]
        mean = float(np.mean(px)) if px.size > 0 else 0.0
        if not (MIN_INT <= mean <= MAX_INT):
            continue

        results.append((ex, ey, eMA, ema))
    return results

# ─── BUILD ONE PANEL ─────────────────────────────────────────────────────────
def build_panel(use_clahe, label):
    rows_imgs = []
    total = 0
    for name in image_names:
        path      = os.path.join(DATA_DIR, name)
        img_color = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_color is None:
            img_color = np.zeros((100, 100, 3), dtype=np.uint8)

        green   = img_color[:, :, 1]
        equalized = clahe_obj.apply(green) if use_clahe else green
        blurred   = cv2.GaussianBlur(equalized, (5, 5), 0)

        # Use the blurred (processed) channel as background — what Canny sees
        display_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

        hits = detect(path, use_clahe=use_clahe)
        total += len(hits)

        for (cx, cy, eMA, ema) in hits:
            r = max(20, int(min(eMA, ema) / 2 * 3.0))
            cv2.circle(display_bgr, (int(cx), int(cy)), r, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.line(display_bgr, (int(cx)-14, int(cy)), (int(cx)+14, int(cy)), (0,0,255), 3, cv2.LINE_AA)
            cv2.line(display_bgr, (int(cx), int(cy)-14), (int(cx), int(cy)+14), (0,0,255), 3, cv2.LINE_AA)

        small = cv2.resize(display_bgr, (0, 0), fx=SCALE, fy=SCALE,
                            interpolation=cv2.INTER_AREA)
        rows_imgs.append(small)

    # Assemble 3-column grid
    rows = []
    for r in range(0, len(rows_imgs), COLS):
        row_group = rows_imgs[r:r + COLS]
        while len(row_group) < COLS:
            row_group.append(np.zeros_like(row_group[0]))
        rows.append(np.hstack(row_group))
    mosaic = np.vstack(rows)

    # Label bar
    bar = np.zeros((60, mosaic.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, label,
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bar, f"{total} craters detected",
                (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
    labeled = np.vstack([bar, mosaic])
    return cv2.cvtColor(labeled, cv2.COLOR_BGR2RGB), total


# ─── GENERATE PANELS ─────────────────────────────────────────────────────────
print("Processing WITHOUT CLAHE…")
panel_no_clahe, n_no  = build_panel(use_clahe=False, label="A — No CLAHE  (current pipeline)")
print(f"  → {n_no} craters")

print("Processing WITH CLAHE…")
panel_clahe,    n_yes = build_panel(use_clahe=True,  label=f"B — CLAHE ON  (clipLimit={CLAHE_CLIP}, tile={CLAHE_TILE})")
print(f"  → {n_yes} craters")

# ─── DISPLAY SIDE BY SIDE ────────────────────────────────────────────────────
OUT_PATH = os.path.join(PROJECT_ROOT, "report_figures", "debug_clahe_output.png")

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 9), dpi=100)
fig.suptitle(f"CLAHE comparison — {TARGET_MOSAIC}  (morpho filter OFF)",
             fontsize=13, fontweight="bold", y=0.998)
fig.subplots_adjust(top=0.975, hspace=0.02, wspace=0.03,
                    left=0.01, right=0.99, bottom=0.01)

ax_l.imshow(panel_no_clahe); ax_l.axis("off")
ax_r.imshow(panel_clahe);    ax_r.axis("off")

fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Figure saved → {OUT_PATH}")
plt.show()
