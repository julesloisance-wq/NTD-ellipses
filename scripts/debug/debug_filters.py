#!/usr/bin/env python3
"""
debug_filters.py
================
Shows crater detection results on a mosaic's raw images under 4 filter configs:
  A) All filters ON (baseline)
  B) Morphological filter OFF (interior ≥ ring)
  C) Intensity filter OFF
  D) Both morphological AND intensity filters OFF

Produces a 2×2 grid of annotated mosaics for visual comparison.

Run from the project root:
    .venv/bin/python scripts/debug_filters.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── CONFIG ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
with open(os.path.join(PROJECT_ROOT, "config.json")) as f:
    config = json.load(f)

TARGET_MOSAIC  = "Mosaic_6_4.png"
ELEMENT        = config["element"]
DATA_DIR       = os.path.join(config["folder_path"], ELEMENT)
SAVE_DIR       = os.path.join(config["save_folder"], ELEMENT)
INDEX_PATH     = os.path.join(SAVE_DIR, f"index_mosaics_{ELEMENT}.json")

PIXEL_RES   = config.get("pixel_resolution", 1.75)
MIN_AREA    = config.get("min_area", 20)
MAX_AREA    = config.get("max_area", 2000)
MIN_INT     = config.get("min_intensity", 0)
MAX_INT     = config.get("max_intensity", 200)

SCALE       = 0.30          # downscale each image for the composite view
COLS        = 3             # images per mosaic row

# ─── LOAD IMAGE LIST ────────────────────────────────────────────────────────
with open(INDEX_PATH) as f:
    index = json.load(f)

if TARGET_MOSAIC not in index:
    print(f"[ERROR] {TARGET_MOSAIC} not found in index.")
    sys.exit(1)

image_names = sorted(index[TARGET_MOSAIC])
print(f"Mosaic {TARGET_MOSAIC}: {len(image_names)} images")

# ─── DETECTION FUNCTION (parametrised filters) ──────────────────────────────
def detect(img_path, use_morpho=True, use_intensity=True):
    """Returns list of (cx, cy, minor, major) in original pixel coords."""
    img_color   = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_color is None:
        return []
    green       = img_color[:, :, 1]
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
        if perim == 0 or (4 * np.pi * area / perim**2) < 0.25:
            continue

        ellipse = cv2.fitEllipse(cnt)
        (ex, ey), (eMA, ema), eangle = ellipse

        # ── Morphological filter ────────────────────────────────────────────
        if use_morpho:
            ex_i, ey_i   = int(ex), int(ey)
            r_inner      = max(1, int(min(eMA, ema) / 2 * 0.6))
            r_outer      = max(2, int(min(eMA, ema) / 2 * 1.4))
            h, w         = green.shape
            ys_g, xs_g   = np.ogrid[:h, :w]
            d2           = (xs_g - ex_i)**2 + (ys_g - ey_i)**2
            int_mask     = d2 < r_inner**2
            ring_mask    = (d2 >= r_inner**2) & (d2 < r_outer**2)
            m_int        = float(np.mean(green[int_mask]))  if np.any(int_mask)  else 0.0
            m_ring       = float(np.mean(green[ring_mask])) if np.any(ring_mask) else 0.0
            if m_int < m_ring - 5:
                continue

        # ── Intensity filter ────────────────────────────────────────────────
        if use_intensity:
            mask = np.zeros(green.shape, dtype=np.uint8)
            cv2.ellipse(mask, ellipse, (255,), thickness=-1)
            px   = green[mask == 255]
            mean = float(np.mean(px)) if px.size > 0 else 0.0
            if not (MIN_INT <= mean <= MAX_INT):
                continue

        results.append((ex, ey, eMA, ema))
    return results

# ─── BUILD COMPOSITE PANEL ──────────────────────────────────────────────────
configs = [
    ("A — All filters ON",                       True,  True),
    ("B — Morpho OFF / Intensity ON",            False, True),
    ("C — Morpho ON  / Intensity OFF",           True,  False),
    ("D — Both filters OFF",                     False, False),
]
COLORS = [
    (0, 0, 220),     # A: blue
    (0, 180, 0),     # B: green
    (220, 120, 0),   # C: orange
    (180, 0, 180),   # D: magenta
]

panels = []   # one annotated mosaic per config

for (label, use_morpho, use_intensity), color in zip(configs, COLORS):
    rows_imgs = []
    for idx_img, name in enumerate(image_names):
        path       = os.path.join(DATA_DIR, name)
        img_color  = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_color is None:
            img_color = np.zeros((100, 100, 3), dtype=np.uint8)

        green = img_color[:, :, 1]
        blurred = cv2.GaussianBlur(green, (5, 5), 0)
        display_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

        # Draw detections — large, thick, bright red circles
        hits = detect(path, use_morpho=use_morpho, use_intensity=use_intensity)
        for (cx, cy, eMA, ema) in hits:
            # Radius in original pixels, enlarged ×3 for visibility, min 20 px
            r_orig = max(20, int(min(eMA, ema) / 2 * 3.0))
            # Thick red circle (draw on full-res image before downscale)
            cv2.circle(display_bgr, (int(cx), int(cy)), r_orig, (0, 0, 255), 4, cv2.LINE_AA)
            # Center cross
            cv2.line(display_bgr, (int(cx)-14, int(cy)), (int(cx)+14, int(cy)), (0,0,255), 3, cv2.LINE_AA)
            cv2.line(display_bgr, (int(cx), int(cy)-14), (int(cx), int(cy)+14), (0,0,255), 3, cv2.LINE_AA)
        # Downscale after annotation
        small = cv2.resize(display_bgr, (0, 0), fx=SCALE, fy=SCALE,
                            interpolation=cv2.INTER_AREA)
        rows_imgs.append(small)

    # Arrange as COLS-wide grid
    rows = []
    for r in range(0, len(rows_imgs), COLS):
        row_group = rows_imgs[r:r + COLS]
        # Pad to COLS if needed
        while len(row_group) < COLS:
            row_group.append(np.zeros_like(row_group[0]))
        rows.append(np.hstack(row_group))
    mosaic = np.vstack(rows)

    # Add label bar — tall enough to be readable
    bar_h  = 60
    bar    = np.zeros((bar_h, mosaic.shape[1], 3), dtype=np.uint8)
    n_det  = sum(len(detect(os.path.join(DATA_DIR, n),
                             use_morpho=use_morpho, use_intensity=use_intensity))
                  for n in image_names)
    cv2.putText(bar, f"{label}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bar, f"{n_det} craters detected",
                (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (180, 180, 180), 1, cv2.LINE_AA)
    labeled = np.vstack([bar, mosaic])
    panels.append(cv2.cvtColor(labeled, cv2.COLOR_BGR2RGB))

# ─── DISPLAY 2×2 GRID ───────────────────────────────────────────────────────
OUT_PATH = os.path.join(PROJECT_ROOT, "report_figures", "debug_filters_output.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
fig.suptitle(f"Filter ablation study — {TARGET_MOSAIC}",
             fontsize=13, fontweight="bold", y=0.995)
fig.subplots_adjust(top=0.975, hspace=0.04, wspace=0.04,
                    left=0.02, right=0.98, bottom=0.02)

for ax, panel in zip(axes.flat, panels):
    ax.imshow(panel)
    ax.axis("off")

fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Figure saved → {OUT_PATH}")
print("Rendering figure…")
plt.show()
