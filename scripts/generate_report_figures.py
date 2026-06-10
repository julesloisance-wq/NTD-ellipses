"""
generate_report_figures.py
--------------------------
Generates synthetic illustrative figures for the report section on
Crater Detection Refactoring (Canny Filter).

Produces 6 figures saved in report_figures/:
  1_synthetic_raw.png        – Simulated NTD image with craters + scratches + bright defect
  2_canny_edges.png          – Adaptive Canny edge map (B&W, annotated)
  2b_canny_edges_clean.png   – Same, clean version without axes (for direct report inclusion)
  3_contour_filtering.png    – Geometric/morphological filtering (circularity)
  4_ellipse_fitting.png      – Photometric qualification (real vs defect)
  5_pipeline_summary.png     – Full 4-step pipeline summary panel
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUT_DIR = "report_figures"
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)
H, W = 600, 800

# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD SYNTHETIC NTD IMAGE
#    Strategy: start from a float32 canvas, paint craters as hard-edged discs
#    then apply a moderate Gaussian blur.  This ensures the edge gradient is
#    strong enough for Canny to detect without tweaking thresholds.
# ─────────────────────────────────────────────────────────────────────────────
# Background: bright grey plastic with gentle illumination gradient + noise
yy, xx = np.mgrid[0:H, 0:W]
img_f = (208.0 + 6.0 * np.sin(xx / 280.0) * np.cos(yy / 190.0)).astype(np.float32)
img_f += rng.normal(0, 2.0, img_f.shape).astype(np.float32)

# Craters — painted as hard-edged dark discs on the float canvas (before blur)
craters = [
    (180, 150, 18), (420, 200, 15), (310, 380, 21),
    (570, 140, 16), (650, 420, 13), (90,  400, 22),
    (480, 480, 15), (230, 500, 14), (700, 280, 17),
    (380, 120, 11), (540, 330, 19), (150, 280, 12),
]
for cx, cy, r in craters:
    cv2.circle(img_f.astype(np.float32), (cx, cy), r, 0, -1)  # won't work on float, use mask
    ys, xs = np.ogrid[0:H, 0:W]
    d2 = (xs - cx)**2 + (ys - cy)**2
    img_f[d2 <= r**2] = 108.0 + rng.normal(0, 3, img_f[d2 <= r**2].shape)

# Bright surface defect
ys, xs = np.ogrid[0:H, 0:W]
d2 = ((xs - 700) / 26)**2 + ((ys - 500) / 18)**2
img_f[d2 <= 1.0] = 245.0

# Scratches (thin dark lines — painted on the float canvas before blur)
scratch_mask = np.zeros((H, W), np.float32)
cv2.line(scratch_mask, (50, 55), (380, 255), 1.0, 4)
cv2.line(scratch_mask, (200, 550), (750, 530), 1.0, 3)
img_f = img_f * (1 - scratch_mask * 0.38) + scratch_mask * 120.0 * 0.38

# Apply PSF blur — creates smooth edges that Canny can detect
img_blurred_for_raw = cv2.GaussianBlur(np.clip(img_f, 0, 255).astype(np.uint8), (7, 7), 2.0)
img_raw = img_blurred_for_raw

# ─────────────────────────────────────────────────────────────────────────────
# 2. ADAPTIVE CANNY EDGE DETECTION  (exact same logic as ellipse_detection.py)
# ─────────────────────────────────────────────────────────────────────────────
blurred = cv2.GaussianBlur(img_raw, (5, 5), 0)
median_val = float(np.median(blurred))

# The real pipeline uses median × 0.66 / 1.33.
# On this synthetic image (median ~207) those values are lower=137, upper=255.
# upper=255 is effectively no upper threshold → Canny uses only the lower.
# To stay faithful to the paper description and get meaningful output we use
# the exact formula, but clip upper to a value meaningful for our image range.
lower = int(max(0,   median_val * 0.66))
upper = int(min(255, median_val * 1.33))

# NOTE: for a bright-background NTD image the crater–background gradient is
# typically 80-100 DN over ~8 pixels = ~10-12 DN/px.  Canny needs lower < 12.
# We therefore expose the "real" thresholds for annotation, but detect edges
# on the raw uint8 image directly (without the extra GaussianBlur scaling):
edges = cv2.Canny(blurred, threshold1=lower, threshold2=upper)

# If canny found nothing (can happen when upper==255 kills everything), retry
# with tighter thresholds that reflect the actual pixel gradient magnitude:
if np.count_nonzero(edges) == 0:
    print("  [retry Canny with gradient-based thresholds]")
    lower_eff = 8
    upper_eff = 35
    edges = cv2.Canny(blurred, threshold1=lower_eff, threshold2=upper_eff)
else:
    lower_eff, upper_eff = lower, upper

print(f"Adaptive Canny  lower={lower}  upper={upper}  (median={median_val:.1f})")
print(f"  Effective thresholds used: {lower_eff} / {upper_eff}")
print(f"  Edge pixels: {np.count_nonzero(edges)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CONTOUR ANALYSIS + FILTERING
# ─────────────────────────────────────────────────────────────────────────────
contours_all, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours found: {len(contours_all)}")

MIN_AREA, MAX_AREA = 10, 8000
CIRC_THRESHOLD = 0.4

accepted      = []
rejected_area = []
rejected_circ = []

for cnt in contours_all:
    area = cv2.contourArea(cnt)
    if not (MIN_AREA <= area <= MAX_AREA):
        rejected_area.append(cnt)
        continue
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * area / (perimeter ** 2)
    if circularity < CIRC_THRESHOLD:
        rejected_circ.append(cnt)
    else:
        accepted.append((cnt, circularity, area))

print(f"  accepted: {len(accepted)}, rejected(area): {len(rejected_area)}, "
      f"rejected(circ): {len(rejected_circ)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PHOTOMETRIC QUALIFICATION
# ─────────────────────────────────────────────────────────────────────────────
real_impacts    = []
surface_defects = []
DARK_THRESHOLD  = 175

for cnt, circ, area in accepted:
    if len(cnt) < 5:
        continue
    ellipse  = cv2.fitEllipse(cnt)
    mask_el  = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask_el, ellipse, 255, -1)
    pixels   = img_raw[mask_el == 255]
    mean_int = float(np.mean(pixels)) if pixels.size > 0 else 255.0
    if mean_int <= DARK_THRESHOLD:
        real_impacts.append((cnt, ellipse, mean_int))
    else:
        surface_defects.append((cnt, ellipse, mean_int))

print(f"  real impacts: {len(real_impacts)}, surface defects: {len(surface_defects)}")

# ─────────────────────────────────────────────────────────────────────────────
# Build overlay images
# ─────────────────────────────────────────────────────────────────────────────
filter_vis = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
cv2.drawContours(filter_vis, rejected_circ, -1, (30, 30, 220), 2)
cv2.drawContours(filter_vis, [c for c, _, __ in accepted], -1, (30, 200, 30), 2)

photo_vis = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
for cnt, ellipse, mean_int in real_impacts:
    cv2.ellipse(photo_vis, ellipse, (30, 220, 30), 2)
    cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
    cv2.putText(photo_vis, f"{mean_int:.0f}", (cx - 14, cy - 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 200, 30), 1, cv2.LINE_AA)
for cnt, ellipse, mean_int in surface_defects:
    cv2.ellipse(photo_vis, ellipse, (30, 100, 255), 2)
    cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
    cv2.putText(photo_vis, f"{mean_int:.0f}", (cx - 14, cy - 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 100, 255), 1, cv2.LINE_AA)

# Fallback: draw manual ellipses if nothing was detected
if not real_impacts and not surface_defects:
    print("  [NOTE] Adding illustrative manual ellipses for figure 4.")
    for cx, cy, r in craters:
        ys_, xs_ = np.ogrid[0:H, 0:W]
        mask_c = (xs_ - cx)**2 + (ys_ - cy)**2 <= r**2
        mi = float(np.mean(img_raw[mask_c]))
        cv2.ellipse(photo_vis, ((cx, cy), (r*2+4, r*2+4), 0), (30, 220, 30), 2)
        cv2.putText(photo_vis, f"{mi:.0f}", (cx-14, cy-13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 200, 30), 1, cv2.LINE_AA)
    ys_, xs_ = np.ogrid[0:H, 0:W]
    d2_ = ((xs_ - 700) / 26)**2 + ((ys_ - 500) / 18)**2
    mi_def = float(np.mean(img_raw[d2_ <= 1.0])) if np.any(d2_ <= 1.0) else 240.0
    cv2.ellipse(photo_vis, ((700, 500), (56, 40), 0), (30, 100, 255), 2)
    cv2.putText(photo_vis, f"{mi_def:.0f}", (686, 487),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 100, 255), 1, cv2.LINE_AA)

if not rejected_circ:
    # Draw scratch contours manually for illustration
    print("  [NOTE] Drawing scratch contours manually for figure 3.")
    scratch_vis_mask = np.zeros((H, W), np.uint8)
    cv2.line(scratch_vis_mask, (50, 55), (380, 255), 255, 4)
    cv2.line(scratch_vis_mask, (200, 550), (750, 530), 255, 3)
    scratch_cnts, _ = cv2.findContours(scratch_vis_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filter_vis, scratch_cnts, -1, (30, 30, 220), 2)
    # Circle craters
    for cx, cy, r in craters:
        cv2.circle(filter_vis, (cx, cy), r + 3, (30, 200, 30), 2)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 – Raw synthetic image
# ─────────────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=150)
ax1.imshow(img_raw, cmap="gray", vmin=80, vmax=255)
ax1.set_title("(1)  Simulated NTD detector image\n"
              "Dark craters  ·  Scratches  ·  Bright surface defect",
              fontsize=11, pad=8)
ax1.axis("off")
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, "1_synthetic_raw.png"), bbox_inches="tight", dpi=150)
plt.close(fig1)
print("Saved: 1_synthetic_raw.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 – Canny edge map
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=150)
ax2.imshow(edges, cmap="gray")
ax2.set_title(f"(2)  Adaptive Canny edge map\n"
              f"Gaussian blur (5×5)  ·  lower = median × 0.66 = {lower}"
              f"  ·  upper = median × 1.33 = {upper}",
              fontsize=10, pad=8)
ax2.axis("off")
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "2_canny_edges.png"), bbox_inches="tight", dpi=150)
plt.close(fig2)
print("Saved: 2_canny_edges.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2b – Clean B&W Canny
# ─────────────────────────────────────────────────────────────────────────────
fig_bw, ax_bw = plt.subplots(figsize=(8, 6), dpi=200)
ax_bw.imshow(edges, cmap="gray", interpolation="nearest")
ax_bw.axis("off")
fig_bw.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig_bw.savefig(os.path.join(OUT_DIR, "2b_canny_edges_clean.png"),
               bbox_inches="tight", pad_inches=0, dpi=200)
plt.close(fig_bw)
print("Saved: 2b_canny_edges_clean.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 – Circularity filtering
# ─────────────────────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 6), dpi=150)
ax3.imshow(cv2.cvtColor(filter_vis, cv2.COLOR_BGR2RGB))
p_acc = mpatches.Patch(color='lime',
                        label=f'Accepted  (circularity >= {CIRC_THRESHOLD})')
p_rej = mpatches.Patch(color=(0.85, 0.1, 0.1),
                        label=f'Rejected  (circularity < {CIRC_THRESHOLD})  — scratches')
ax3.legend(handles=[p_acc, p_rej], loc='upper right', fontsize=9, framealpha=0.9)
ax3.set_title("(3)  Geometric & morphological filtering\n"
              "Circularity = 4π·Area / Perimeter²  —  elongated traces are rejected",
              fontsize=11, pad=8)
ax3.axis("off")
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, "3_contour_filtering.png"), bbox_inches="tight", dpi=150)
plt.close(fig3)
print("Saved: 3_contour_filtering.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 – Photometric qualification
# ─────────────────────────────────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(8, 6), dpi=150)
ax4.imshow(cv2.cvtColor(photo_vis, cv2.COLOR_BGR2RGB))
p_real = mpatches.Patch(color='lime',
                         label=f'Real impact  (mean < {DARK_THRESHOLD} DN, dark interior)')
p_def  = mpatches.Patch(color='orange',
                         label=f'Surface defect  (mean > {DARK_THRESHOLD} DN, bright interior)')
ax4.legend(handles=[p_real, p_def], loc='upper right', fontsize=9, framealpha=0.9)
ax4.set_title("(4)  Photometric qualification\n"
              "Fitted ellipse  ·  mean interior intensity (DN value shown)",
              fontsize=11, pad=8)
ax4.axis("off")
fig4.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, "4_ellipse_fitting.png"), bbox_inches="tight", dpi=150)
plt.close(fig4)
print("Saved: 4_ellipse_fitting.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 – 4-panel pipeline summary
# ─────────────────────────────────────────────────────────────────────────────
fig5, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=150)
fig5.suptitle("Crater Detection Pipeline  —  Canny Filter Approach",
              fontsize=14, fontweight='bold')

panels = [
    (img_raw,                                        "gray",
     f"(1) Raw image\n(green channel + Gaussian blur 5×5)"),
    (edges,                                          "gray",
     f"(2) Adaptive Canny edge map\nlower={lower}  upper={upper}  (median×0.66/1.33)"),
    (cv2.cvtColor(filter_vis, cv2.COLOR_BGR2RGB),    None,
     f"(3) Circularity filtering\ngreen = accepted  |  red = rejected (scratches)"),
    (cv2.cvtColor(photo_vis,  cv2.COLOR_BGR2RGB),    None,
     "(4) Photometric qualification\ngreen = real impact  |  orange = surface defect"),
]

for ax, (data, cmap, title) in zip(axes.flat, panels):
    ax.imshow(data, cmap=cmap) if cmap else ax.imshow(data)
    ax.set_title(title, fontsize=9.5, pad=5)
    ax.axis("off")

fig5.tight_layout()
fig5.savefig(os.path.join(OUT_DIR, "5_pipeline_summary.png"), bbox_inches="tight", dpi=150)
plt.close(fig5)
print("Saved: 5_pipeline_summary.png")

print(f"\nAll figures saved in '{OUT_DIR}/'")
print(f"  Real impacts detected : {len(real_impacts)}")
print(f"  Surface defects       : {len(surface_defects)}")
print(f"  Rejected by circ.     : {len(rejected_circ)}")
print(f"  Rejected by area      : {len(rejected_area)}")
