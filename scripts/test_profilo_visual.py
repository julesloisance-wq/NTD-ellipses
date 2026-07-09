#!/usr/bin/env python3
"""
test_profilo_visual.py
======================
Interactive visual demo that validates the profilo_target.py algorithm end-to-end.

WORKFLOW
--------
Phase 1 — Scan image
  • Loads a real scan image with detected craters (red circles + IDs).
  • Two arbitrary points chosen as virtual reference holes R1 and R2.
  • User picks a target crater ID in the terminal.

Phase 2 — Simulated profilometer canvas
  • The scan image is rotated clockwise by ROT_DEG and embedded in a large
    blank canvas (simulating the foil placed on the profilometer support plate).
  • The user hovers over R1 crosshair → reads motor y-UP coordinates shown
    live in the figure → types them in the terminal.  Same for R2.
    (This simulates reading the profilometer motor position display.)
  • Scroll wheel: zoom in/out.

Phase 3 — Result overlay
  • Predicted crater position: thin yellow cross ✕
  • Ground-truth position (known rotation): thin green cross ✕
  • Error printed and displayed on the figure.

COORDINATE CONVENTIONS (same as profilo_target.py, y-UP profilometer)
-----------------------------------------------------------------------
  Scan frame  : x → right, y → DOWN  (OpenCV image pixels)
  Motor frame : x → right, y → UP    (Cartesian)
  Canvas ↔ Motor : x_motor = canvas_px,  y_motor = CANVAS_SIZE - canvas_py
  theta_code  : atan2(-(R2_cv_y - R1_cv_y), R2_cv_x - R1_cv_x)   [y-UP]
  delta_theta : theta_machine_yUP - theta_code_yUP
  Prediction formula (profilo_target.py y-UP branch):
      x_pred = x_m1 + x_code * cos(δθ) + y_code * sin(δθ)
      y_pred = y_m1 + x_code * sin(δθ) - y_code * cos(δθ)

Run from the project root:
    .venv/bin/python scripts/test_profilo_visual.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS  ← only edit this block
# ══════════════════════════════════════════════════════════════════════════════
PREFERRED_IMAGE = "MoEDAL-037-047.png"  # image to load (must exist in target dir)
SCALE           = 0.40    # downscale factor (keeps canvas manageable)
ROT_DEG         = 23.0    # clockwise visual rotation applied to the scan image (°)
CANVAS_MARGIN   = 300     # extra padding around the rotated image (pixels)
CANVAS_BG       = 50      # background grey level (0–255)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD CONFIG AND JSON DATA
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path  = os.path.join(PROJECT_ROOT, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

PIXEL_RES_ORIG = config.get("pixel_resolution", 1.75)   # µm / original pixel
PIXEL_RES      = PIXEL_RES_ORIG / SCALE                 # µm / scaled pixel
element        = config["element"]
target_dir     = os.path.join(config["folder_path"], element)
json_path      = os.path.join(config["save_folder"], element, f"all_data_{element}.json")

if not os.path.exists(json_path):
    print(f"[ERROR] JSON not found: {json_path}")
    print("  → Run main.py first to generate the data.")
    sys.exit(1)

with open(json_path, "r") as f:
    data = json.load(f)

# ══════════════════════════════════════════════════════════════════════════════
# 2. PICK A SCAN IMAGE WITH DETECTED CRATERS
# ══════════════════════════════════════════════════════════════════════════════
chosen_name    = None
chosen_craters = []

if PREFERRED_IMAGE in data["images"]:
    ells = data["images"][PREFERRED_IMAGE].get("ellipses", [])
    if len(ells) >= 1:
        chosen_name    = PREFERRED_IMAGE
        chosen_craters = ells

if chosen_name is None:
    for name, img_data in data["images"].items():
        ells = img_data.get("ellipses", [])
        if len(ells) >= 1:
            chosen_name    = name
            chosen_craters = ells
            break

if chosen_name is None:
    print("[ERROR] No image with detected craters found.")
    sys.exit(1)

img_path = os.path.join(target_dir, chosen_name)
scan_bgr_full = cv2.imread(img_path)
if scan_bgr_full is None:
    print(f"[ERROR] Could not read: {img_path}")
    sys.exit(1)

green_channel = scan_bgr_full[:, :, 1]
blurred = cv2.GaussianBlur(green_channel, (5, 5), 0)
scan_bgr_full = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

scan_bgr = cv2.resize(scan_bgr_full, (0, 0), fx=SCALE, fy=SCALE,
                       interpolation=cv2.INTER_AREA)
scan_rgb = cv2.cvtColor(scan_bgr, cv2.COLOR_BGR2RGB)
H, W = scan_rgb.shape[:2]

print(f"\n{'═'*60}")
print(f"  test_profilo_visual.py")
print(f"{'═'*60}")
print(f"  Image  : {chosen_name}  ({scan_bgr_full.shape[1]}×{scan_bgr_full.shape[0]} → {W}×{H} px)")
print(f"  Craters: {len(chosen_craters)}")
print(f"  Scale  : {SCALE:.2f}   Pixel resolution: {PIXEL_RES:.2f} µm/scaled-px")
print(f"  Simulated rotation : {ROT_DEG}° CW")
print(f"{'═'*60}\n")

# ══════════════════════════════════════════════════════════════════════════════
# 3. CRATER PIXEL POSITIONS (scaled)
#    local_x, local_y = raw cv2.fitEllipse outputs → x right, y DOWN (no flip)
# ══════════════════════════════════════════════════════════════════════════════
crater_pixels = []
for e in chosen_craters:
    cv_x = int(round(e["local_x"] * SCALE))
    cv_y = int(round(e["local_y"] * SCALE))
    crater_pixels.append((cv_x, cv_y, e["id"], e.get("area_um2", 0.0)))

# ══════════════════════════════════════════════════════════════════════════════
# 4. VIRTUAL REFERENCE HOLES R1 AND R2 IN THE SCALED SCAN IMAGE
# ══════════════════════════════════════════════════════════════════════════════
R1_CV = (W // 4,      3 * H // 4)
R2_CV = (3 * W // 4,     H // 4)

dx_code       = R2_CV[0] - R1_CV[0]
dy_code_yDOWN = R2_CV[1] - R1_CV[1]
theta_code    = math.atan2(-dy_code_yDOWN, dx_code)   # y-UP convention

print(f"  R1 pixel (scaled) : {R1_CV}")
print(f"  R2 pixel (scaled) : {R2_CV}")
print(f"  theta_code (y-UP) : {math.degrees(theta_code):.3f}°\n")

# ══════════════════════════════════════════════════════════════════════════════
# 5. HELPER: draw a thin precise crosshair on a cv2 image
# ══════════════════════════════════════════════════════════════════════════════
def draw_cross(img, pt, color, size=18, thickness=1):
    """Draw a thin + crosshair at pt=(x,y) on img (BGR)."""
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════════════════
# 6. PHASE-1 FIGURE — annotated scan image
# ══════════════════════════════════════════════════════════════════════════════
scan_cv = scan_bgr.copy()

for (cv_x, cv_y, cid, area) in crater_pixels:
    r_px = max(12, int(math.sqrt(area / math.pi) / PIXEL_RES) * 2)
    cv2.circle(scan_cv, (cv_x, cv_y), r_px, (0, 0, 255), 3, cv2.LINE_AA)
    draw_cross(scan_cv, (cv_x, cv_y), (0, 0, 255), size=10, thickness=2)
    cv2.putText(scan_cv, str(cid), (cv_x + r_px + 5, cv_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

# R1 and R2: solid black circles (simulate alignment artefacts) + thin white outline
cv2.circle(scan_cv, R1_CV, 9,  (0, 0, 0),       -1, cv2.LINE_AA)   # filled black
cv2.circle(scan_cv, R1_CV, 9,  (220, 220, 220),   1, cv2.LINE_AA)   # white outline
cv2.putText(scan_cv, "R1", (R1_CV[0] + 12, R1_CV[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

cv2.circle(scan_cv, R2_CV, 9,  (0, 0, 0),       -1, cv2.LINE_AA)
cv2.circle(scan_cv, R2_CV, 9,  (220, 220, 220),   1, cv2.LINE_AA)
cv2.putText(scan_cv, "R2", (R2_CV[0] + 12, R2_CV[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

cv2.line(scan_cv, R1_CV, R2_CV, (180, 180, 180), 1, cv2.LINE_AA)

scan_annotated = cv2.cvtColor(scan_cv, cv2.COLOR_BGR2RGB)

print("  Available craters:")
for (cv_x, cv_y, cid, area) in crater_pixels:
    print(f"    ID {cid:4d}  |  scaled pixel ({cv_x:4d}, {cv_y:4d})  |  area = {area:6.0f} µm²")
print()

fig1, ax1 = plt.subplots(figsize=(13, 9))
ax1.imshow(scan_annotated)
ax1.set_title(
    f"PHASE 1 — {chosen_name}  (×{SCALE})\n"
    f"R1 = {R1_CV}   R2 = {R2_CV}   Rotation to simulate: {ROT_DEG}° CW\n"
    f"Note the crater ID in the terminal, then enter it below.",
    fontsize=10
)
ax1.axis("off")
ax1.legend(handles=[
    Line2D([0],[0], marker="o", color="w", markerfacecolor="black",
           markersize=9, markeredgecolor="silver", label="R1 / R2 (virtual artefacts)"),
    Line2D([0],[0], color=(0.3, 0.3, 1.0), linewidth=1, marker="o", markersize=6,
           markerfacecolor="none", label="Detected craters"),
], loc="lower right", fontsize=9)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.5)

raw = input("  Enter target crater ID: ").strip()
try:
    TARGET_ID = int(raw)
except ValueError:
    TARGET_ID = crater_pixels[0][2]
    print(f"  Invalid → using first crater (ID {TARGET_ID})")

target = next(((x, y, cid, a) for (x, y, cid, a) in crater_pixels if cid == TARGET_ID), None)
if target is None:
    print(f"  ID {TARGET_ID} not found → using first crater.")
    target = crater_pixels[0]

tgt_cv_x, tgt_cv_y, tgt_id, tgt_area = target
x_code_px = tgt_cv_x - R1_CV[0]
y_code_px = tgt_cv_y - R1_CV[1]

plt.close(fig1)

print(f"\n  Target crater ID {tgt_id}:")
print(f"    scaled pixel   : ({tgt_cv_x}, {tgt_cv_y})")
print(f"    relative to R1 : ({x_code_px:+d}, {y_code_px:+d}) px  [x→right, y→DOWN]")
print(f"    theta_code     : {math.degrees(theta_code):.4f}° (y-UP, from scan image only)\n")

# ══════════════════════════════════════════════════════════════════════════════
# 7. BUILD THE CANVAS  (simulation — the algorithm never sees this)
# ══════════════════════════════════════════════════════════════════════════════
ROT_RAD = math.radians(ROT_DEG)
cos_r, sin_r = math.cos(ROT_RAD), math.sin(ROT_RAD)

corners_rel = [
    ( 0 - R1_CV[0],  0 - R1_CV[1]),
    ( W - R1_CV[0],  0 - R1_CV[1]),
    ( 0 - R1_CV[0],  H - R1_CV[1]),
    ( W - R1_CV[0],  H - R1_CV[1]),
]
rot_xs = [cos_r*dx - sin_r*dy for (dx, dy) in corners_rel]
rot_ys = [sin_r*dx + cos_r*dy for (dx, dy) in corners_rel]
max_ext_x = max(abs(min(rot_xs)), abs(max(rot_xs)))
max_ext_y = max(abs(min(rot_ys)), abs(max(rot_ys)))
CANVAS_SIZE = int(max(2*max_ext_x, 2*max_ext_y)) + 2 * CANVAS_MARGIN
R1_CANVAS   = (CANVAS_SIZE // 2, CANVAS_SIZE // 2)

canvas_bgr = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), CANVAS_BG, dtype=np.uint8)
M = cv2.getRotationMatrix2D((float(R1_CV[0]), float(R1_CV[1])), -ROT_DEG, 1.0)
M[0, 2] += R1_CANVAS[0] - R1_CV[0]
M[1, 2] += R1_CANVAS[1] - R1_CV[1]
cv2.warpAffine(scan_bgr, M, (CANVAS_SIZE, CANVAS_SIZE),
               dst=canvas_bgr, borderMode=cv2.BORDER_TRANSPARENT)

# ── Geometry helpers (used ONLY for simulation / display) ───────────────────

def scan_to_canvas(cv_x, cv_y):
    dx, dy = cv_x - R1_CV[0], cv_y - R1_CV[1]
    return (int(round(R1_CANVAS[0] + cos_r*dx - sin_r*dy)),
            int(round(R1_CANVAS[1] + sin_r*dx + cos_r*dy)))

def canvas_to_motor(cpx, cpy):
    """Canvas pixel → motor y-UP.  Shown to user for coordinate reading."""
    return cpx, CANVAS_SIZE - cpy

def motor_to_canvas(xm, ym):
    return int(round(xm)), int(round(CANVAS_SIZE - ym))

# Ground-truth canvas positions (for display verification only)
r1_canvas_gt  = scan_to_canvas(*R1_CV)
r2_canvas_gt  = scan_to_canvas(*R2_CV)
tgt_canvas_gt = scan_to_canvas(tgt_cv_x, tgt_cv_y)
tgt_motor_gt  = canvas_to_motor(*tgt_canvas_gt)

print(f"  Canvas: {CANVAS_SIZE}×{CANVAS_SIZE} px")
print(f"  R1 canvas pos : {r1_canvas_gt}   motor y-UP : {canvas_to_motor(*r1_canvas_gt)}")
print(f"  R2 canvas pos : {r2_canvas_gt}   motor y-UP : {canvas_to_motor(*r2_canvas_gt)}")
print(f"  [Crater GT]   : {tgt_canvas_gt}   motor y-UP : {tgt_motor_gt}  (revealed only at end)")

# ── Annotate canvas ─────────────────────────────────────────────────────────
canvas_ann = canvas_bgr.copy()

for (cv_x, cv_y, cid, area) in crater_pixels:
    c_px, c_py = scan_to_canvas(cv_x, cv_y)
    r_px = max(12, int(math.sqrt(area / math.pi) / PIXEL_RES) * 2)
    is_tgt = (cid == tgt_id)
    col = (0, 0, 255) if is_tgt else (100, 100, 150)
    thick = 3 if is_tgt else 2
    cv2.circle(canvas_ann, (c_px, c_py), r_px, col, thick, cv2.LINE_AA)
    draw_cross(canvas_ann, (c_px, c_py), col, size=10, thickness=thick)
    if is_tgt:
        cv2.putText(canvas_ann, f"TARGET {cid}", (c_px + r_px + 5, c_py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

# R1 and R2 on canvas: same solid black circles
cv2.circle(canvas_ann, r1_canvas_gt, 9, (0, 0, 0),       -1, cv2.LINE_AA)
cv2.circle(canvas_ann, r1_canvas_gt, 9, (220, 220, 220),   1, cv2.LINE_AA)
cv2.putText(canvas_ann, "R1", (r1_canvas_gt[0] + 12, r1_canvas_gt[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

cv2.circle(canvas_ann, r2_canvas_gt, 9, (0, 0, 0),       -1, cv2.LINE_AA)
cv2.circle(canvas_ann, r2_canvas_gt, 9, (220, 220, 220),   1, cv2.LINE_AA)
cv2.putText(canvas_ann, "R2", (r2_canvas_gt[0] + 12, r2_canvas_gt[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

canvas_rgb = cv2.cvtColor(canvas_ann, cv2.COLOR_BGR2RGB)

# ══════════════════════════════════════════════════════════════════════════════
# 8. PHASE-2 — INTERACTIVE CANVAS FIGURE
#    Hover over R1 → read motor y-UP coords in the figure → type in terminal.
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(12, 12))
img_disp = ax2.imshow(canvas_rgb)
ax2.axis("off")

# Large live coordinate readout at top of figure
coord_text = ax2.text(
    0.5, 0.985,
    "Hover over the image — motor y-UP coords appear here",
    transform=ax2.transAxes, fontsize=13, color="white",
    ha="center", va="top", fontfamily="monospace",
    bbox=dict(facecolor="#111111", alpha=0.8, boxstyle="round,pad=0.4")
)

status_text = ax2.text(
    0.01, 0.01,
    "Hover over R1 crosshair — note the coords — then type in terminal",
    transform=ax2.transAxes, fontsize=10, color="gold", va="bottom",
    bbox=dict(facecolor="black", alpha=0.65, boxstyle="round,pad=0.3")
)

plt.tight_layout()

# ── Scroll-wheel zoom ────────────────────────────────────────────────────────
ZOOM_FACTOR = 1.18

def on_scroll(event):
    if event.inaxes != ax2 or event.xdata is None:
        return
    f = 1.0 / ZOOM_FACTOR if event.button == "up" else ZOOM_FACTOR
    xl, xr = ax2.get_xlim()
    yb, yt = ax2.get_ylim()
    cx, cy = event.xdata, event.ydata
    ax2.set_xlim(cx + (xl - cx) * f, cx + (xr - cx) * f)
    ax2.set_ylim(cy + (yb - cy) * f, cy + (yt - cy) * f)
    fig2.canvas.draw_idle()

# ── Live coordinate display ──────────────────────────────────────────────────
def on_move(event):
    if event.inaxes != ax2 or event.xdata is None:
        return
    px, py = int(event.xdata), int(event.ydata)
    x_m, y_m = canvas_to_motor(px, py)
    coord_text.set_text(f"Motor y-UP :  x = {x_m}    y = {y_m}   "
                        f"  [canvas pixel: ({px}, {py})]")
    fig2.canvas.draw_idle()

fig2.canvas.mpl_connect("scroll_event",         on_scroll)
fig2.canvas.mpl_connect("motion_notify_event",  on_move)

plt.show(block=False)
plt.pause(0.5)

# ══════════════════════════════════════════════════════════════════════════════
# 9. TERMINAL INPUT — USER READS COORDS FROM FIGURE AND TYPES THEM
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print("  PHASE 2 — canvas is open")
print("  Hover over the crosshairs to read motor y-UP coordinates.")
print("  Scroll wheel = zoom in/out.  Use toolbar for pan.")
print(f"{'─'*60}")

def read_motor_coords(label):
    """Ask user to hover then type two integer coords."""
    while True:
        print(f"\n  Hover over {label} crosshair  → note the coords shown at the top of the figure.")
        raw = input(f"  Type {label} motor coords  →  x_m  y_m  (space-separated): ").strip()
        parts = raw.split()
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass
        print("  [!] Please enter two integers separated by a space, e.g.:  1682 1682")

x_m1, y_m1 = read_motor_coords("R1")

# Mark R1 on canvas
r1_input_px, r1_input_py = motor_to_canvas(x_m1, y_m1)
ax2.plot(r1_input_px, r1_input_py, "+", markersize=20, markeredgewidth=1.5,
         color="gold", zorder=10)
ax2.annotate(f"R1 entered\n({x_m1}, {y_m1})",
             xy=(r1_input_px, r1_input_py),
             xytext=(r1_input_px + 25, r1_input_py - 25),
             color="gold", fontsize=8,
             bbox=dict(facecolor="black", alpha=0.55, boxstyle="round"),
             arrowprops=dict(arrowstyle="-", color="gold", lw=0.8))
status_text.set_text("R1 recorded  —  now hover over R2")
status_text.set_color("deepskyblue")
plt.pause(0.3)

x_m2, y_m2 = read_motor_coords("R2")

r2_input_px, r2_input_py = motor_to_canvas(x_m2, y_m2)
ax2.plot(r2_input_px, r2_input_py, "+", markersize=20, markeredgewidth=1.5,
         color="deepskyblue", zorder=10)
ax2.annotate(f"R2 entered\n({x_m2}, {y_m2})",
             xy=(r2_input_px, r2_input_py),
             xytext=(r2_input_px + 25, r2_input_py - 25),
             color="deepskyblue", fontsize=8,
             bbox=dict(facecolor="black", alpha=0.55, boxstyle="round"),
             arrowprops=dict(arrowstyle="-", color="deepskyblue", lw=0.8))
plt.pause(0.2)

# ══════════════════════════════════════════════════════════════════════════════
# 10. ALGORITHM  — exact copy of profilo_target.py logic
#     !! THIS BLOCK KNOWS NOTHING ABOUT ROT_DEG, ROT_RAD, cos_r, sin_r,
#        R1_CANVAS, tgt_canvas_gt, scan_to_canvas() or motor_to_canvas() !!
#     It only uses:  theta_code (from scan image), x_code_px / y_code_px
#                    (crater position in scan image), and the motor coords
#                    of R1 and R2 typed by the user.
# ══════════════════════════════════════════════════════════════════════════════
theta_machine = math.atan2(y_m2 - y_m1, x_m2 - x_m1)   # y-UP
delta_theta   = theta_machine - theta_code               # both y-UP

x_pred_motor = x_m1 + x_code_px * math.cos(delta_theta) + y_code_px * math.sin(delta_theta)
y_pred_motor = y_m1 + x_code_px * math.sin(delta_theta) - y_code_px * math.cos(delta_theta)

# ── Back to canvas pixels for display (display helper only, not part of algo) ──
pred_px, pred_py = motor_to_canvas(x_pred_motor, y_pred_motor)
gt_px,   gt_py   = tgt_canvas_gt

err_px = math.sqrt((pred_px - gt_px)**2 + (pred_py - gt_py)**2)
err_um = err_px * PIXEL_RES

# ══════════════════════════════════════════════════════════════════════════════
# 11. PRINT REPORT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  {'═'*58}")
print(f"  ALGORITHM RESULT")
print(f"  {'─'*58}")
print(f"    theta_code    = {math.degrees(theta_code):.4f}  deg  (scan image only)")
print(f"    theta_machine = {math.degrees(theta_machine):.4f}  deg  (from user motor input)")
print(f"    delta_theta   = {math.degrees(delta_theta):.4f}  deg")
print(f"    (sim. rotation  = {ROT_DEG} deg CW  =>  expected delta_theta = {-ROT_DEG:.4f} deg)")
print(f"  {'─'*58}")
print(f"    Predicted motor : ({x_pred_motor:.1f}, {y_pred_motor:.1f})")
print(f"    Ground truth    : ({tgt_motor_gt[0]}, {tgt_motor_gt[1]})")
print(f"    Predicted canvas: ({pred_px}, {pred_py})")
print(f"    Ground truth    : ({gt_px}, {gt_py})")
print(f"    Error           : {err_px:.2f} px  ~  {err_um:.1f} um")
print(f"  {'═'*58}\n")

# ══════════════════════════════════════════════════════════════════════════════
# 12. DISPLAY RESULT ON CANVAS
# ══════════════════════════════════════════════════════════════════════════════
# Ground truth — thin green cross
ax2.plot(gt_px, gt_py, "+", markersize=28, markeredgewidth=1.8,
         color="limegreen", zorder=12, label=f"Ground truth — crater {tgt_id}")

# Prediction — thin yellow cross
ax2.plot(pred_px, pred_py, "+", markersize=28, markeredgewidth=1.8,
         color="yellow", zorder=13, label="Prediction (algorithm)")

# Connecting line
ax2.plot([gt_px, pred_px], [gt_py, pred_py],
         "--", color="white", lw=0.9, alpha=0.7, zorder=11)

# Label the two crosses
ax2.annotate("GT", xy=(gt_px, gt_py), xytext=(gt_px + 15, gt_py - 15),
             color="limegreen", fontsize=8,
             bbox=dict(facecolor="black", alpha=0.5, boxstyle="round"))
ax2.annotate("PRED", xy=(pred_px, pred_py), xytext=(pred_px + 15, pred_py + 15),
             color="yellow", fontsize=8,
             bbox=dict(facecolor="black", alpha=0.5, boxstyle="round"))

# Result overlay text
ok   = err_px < 1.5
good = err_px < 15.0
verdict = "PERFECT" if ok else ("GOOD" if good else "CLICK MORE PRECISELY")
result_str = (
    f"delta_theta computed : {math.degrees(delta_theta):.2f} deg   "
    f"(expected : {-ROT_DEG:.2f} deg)\n"
    f"Error : {err_px:.1f} px  ~  {err_um:.0f} um    [{verdict}]"
)
status_text.set_text(result_str)
status_text.set_color("limegreen" if ok else ("gold" if good else "tomato"))

ax2.legend(loc="upper right", fontsize=10,
           facecolor="black", labelcolor="white", framealpha=0.75)

fig2.canvas.draw_idle()
plt.show(block=True)
