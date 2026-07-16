"""
generate_architecture_figure.py
--------------------------------
Clean, publication-quality "before / after" architecture diagram.
Output: report_figures/architecture_refactoring.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "report_figures"))
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
GREY_BG   = "#F5F6FA"
WHITE     = "#FFFFFF"

NB_ORANGE = "#E8A020"       # notebook accent
NB_BG     = "#FFF8E8"
NB_CELL   = "#FFFBF0"
NB_WARN   = "#D64444"

CFG_BLUE  = "#4A7FBF"
IMG_GREEN = "#4A9E6A"
DET_RED   = "#C8603A"
EXP_PURP  = "#7A56B8"
MAIN_BLUE = "#2B6CB0"
OUT_TEAL  = "#2A8080"

ARROW_COL = "#555577"
CONN_COL  = "#AAAACC"

# ── Figure ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 14, 8.5
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=160)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
ax.set_facecolor(GREY_BG)
fig.patch.set_facecolor(GREY_BG)

# ── Helpers ───────────────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, lw=1.2, alpha=1.0, z=2, r=0.15):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z)
    ax.add_patch(p)

def txt(x, y, s, fs=9, color="black", ha="center", va="center",
        bold=False, italic=False, z=5, alpha=1.0):
    fw = "bold" if bold else "normal"
    fs_style = "italic" if italic else "normal"
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va,
            fontweight=fw, fontstyle=fs_style, zorder=z, alpha=alpha)

def arrow_right(x0, y0, x1, y1, color=ARROW_COL, lw=2.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"), zorder=6)

def vline(x, y0, y1, color=CONN_COL, lw=1.2, style="-"):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, ls=style, zorder=4)

def hline(x0, x1, y, color=CONN_COL, lw=1.2):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=4)

def small_arrow(x0, y0, x1, y1, color=CONN_COL, lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"), zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
txt(FIG_W/2, 8.22, "Software Architecture Refactoring",
    fs=15, bold=True, color="#1A1A3A")

# ─────────────────────────────────────────────────────────────────────────────
# DIVIDER
# ─────────────────────────────────────────────────────────────────────────────
ax.axvline(x=6.6, color="#CCCCCC", lw=1.5, ls="--", zorder=1, ymin=0.04, ymax=0.94)

# ─────────────────────────────────────────────────────────────────────────────
# LEFT — BEFORE  (Notebook)
# ─────────────────────────────────────────────────────────────────────────────
txt(2.9, 7.82, "BEFORE", fs=13, bold=True, color=NB_ORANGE)
txt(2.9, 7.52, "Monolithic Jupyter Notebook", fs=9, color=NB_ORANGE, italic=True)

# Big notebook box
rbox(0.3, 0.6, 5.8, 6.7, NB_BG, NB_ORANGE, lw=2.2, r=0.25, z=1)

# Notebook tab at top
rbox(0.3, 6.95, 5.8, 0.35, NB_ORANGE, NB_ORANGE, lw=0, r=0.2, z=2)
txt(3.2, 7.13, "Code.ipynb", fs=10, bold=True, color=WHITE)

# Problem bullets — clean list style
PROBLEMS = [
    ("Hardcoded parameters",
     "Paths, thresholds and dimensions defined\ndirectly inside the calculation cells."),
    ("Memory saturation",
     "All images loaded into RAM simultaneously.\nRisk of crash on large datasets."),
    ("Sequential execution",
     "OpenCV processing limited to a single CPU core.\nOther cores remain idle."),
    ("Blocking interactivity",
     "Manual click required mid-execution to define\nthe reference coordinate — no automation."),
]

y_start = 6.45
dy = 1.42

for i, (title, desc) in enumerate(PROBLEMS):
    yb = y_start - i * dy
    # Card background
    rbox(0.55, yb - 0.88, 5.3, 1.0, NB_CELL, NB_ORANGE, lw=0.8, r=0.12, z=3)
    # Warning dot
    ax.plot(0.95, yb - 0.38, "o", ms=7, color=NB_WARN, zorder=5)
    # Title
    txt(1.12, yb - 0.38, title, fs=8.5, bold=True, color="#333333",
        ha="left", va="center")
    # Description
    txt(1.12, yb - 0.68, desc, fs=7.2, color="#666666",
        ha="left", va="center")

# ─────────────────────────────────────────────────────────────────────────────
# CENTER — Big arrow
# ─────────────────────────────────────────────────────────────────────────────
arrow_patch = FancyArrowPatch(
    posA=(6.25, 4.0), posB=(7.1, 4.0),
    arrowstyle="-|>", mutation_scale=24,
    color=MAIN_BLUE, linewidth=3.0, zorder=7)
ax.add_patch(arrow_patch)
txt(6.67, 4.38, "Refactoring", fs=7.5, color=MAIN_BLUE, italic=True, bold=True)

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT — AFTER  (Modular)
# ─────────────────────────────────────────────────────────────────────────────
txt(10.8, 7.82, "AFTER", fs=13, bold=True, color=MAIN_BLUE)
txt(10.8, 7.52, "Modular Python Architecture", fs=9, color=MAIN_BLUE, italic=True)

RX = 7.2   # right section origin x

# ── config.json (top center) ──────────────────────────────────────────────────
cfg_x, cfg_y, cfg_w, cfg_h = RX + 1.45, 6.15, 2.8, 0.95
rbox(cfg_x, cfg_y, cfg_w, cfg_h, "#EBF3FF", CFG_BLUE, lw=1.8, r=0.14, z=3)
rbox(cfg_x, cfg_y + cfg_h - 0.27, cfg_w, 0.27, CFG_BLUE, CFG_BLUE, lw=0, r=0.12, z=4)
txt(cfg_x + cfg_w/2, cfg_y + cfg_h - 0.135, "config.json", fs=8.5, bold=True, color=WHITE)
txt(cfg_x + cfg_w/2, cfg_y + 0.33,
    "Paths  |  Thresholds  |  Resolution  |  Tolerances",
    fs=7.2, color="#333355")

# ── Three modules (middle row) ────────────────────────────────────────────────
mods = [
    dict(label="image_processing.py", color=IMG_GREEN,
         x=RX+0.1, lines=["Batch mosaic assembly", "Constant RAM usage"]),
    dict(label="ellipse_detection.py", color=DET_RED,
         x=RX+2.35, lines=["Adaptive Canny filter", "Circularity + photometry"]),
    dict(label="data_export.py",      color=EXP_PURP,
         x=RX+4.6,  lines=["JSON  |  CSV", "Histograms  |  Heatmap"]),
]
MOD_W, MOD_H = 2.0, 1.25
mod_y = 4.35

for m in mods:
    mx = m["x"]
    rbox(mx, mod_y, MOD_W, MOD_H, WHITE, m["color"], lw=1.8, r=0.14, z=3)
    # colour header strip
    rbox(mx, mod_y + MOD_H - 0.27, MOD_W, 0.27, m["color"], m["color"],
         lw=0, r=0.12, z=4)
    txt(mx + MOD_W/2, mod_y + MOD_H - 0.135,
        m["label"], fs=7.0, bold=True, color=WHITE)
    for li, line in enumerate(m["lines"]):
        txt(mx + MOD_W/2, mod_y + 0.68 - li * 0.32,
            line, fs=7.2, color="#444444")

# ── main.py (bottom center) ───────────────────────────────────────────────────
main_x, main_y, main_w, main_h = RX + 0.65, 2.6, 5.4, 1.2
rbox(main_x, main_y, main_w, main_h, "#E6F0FF", MAIN_BLUE, lw=2.2, r=0.14, z=3)
rbox(main_x, main_y + main_h - 0.28, main_w, 0.28, MAIN_BLUE, MAIN_BLUE,
     lw=0, r=0.12, z=4)
txt(main_x + main_w/2, main_y + main_h - 0.14,
    "main.py  —  entry point & orchestrator", fs=8.5, bold=True, color=WHITE)
txt(main_x + main_w/2, main_y + 0.65,
    "Reads config  |  Calls all modules  |  Parallel processing (all CPU cores)",
    fs=7.5, color="#334466")
txt(main_x + main_w/2, main_y + 0.28,
    "concurrent.futures.ProcessPoolExecutor", fs=7.0, color="#7090C0", italic=True)

# ── Outputs (bottom row) ──────────────────────────────────────────────────────
out_items = [
    ("JSON / CSV",   "#D8EEE8", OUT_TEAL),
    ("Histograms",   "#EFE8FF", EXP_PURP),
    ("Heatmap 2D",   "#E8F3FF", CFG_BLUE),
    ("Images",       "#FFF0E8", DET_RED),
]
OUT_W, OUT_H = 1.2, 0.6
out_y = 1.3
total_out_w = len(out_items) * OUT_W + (len(out_items) - 1) * 0.15
out_x0 = RX + (6.6 - total_out_w) / 2

for i, (label, bg, ec) in enumerate(out_items):
    ox = out_x0 + i * (OUT_W + 0.15)
    rbox(ox, out_y, OUT_W, OUT_H, bg, ec, lw=1.4, r=0.1, z=3)
    txt(ox + OUT_W/2, out_y + OUT_H/2, label, fs=7.5, bold=True, color="#222222")

txt(RX + 3.3, 1.0, "Automated outputs", fs=8, color="#888888", italic=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONNECTORS (right side)
# ─────────────────────────────────────────────────────────────────────────────
cfg_cx = cfg_x + cfg_w / 2
cfg_bot = cfg_y

# config → each module (fan down)
mod_centers = [m["x"] + MOD_W/2 for m in mods]
mod_top = mod_y + MOD_H
# Draw a horizontal bus line at cfg_y - 0.18
bus_y = cfg_bot - 0.18
hline(mod_centers[0], mod_centers[-1], bus_y, CONN_COL, lw=1.2)
small_arrow(cfg_cx, cfg_bot, cfg_cx, bus_y + 0.01, CONN_COL)
for mc in mod_centers:
    vline(mc, bus_y, mod_top, CONN_COL, lw=1.1)
    # tick at bottom
    ax.plot(mc, mod_top, "^", ms=5, color=CONN_COL, zorder=5)

# modules → main.py (fan down to main)
main_cx = main_x + main_w / 2
bus2_y = mod_y - 0.18
hline(mod_centers[0], mod_centers[-1], bus2_y, CONN_COL, lw=1.2)
for mc in mod_centers:
    vline(mc, bus2_y, mod_y, CONN_COL, lw=1.1)
vline(main_cx, bus2_y, main_y + main_h, MAIN_BLUE, lw=1.4)
ax.plot(main_cx, main_y + main_h, "^", ms=5, color=MAIN_BLUE, zorder=5)

# main.py → outputs
main_bot = main_y
bus3_y = out_y + OUT_H + 0.15
vline(main_cx, main_bot, bus3_y, MAIN_BLUE, lw=1.4)
out_centers = [out_x0 + i * (OUT_W + 0.15) + OUT_W/2
               for i in range(len(out_items))]
hline(out_centers[0], out_centers[-1], bus3_y, CONN_COL, lw=1.2)
for oc in out_centers:
    small_arrow(oc, bus3_y, oc, out_y + OUT_H + 0.01, CONN_COL)

# ─────────────────────────────────────────────────────────────────────────────
# Bottom caption
# ─────────────────────────────────────────────────────────────────────────────
txt(FIG_W/2, 0.45,
    "Refactoring: from a monolithic Jupyter Notebook to a modular, parallel and reproducible Python architecture",
    fs=8, color="#888888", italic=True)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "architecture_refactoring.png")
fig.tight_layout(pad=0.3)
fig.savefig(out_path, bbox_inches="tight", dpi=160, facecolor=GREY_BG)
plt.close(fig)
print(f"Saved: {out_path}")
