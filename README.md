# MoEDAL NTD Ellipse Detection & Profilometer Targeting

This project provides a full pipeline for processing MoEDAL NTD (Nuclear Track Detector) foil sheets scanned by an optical microscope scanner. It has **two distinct roles**:

1. **Detection** (`main.py`): automatically find and catalogue every microscopic crater left by heavy ions on a plastic foil sheet, and record their physical positions.
2. **Targeting** (`profilo_target.py`): from that catalogue, physically navigate the profilometer stage to any crater so its 3D depth profile can be measured.

---

## 🔬 What This Project Is About — Scientific Context

When a high-energy heavy ion crosses a MoEDAL plastic NTD foil, it leaves a tiny damage trail. After chemical etching, this trail becomes a microscopic conical crater, typically a few micrometres wide. These craters are the physical signatures of the particles we are looking for.

The optical scanner produces a large grid of raw images of the full foil surface. Each image is 3840 × 2748 pixels at ~1.68 µm/px, covering a ~6.4 × 4.6 mm field of view. Hundreds of these images tile the full foil. `main.py` processes all of them in parallel, detects the craters, and exports a complete database with the **physical position** of every crater on the foil.

But detecting a crater optically (2D image) is not enough. To measure its **depth profile** — which tells us the energy and identity of the ion that created it — we need to bring a 3D profilometer needle (stylus) to that exact point on the physical foil. The profilometer field of view is only ~1 × 1 mm, so finding a specific crater among thousands, without being able to see the full foil, requires precise coordinate conversion from scanner pixels to profilometer motor coordinates in µm.

This is what `profilo_target.py` does.

---

## 🎯 `profilo_target.py` — Targeting a Crater on the Profilometer

### Why this script is the most important one

This is the tool a physicist uses at the profilometer in the lab. After `main.py` has run and produced the crater database, `profilo_target.py` lets you:

1. Display any raw scanner image with the detected craters overlaid as cyan circles
2. Click on any point (a detected crater, or anything else like a scratch)
3. Get the **exact motor coordinates in µm** to enter into the profilometer controller

This bridges the gap between the optical scanner coordinate system and the profilometer motor coordinate system, accounting for:
- The overlap between adjacent scanner images
- The 3×3 mosaic block structure of the acquisition
- The physical tilt of the foil on the profilometer stage (δθ)
- The axis flip between image convention (y downward) and physical convention (y upward)

### Before running — one-time setup per session

Open `profilo_target.py` and update the two constants at the very top of the file:

```python
R1_MOTOR = {"x": 21805, "y": -21794}   # motor position of reference hole R1 (µm)
R2_MOTOR = {"x": -1637, "y":  38563}   # motor position of reference hole R2 (µm)
```

These values come from physically centering the profilometer objective on each reference hole and reading the motor display. They change only when you start a new session on a different foil sheet. Once set, you do not need to enter them again during the session.

> **How to get R1 and R2 motor coordinates:** navigate the profilometer stage to the centre of each reference hole (the drilled holes at the edges of the foil). The profilometer software shows the current motor position in µm. Write those numbers down and paste them above.

### How to run

Activate the virtual environment, then launch the script:

```bash
# macOS/Linux
source .venv/bin/activate
python3 profilo_target.py

# Windows
.venv\Scripts\activate
python profilo_target.py
```

### Step-by-step workflow

**Step 1 — Enter the target image number**

The script asks which raw image contains the crater you want to reach. You can use the short format:
```
Enter target image (e.g. '57 35' or '57-35' or 'MoEDAL-057-035.png'): 57 35
```
The script pads the numbers automatically → `MoEDAL-057-035.png`.

**Step 2 — Use the interactive window**

A matplotlib window opens with the raw image. All craters detected by `main.py` are shown as **cyan circles** on the image.

The tool has two modes, toggled by pressing **S**:

| Mode | Title bar | What it does |
|---|---|---|
| **FREE** (default) | `Mode: FREE (→ click position)` | Click anywhere — cross appears at click, coordinates computed from that exact pixel. Use this for scratches, surface marks, or any arbitrary point. |
| **SNAP** | `Mode: SNAP (→ nearest crater)` | Click snaps to the nearest detected crater in the database. Useful for precisely targeting a known crater. |

Left-click to select a point. The terminal immediately prints:
```
[FREE] click position
  Click pos (display)  : X = 1823 px,  Y = 934 px
  Global (stitched)    : X = 44951 px, Y = 45010 px
  *** Profilometer     : X = 15342.1 µm,  Y = 8901.7 µm ***
```

Right-click to close the window.

**Step 3 — Move the profilometer stage**

Enter the printed µm coordinates into the profilometer motor controller. The stylus will be positioned directly over the selected point.

### What the coordinate conversion does

The math behind the targeting, done automatically:

```
Click in raw image (local pixels, y from top)
    ↓  local_to_global_px()
        accounts for the 3×3 mosaic block structure
        and the calibrated scanner overlap (C_x = 667 px, C_y = 323.5 px)
Global pixel in the full virtual stitched canvas
    ↓  pixel_to_profilometer()
        1. Compute displacement from R1 in pixels
        2. Flip Y  (image y↓  →  physical y↑)
        3. Scale by pixel_resolution  (µm/px)
        4. Rotate by δθ  (foil tilt = θ_machine − θ_code)
        5. Translate by R1 motor position
Profilometer motor coordinates (µm)
```

δθ is the angular difference between how the foil was oriented in the scanner and how it sits on the profilometer stage. It is computed automatically from R1_MOTOR, R2_MOTOR, and the angle stored in the JSON database.

---

## 🗃️ `main.py` — Building the Crater Database

### What it does

`main.py` is the detection pipeline. It scans every raw image of the foil, finds every crater, and produces a complete spatial database. You run it once per foil sheet, before using `profilo_target.py`.

Here is exactly what happens when you run it:

1. **Reads `config.json`** — loads the path to the raw images, the element name, detection thresholds, overlap calibration values, and pixel resolution.

2. **Scans for reference holes** — runs `detect_reference_holes()` on all images to find the two drilled reference holes (R1 and R2) that serve as coordinate anchors. Results are cached in `reference_holes.json` to avoid rescanning on subsequent runs. If the cache already exists, it is loaded directly.

3. **Asks you to select R1** — opens a matplotlib window showing all detected reference hole candidates. You click on the one that should be the coordinate origin (R1). The pipeline then automatically selects R2 as the physically farthest hole from R1.

4. **Detects craters in parallel** — dispatches all images to all available CPU cores simultaneously using Python's `concurrent.futures`. For each image, the pipeline:
   - Extracts the green channel (most sensitive to the etched craters)
   - Applies Gaussian blur to reduce sensor noise
   - Runs an adaptive Canny edge detector (thresholds based on the image's median intensity, so lighting variations across images don't affect detection)
   - Finds all closed contours
   - Filters by area, circularity (≥ 0.25, to reject scratches and elongated marks), and mean intensity (to reject bright surface defects)
   - Fits an ellipse to each valid contour
   - Computes the local pixel position and global physical position (µm) of each crater relative to R1

5. **Stores the angle θ_code** — computes the theoretical orientation of the R1→R2 axis in the scanner frame and saves it in the JSON. This is what `profilo_target.py` uses to calculate δθ.

6. **Stores R1's local pixel position** (`r1_local_px_x`, `r1_local_px_y`) in the JSON — needed by `profilo_target.py` to reconstruct R1's position in the global stitched canvas.

7. **Writes all output files** to the `save_folder` set in `config.json`.

### How to run

**First time / full run:**

On Windows, double-click `run_windows.bat`. On macOS/Linux:
```bash
bash run_macOS_linux.sh
```

These launcher scripts automatically create a Python virtual environment, install all dependencies, and run `main.py`. You do not need to install anything manually.

**With annotated mosaics** (shows crater IDs drawn directly on the mosaic images):
```bash
# after activating .venv:
python3 main.py --annotate-json
```

**What main.py produces:**

| File | Description |
|---|---|
| `all_data_{element}.json` | Full crater database: local/global coordinates, area, axes, intensity, parent mosaic, reference system metadata |
| `all_data_{element}.csv` | Same data, flat table format (opens in Excel, pandas) |
| `index_mosaics_{element}.json` | Index mapping each mosaic filename to its source images |
| `Mosaic_X_Y.png` | Visual 3×3 mosaics of the foil for visual inspection |
| `reference_holes.json` | Cached reference hole positions (speeds up reruns) |
| `Heatmap_Density_{element}.png` | Spatial density map of all detected craters |
| `histo_areas_{element}.png` | Area distribution histogram |

---

## 🚀 Getting Started (from scratch)

Follow these steps exactly if you are setting up this project for the first time on a new machine.

### Step 1 — Install the prerequisites

You need two things installed on your machine before anything else:

- **Python 3.8+** → Download from [python.org](https://www.python.org/downloads/)
  - On Windows: during installation, **check the box "Add Python to PATH"**
  - Verify it works: open a terminal and type `python --version` (Windows) or `python3 --version` (macOS/Linux)
- **Git** → Download from [git-scm.com](https://git-scm.com/downloads)
  - Verify it works: open a terminal and type `git --version`

> VS Code is recommended as your code editor. Install it from [code.visualstudio.com](https://code.visualstudio.com/).

---

### Step 2 — Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/julesloisance-wq/NTD-ellipses
cd NTD-ellipses
```

---

### Step 3 — Set up `config.json`

Open `config.json` and fill in the three paths for your dataset:

```json
{
  "folder_path": "/absolute/path/to/your/raw/images/root",
  "element":     "subfolder_name_of_the_sheet_to_process",
  "save_folder": "/absolute/path/where/results/will/be/saved",
  ...
}
```

- `folder_path` + `element`: the raw images are expected at `folder_path/element/MoEDAL-xxx-yyy.png`
- `save_folder`: all outputs will be written here (created automatically if it doesn't exist)

> Leave the other parameters at their default values for a first run. See the **Parameters Guide** section for details.

---

### Step 4 — Run the pipeline

**On Windows:** double-click `run_windows.bat`

**On macOS / Linux:**
```bash
bash run_macOS_linux.sh
```

The launcher will create `.venv/`, install dependencies, and run `main.py` automatically.

---

### Step 5 — Select the reference hole

`main.py` will display a matplotlib window with the detected reference hole candidates. Click on the one that should be the coordinate origin (R1). The pipeline then continues automatically.

Results will be written to the `save_folder` you set in `config.json`.

---

## ⚙️ Quickstart (existing users)

If you already have the project set up, just run the launcher:

```bash
# macOS/Linux
bash run_macOS_linux.sh

# Windows
run_windows.bat
```

To run the profilometer targeting tool after the pipeline, activate the virtual environment:
```bash
# macOS/Linux
source .venv/bin/activate
python3 profilo_target.py

# Windows
.venv\Scripts\activate
python profilo_target.py
```

---

## 📁 Repository Structure & File Map

Files are listed roughly in order of scientific importance.

### Primary Tools (Project Root)

* [profilo_target.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/profilo_target.py) — **Primary scientific tool.** Converts a click on a raw scanner image into profilometer motor coordinates (µm). Opens an interactive window with detected craters overlaid as cyan circles. FREE mode (default): click anywhere. SNAP mode (press S): snaps to nearest database crater. Requires `main.py` to have been run first.

* [main.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/main.py) — Detection pipeline entry point. Scans all raw images in parallel, detects craters, prompts for reference hole selection, and writes the full crater database (JSON/CSV) and visual outputs.

* [ellipse_detection.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/ellipse_detection.py) — Computer vision core. Implements `detect_reference_holes()` (Hough circles + angular spread validation) and `analyze_ellipses()` (Canny edge detection, geometric filtering, ellipse fitting, intensity gating). Called by `main.py`, not run directly.

* [data_export.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/data_export.py) — Generates all output files: JSON database, CSV table, spatial density heatmap, area histogram. Called by `main.py`.

* [build_mosaics.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/build_mosaics.py) — Assembles raw images into 3×3 visual tiles for human inspection. Can also annotate tiles with crater IDs from the JSON (`--annotate-json` flag). Run independently if you need to regenerate mosaics without rerunning the full detection.

* [update_parent_mosaics.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/update_parent_mosaics.py) — Patches the `parent_mosaic` field in the JSON to link each crater to its 3×3 mosaic tile filename. Fast (no image processing), run independently after detection.

* [image_processing.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/image_processing.py) — Low-level utility: reads raw image dimensions, parses grid indices from filenames, extracts acquisition boundaries. Used internally by `main.py` and `build_mosaics.py`.

* [config.json](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/config.json) — Single source of truth for all parameters. Edit before each new foil sheet.

### Calibration Tools (`scripts/calibration/`)

* [profilometer_recalibration.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/calibration/profilometer_recalibration.py) — Free-click variant of `profilo_target.py` for features **not in the crater database** (scratches, surface marks, any arbitrary point). Unlike `profilo_target.py`, it does not read the JSON — R1/R2 global pixel positions and motor positions are hardcoded at the top of the file before use.

* [calculate_hole_positions.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/calibration/calculate_hole_positions.py) — Reads `reference_holes.json` and prints the global stitched pixel coordinates of R1 and R2. Useful when you need to update the hardcoded values in `profilometer_recalibration.py`.

### Debug & Reports (`scripts/debug/`, `scripts/reports/`)

* [generate_overlap_checks.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/reports/generate_overlap_checks.py) — Generates 4 border-alignment figures to visually verify that `crop_width_X` / `crop_height_Y` are correctly calibrated.
* [test_detection_debug.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/debug/test_detection_debug.py) — Runs detection on a few sample images and saves annotated output (accepted/rejected contours) to `scripts/debug_output/`.
* [debug_clahe.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/debug/debug_clahe.py) — Compares detection with/without CLAHE local illumination correction.
* [debug_filters.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/debug/debug_filters.py) — Side-by-side analysis of geometric, circularity, and intensity filters.
* [stitch_foil.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/debug/stitch_foil.py) — Stitches all mosaics into a single very large PNG of the entire foil.

---

## 🔧 Parameters Guide (`config.json`)

* `folder_path` (string): Absolute path to the raw data root directory.
* `element` (string): Subfolder name of the foil sheet to process. Raw images are expected at `folder_path/element/MoEDAL-xxx-yyy.png`.
* `save_folder` (string): Path where pipeline results, logs, and plots will be written.
* `crop_width_X` (float): Horizontal overlap in pixels between adjacent columns of images. Calibrated by template matching on border features (currently 667 px).
* `crop_height_Y` (float): Vertical overlap in pixels between adjacent rows of images. Calibrated by duplicate crater detection (currently 323.5 px).
* `min_area` / `max_area` (int): Valid area range in pixels for detected contours. Filters out dust (too small) and large surface defects (too large).
* `min_intensity` / `max_intensity` (int): Grayscale intensity range for ellipse pixels. Rejects overly bright surface spots that are not craters.
* `pixel_resolution` (float): Physical size of one scanner pixel in µm/px. Used everywhere coordinates are converted to physical units. Calibrated by least-squares fit on reference hole distances (currently 1.6752 µm/px).
* `angle_tolerance` (float): Angular deviation threshold in degrees (used in orientation statistics).

---

## 📐 Calibration Protocol (For a new foil sheet)

### Step 1: Calibrate overlap

1. Enter rough estimates for `crop_width_X` and `crop_height_Y` in `config.json`.
2. Run: `python scripts/reports/generate_overlap_checks.py`
3. Open the output figures in `report_figures/`. Adjust the values until the border features align perfectly across adjacent images.

### Step 2: Calibrate pixel resolution

1. Run `python main.py` and select R1.
2. Run `python scripts/calibration/calculate_hole_positions.py`. Note the global pixel coordinates of R1, R2 (and R3 if available).
3. Physically measure the distances between those same holes on the profilometer stage (µm).
4. Compute the optimal resolution by least-squares fit:

   ```
   R = (d12·D12 + d23·D23 + d13·D13) / (d12² + d23² + d13²)
   ```

   where d_ij = pixel distance, D_ij = physical distance in µm.

5. Update `pixel_resolution` in `config.json`.