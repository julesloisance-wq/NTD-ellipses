# MoEDAL NTD Ellipse Detection & Alignment Pipeline

This software suite automates the detection, qualification, and spatial alignment of microscopic etched tracks (craters) on MoEDAL plastic NTD (Nuclear Track Detector) sheets. It processes raw, unstitched scanner images, stitches them into a virtual global coordinate system, and exports their physical locations.

---

## 🔬 Profilometer Targeting — The Core Scientific Tool

`profilo_target.py` is the most scientifically important script in this project. Once the detection pipeline has run and produced a crater database (JSON), this tool lets you click anywhere on a raw image and obtain the exact profilometer motor coordinates (in µm) needed to physically move the profilometer stage to that point. All detected craters are overlaid as cyan circles for reference.

### Prerequisites

- `main.py` must have been run at least once → it generates `all_data_<element>.json`
- Raw scanner images must be accessible (path set in `config.json`)
- The profilometer must be physically connected and calibrated

### How to use it — Step by step

**Step 1 — Activate the virtual environment and launch the script**

On Windows:
```bat
.venv\Scripts\activate
python profilo_target.py
```
On macOS/Linux:
```bash
source .venv/bin/activate
python3 profilo_target.py
```

**Step 2 — Enter motor coordinates for R1**

Navigate the profilometer stage to the centre of reference hole R1 (the hole you selected as origin when running `main.py`). Read the motor position displayed by the profilometer software and enter:
```
Enter motor X for R1 (µm): 21805
Enter motor Y for R1 (µm): -21794
```

**Step 3 — Enter motor coordinates for R2**

Navigate to reference hole R2 (the second reference hole). Enter its motor position:
```
Enter motor X for R2 (µm): -1637
Enter motor Y for R2 (µm): 38563
```
The script computes the real physical rotation of the foil relative to the motor frame (δθ).

**Step 4 — Enter the target image number**

Specify which raw image contains the crater you want to target. You can use the short format:
```
Enter target image (e.g. '57 35' or '57-35' or 'MoEDAL-057-035.png'): 57 35
```
The script will pad the numbers automatically → `MoEDAL-057-035.png`.

**Step 5 — Click on the point of interest in the interactive window**

A matplotlib window opens with the raw image displayed. All detected craters are shown as **cyan circles**. The tool starts in **FREE mode** (click anywhere on the image).

| Mode | Title bar shows | Behaviour |
|---|---|---|
| **FREE** (default) | `Mode: FREE (→ click position)` | Click anywhere — cross appears there, coords computed from click |
| **SNAP** | `Mode: SNAP (→ nearest crater)` | Click snaps to the nearest detected crater in the database |

Press **S** to toggle between the two modes at any time.

Left-click to mark a point — the terminal immediately prints:
- The click position in local pixels
- Its position in the global stitched canvas (pixels)
- Its **profilometer coordinates in µm**

Right-click to close the window.

**Step 6 — Read the final result**

```
=== FINAL TARGETING RESULT ===
  → MOVE PROFILOMETER TO:
     X = 15342.7 µm
     Y = 8901.3 µm
```
Enter these coordinates into the profilometer motor controller to move the stage to the crater.

### Transformation pipeline

```
Local pixel click (raw image, OpenCV convention)
    ↓  local_to_global_px()    ← accounts for crop_width_X / crop_height_Y overlap
Global pixel in full stitched canvas
    ↓  pixel_to_profilometer()
      1. Centre on R1_GLOBAL_PX
      2. Flip Y  (image y↓ → physical y↑)
      3. Multiply by pixel_resolution  (µm/px)
      4. Rotate by δθ  (= θ_machine − θ_code)
      5. Translate to R1 motor position
Profilometer coordinates (µm)
```

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

Open a terminal (Command Prompt / PowerShell on Windows, Terminal on macOS/Linux) and run:

```bash
git clone https://github.com/julesloisance-wq/NTD-ellipses
```

Then navigate into the project folder:

```bash
cd NTD-ellipses
```

You now have the full project on your machine.

---

### Step 3 — Set up `config.json`

Open `config.json` at the root of the project and fill in the following fields for your specific dataset:

```json
{
  "folder_path": "/absolute/path/to/your/raw/images/root",
  "element":     "subfolder_name_of_the_sheet_to_process",
  "save_folder": "/absolute/path/where/results/will/be/saved",
  ...
}
```

- `folder_path`: the root directory that contains your raw scanner images
- `element`: the name of the subfolder corresponding to the foil sheet you want to process
- `save_folder`: where all outputs (CSV, plots, logs) will be written — the folder will be created if it does not exist

> Leave the other parameters (`crop_width_X`, `crop_height_Y`, `min_area`, etc.) at their default values for a first run. See the **Parameters Guide** section below for full details.

---

### Step 4 — Run the pipeline

No manual installation of Python packages is needed. The launcher scripts handle everything automatically (virtual environment creation + dependency installation + pipeline execution).

**On Windows:**
Double-click `run_windows.bat`, or from your terminal:
```bat
run_windows.bat
```

**On macOS / Linux:**
From your terminal (make sure you are inside the project folder):
```bash
bash run_macOS_linux.sh
```

The script will:
1. Create a Python virtual environment (`.venv/`) if it doesn't exist yet
2. Install all required packages from `requirements.txt` (`opencv-python`, `numpy`, `matplotlib`, `Pillow`, `tqdm`)
3. Launch `main.py`

The terminal will stay open at the end so you can read the output.

---

### Step 5 — Follow the on-screen prompts

`main.py` is interactive. It will ask you to:
- Select a **primary reference hole** to act as the coordinate origin (0, 0)
- Confirm parameters before processing begins

Results (CSV files, plots, heatmaps) will be written to the `save_folder` you set in `config.json`.

---

## ⚙️ Quickstart Setup (existing users)

You do not need to manually configure python or install packages. Run scripts from the project root:

* **Windows:** Double-click on `run_windows.bat`
* **macOS/Linux:** Open a terminal and run `bash run_macOS_linux.sh`

These runner scripts automatically create a virtual environment (`.venv`), install dependencies, and execute the main pipeline.

To run the profilometer targeting tool after the pipeline has completed, activate the virtual environment manually and launch:
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

### Core Pipeline (Project Root)
* [main.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/main.py) - Main entrypoint. Performs grid scanning, manages reference hole caching, runs parallel ellipse detection, and saves exports.
* [ellipse_detection.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/ellipse_detection.py) - The computer vision core. Applies adaptive Gaussian blur, Canny edge detection, circularity filtering, and photometric qualification.
* [build_mosaics.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/build_mosaics.py) - Assembles the raw scanned images into local 3x3 tiles to make visual inspection of intermediate steps easier.
* [image_processing.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/image_processing.py) - Low-level image utilities (loading, cropping, raw conversions).
* [data_export.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/data_export.py) - Generates final CSV reports, histograms, and spatial heatmaps.
* [profilo_target.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/profilo_target.py) - **Primary scientific tool.** Interactive window: displays a raw image with all detected craters overlaid as cyan circles. Left-click to snap to the nearest crater and compute its exact profilometer motor coordinates (µm). Uses the same `local_to_global_px` → `pixel_to_profilometer` pipeline as `profilometer_recalibration.py`. Requires `main.py` to have been run first.
* [config.json](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/config.json) - Contains the configuration parameters for the current sheet.

### Calibration Tools (`scripts/calibration/`)
* [calculate_hole_positions.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/calibration/calculate_hole_positions.py) - Prints the exact global pixel coordinates of the reference holes in the stitched coordinate system.
* [profilometer_recalibration.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/calibration/profilometer_recalibration.py) - Free-click variant of the targeting tool for features that were **not detected** by the pipeline (e.g. scratches, manual points of interest). Unlike `profilo_target.py`, it does not snap to the crater database — you click anywhere on the raw image. R1/R2 global pixel coordinates and motor positions must be hardcoded at the top of the file before use.

### Presentation & Report Plots (`scripts/reports/`)
* [generate_overlap_checks.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/reports/generate_overlap_checks.py) - Generates 4 overlap check figures (2 horizontal, 2 vertical) to visually verify border alignment.
* [generate_report_figures.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/reports/generate_report_figures.py) - Generates synthetic figures illustrating the Canny detection pipeline steps.
* [generate_architecture_figure.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/reports/generate_architecture_figure.py) - Generates the visual diagram of the pipeline's software architecture.

### Development & Debugging (`scripts/debug/`)
* [debug_clahe.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/debug/debug_clahe.py) - Compares crater detection with and without CLAHE (local illumination equalization).
* [debug_filters.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/debug/debug_filters.py) - Side-by-side analysis of geometric, circularity, and intensity filters.
* [test_detection_debug.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/debug/test_detection_debug.py) - Runs detection on a few sample images and outputs annotated images inside `scripts/debug_output/` (showing accepted/rejected contours in color).

* [stitch_foil.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/debug/stitch_foil.py) - Stitches the full foil mosaic into one very large PNG file.

---

## 🔧 Parameters Guide (`config.json`)

* `folder_path` (string): Absolute path to the raw data root directory.
* `element` (string): Subfolder name of the foil sheet to process.
* `save_folder` (string): Path where pipeline results, logs, and plots will be written.
* `crop_width_X` (float): Horizontal crop size in pixels to compensate for scanning overlap between adjacent columns.
* `crop_height_Y` (float): Vertical crop size in pixels to compensate for scanning overlap between adjacent rows.
* `min_area` / `max_area` (int): Valid area range in pixels for the detected contours. Used to filter out dust or large defects.
* `min_intensity` / `max_intensity` (int): Grayscale intensity range for ellipse pixels. Used to filter out bright surface spots.
* `pixel_resolution` (float): Physical size of a pixel in micrometers per pixel (µm/px). Used to convert pixel distances into physical distances in µm.
* `angle_tolerance` (float): Angular deviation threshold in degrees.

---

## 📐 Calibration Protocol (For a new foil sheet)

When processing a new set of images, follow these steps to calibrate the overlap and resolution parameters:

### Step 1: Align Overlaps
1. Enter raw guesses for `crop_width_X` and `crop_height_Y` in `config.json`.
2. Run the script: `python scripts/reports/generate_overlap_checks.py`
3. Open the output figures in the `report_figures/` folder. Check if the overlapping borders match up. Adjust the crop values in `config.json` until the visual features align perfectly.

### Step 2: Calibrate Resolution (pixel_resolution)
1. Run `python main.py` and select a primary reference hole to act as origin (0,0).
2. Run `python scripts/calibration/calculate_hole_positions.py`. Write down the global pixel coordinates of the three reference holes (R1, R2, R3).
3. Measure the physical distance between these same three reference holes on the profilometer stage (in micrometers).
4. Calculate the optimal resolution in um/px by minimizing the error between the pixel distances and the physical distances using a Least-Squares fit:
   
   R = [ (d12 * D12) + (d23 * D23) + (d13 * D13) ] / [ d12^2 + d23^2 + d13^2 ]
   
   Where:
   * d_ij is the calculated distance in pixels.
   * D_ij is the measured physical distance in micrometers.
5. Put this calibrated value in `pixel_resolution` in `config.json`.