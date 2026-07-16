# MoEDAL NTD Ellipse Detection & Alignment Pipeline

This software suite automates the detection, qualification, and spatial alignment of microscopic etched tracks (craters) on MoEDAL plastic NTD (Nuclear Track Detector) sheets. It processes raw, unstitched scanner images, stitches them into a virtual global coordinate system, and exports their physical locations.

---

## ⚙️ Quickstart Setup

You do not need to manually configure python or install packages. Run scripts from the project root:

* **Windows:** Double-click on `run_windows.bat`
* **macOS/Linux:** Open a terminal and run `bash run_macOS_linux.sh`

These runner scripts automatically create a virtual environment (`.venv`), install dependencies, and execute the main pipeline.

---

## 📁 Repository Structure & File Map

### Core Pipeline (Project Root)
* [main.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/main.py) - Main entrypoint. Performs grid scanning, manages reference hole caching, runs parallel ellipse detection, and saves exports.
* [ellipse_detection.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/ellipse_detection.py) - The computer vision core. Applies adaptive Gaussian blur, Canny edge detection, circularity filtering, and photometric qualification.
* [build_mosaics.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/build_mosaics.py) - Assembles the raw scanned images into local 3x3 tiles to make visual inspection of intermediate steps easier.
* [image_processing.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/image_processing.py) - Low-level image utilities (loading, cropping, raw conversions).
* [data_export.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/data_export.py) - Generates final CSV reports, histograms, and spatial heatmaps.
* [profilo_target.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/profilo_target.py) - Maps coordinates between the stitched pixel grid and the profilometer mechanical stage.
* [config.json](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/config.json) - Contains the configuration parameters for the current sheet.

### Calibration Tools (`scripts/calibration/`)
* [calculate_hole_positions.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/calibration/calculate_hole_positions.py) - Prints the exact global pixel coordinates of the reference holes in the stitched coordinate system.
* [profilometer_recalibration.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/calibration/profilometer_recalibration.py) - Interactive tool: left-click on a defect (e.g. a scratch) in a raw image, and it outputs the exact absolute coordinates (X, Y in micrometers) to target it on the physical profilometer stage.

### Presentation & Report Plots (`scripts/reports/`)
* [generate_overlap_checks.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/reports/generate_overlap_checks.py) - Generates 4 overlap check figures (2 horizontal, 2 vertical) to visually verify border alignment.
* [generate_report_figures.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/reports/generate_report_figures.py) - Generates synthetic figures illustrating the Canny detection pipeline steps.
* [generate_architecture_figure.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/reports/generate_architecture_figure.py) - Generates the visual diagram of the pipeline's software architecture.

### Development & Debugging (`scripts/debug/`)
* [debug_clahe.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/debug/debug_clahe.py) - Compares crater detection with and without CLAHE (local illumination equalization).
* [debug_filters.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/debug/debug_filters.py) - Side-by-side analysis of geometric, circularity, and intensity filters.
* [test_detection_debug.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/debug/test_detection_debug.py) - Runs detection on a few sample images and outputs annotated images inside `scripts/debug_output/` (showing accepted/rejected contours in color).
* [test_profilo_visual.py](file:///Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/scripts/debug/test_profilo_visual.py) - Interactive simulator validating the profilometer coordinate transformation.
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
* `pixel_resolution` (float): Physical size of a pixel in micrometers per pixel (um/px). Used to scale pixel distances to physical millimeters.
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