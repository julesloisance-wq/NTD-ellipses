# MoEDAL Ellipse Detection Pipeline

This tool automates the detection, filtering, and statistical analysis of etched tracks (microscopic ellipses) directly from raw, unstitched scanner images. It perfectly maps local images to a global coordinate system using automated physical reference hole detection.

## ⚙️ Setup & Execution

You do not need to manually configure the Python environment.

1. Open `config.json` with a text editor.
2. Update the `folder_path` (where your raw images are) and `save_folder` (where you want the results).
3. Update the `element` name and the filtering thresholds (`min_area`, `max_area`, `min_intensity`, `max_intensity`).
4. **Run the code:**
   - **Windows:** Double-click on `run_windows.bat`
   - **macOS/Linux:** Open a terminal and run `bash run_macOS_linux.sh`

*(Note: During execution, an interactive window will pop up allowing you to visually select the main reference hole. Click "Select" to continue.)*

## 🚀 Pipeline Overview

1. **Grid Metadata Extraction**: Automatically parses grid dimensions (rows and columns) from the raw image filenames.
2. **Automated Reference Hole Detection**: Scans all images using contour detection and angular spread validation to find the large, jagged reference holes drilled into the plastic. The results are cached to speed up future runs.
3. **Interactive Selection**: Opens a GUI window allowing the user to browse the detected reference holes and select the best one to act as the absolute global origin `(0,0)`.
4. **Parallel Ellipse Detection**: Processes all raw images in parallel across all CPU cores. It detects the microscopic etched tracks ("red" ellipses) and calculates their exact global physical coordinates based on the chosen reference hole.

## 📊 Outputs

The script will automatically generate a new folder inside your `save_folder` containing:
- **Global JSON Database**: A comprehensive JSON file (`all_data_<element>.json`) containing the global coordinates, area, and angle of every valid red ellipse detected across the entire sheet.
- **Statistical Histograms**: Plots showing the distribution of ellipse Areas and Angles.
- **Global Spatial Heatmap**: A 2D heatmap plot showing the physical distribution and density of ellipses across the entire plastic sheet.