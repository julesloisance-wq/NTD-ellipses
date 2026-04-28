# MoEDAL Ellipse Detection Pipeline

This tool automates the detection, filtering, and statistical analysis of ellipses (heavy ion craters) from scanner images.

## ⚙️ Setup & Execution

You do not need to manually configure the Python environment.

1. Open `config.json` with a text editor.
2. Update the `folder_path` (where your raw images are) and `save_folder` (where you want the results).
3. Update the `element` name and the filtering thresholds (`min_intensity`, `max_intensity`, `angle_tolerance`).
4. **Run the code:**
   - **Windows:** Double-click on `run_windows.bat`
   - **macOS/Linux:** Open a terminal and run `bash run_macOS_linux.sh`

## 📊 Outputs

The script will automatically generate a new folder inside your `save_folder` containing:
- Stitched mosaic images (`.png`).
- Highlighted mosaics showing valid ellipses (Green = correct angle, Red = correct angle + correct intensity).
- JSON files containing the relative coordinates and geometries of valid ellipses.
- Global statistical histograms (Area and Angle distribution).