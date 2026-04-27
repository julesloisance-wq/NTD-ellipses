import json
import glob
import os
from image_processing import process_and_build_mosaics
from ellipse_detection import get_reference_center, analyze_ellipses
from data_export import export_json, export_histogram, export_angle_histogram_from_bins, export_highlighted_mosaic

def main():
    print("Starting the processing...")
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # 1. PROCESSING : Création des mosaïques
    save_folder = process_and_build_mosaics(config)
    print("Mosaics created and saved in:", save_folder)

    # 2. REFERENCE POINT SELECTION : Clic pour définir le centre du trou sur une mosaïque de référence
    print("\nAction required: Please select the reference mosaic and click on the center of the hole.")
    print(save_folder)

    # Loop until a valid existing file is provided
    while True:
        ref_mosaic_name = input("Enter the name of the reference mosaic (ex: Mosaic_2_1.png) : ")
        ref_mosaic_path = os.path.join(save_folder, ref_mosaic_name)
        
        # Check if the file exists
        if os.path.exists(ref_mosaic_path):
            break  # Exit the loop if the path is valid
        else:
            print(f"❌ The file '{ref_mosaic_name}' is not found. Please try again.")

    # Get the reference center coordinates from the user click
    ref_x0, ref_y0 = get_reference_center(ref_mosaic_path)
    print(f"Reference point selected at coordinates: x={ref_x0}, y={ref_y0}")
 
# 3. DETECTION SUR TOUTES LES MOSAIQUES
    mosaic_files = glob.glob(os.path.join(save_folder, "Mosaic_*.png"))
    
    global_red_ellipses_data = []
    
    for mosaic_path in mosaic_files:
        print(f"Analyse de {mosaic_path}...")
        
        # Unpack the three returned values
        ellipses_data, ellipse_histogram, dominant_angle = analyze_ellipses(mosaic_path, config, ref_x0, ref_y0)
        
        # Filter for JSON export and global area calculation (Red only)
        red_ellipses = [e for e in ellipses_data if e["category"] == "red"]
        global_red_ellipses_data.extend(red_ellipses)
        
        base_name = os.path.basename(mosaic_path).replace(".png", "")
        
        # Exports
        export_json(red_ellipses, f"{base_name}_ellipses_red.json", save_folder)
        export_angle_histogram_from_bins(ellipse_histogram, config["element"], base_name, save_folder)
        export_highlighted_mosaic(mosaic_path, ellipses_data, base_name, save_folder)

    # 4. EXPORT DE L'HISTOGRAMME GLOBAL DES AIRES
    print("Génération de l'histogramme des aires...")
    export_histogram(global_red_ellipses_data, config["element"], save_folder)

    print("Processing completed successfully.")

if __name__ == "__main__":
    main()