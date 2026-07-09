import json
import math
import os

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    element     = config["element"]
    save_folder = config["save_folder"]
    json_path   = os.path.join(save_folder, element, f"all_data_{element}.json")

    if not os.path.exists(json_path):
        print(f"Error: data file not found at {json_path}")
        print("Make sure you have run main.py at least once to generate the JSON.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    try:
        theta_code = data["reference_system"]["theta_code_radians"]
        dist_theorique = data["reference_system"]["distance_R1_R2_um"]
    except KeyError:
        print("Error: reference system metadata (R1->R2) is missing from the JSON.")
        return

    print("--- PROFILOMETER TARGETING SYSTEM ---")
    print(f"Theoretical angle loaded: {math.degrees(theta_code):.2f}°")
    print(f"Expected R1->R2 distance: {dist_theorique / 1000:.2f} mm\n")

    print("Aim at the centre of reference hole R1.")
    x_m1 = float(input("Enter motor X for R1 (mm): "))
    y_m1 = float(input("Enter motor Y for R1 (mm): "))

    print("\nMove to reference hole R2...")
    x_m2 = float(input("Enter motor X for R2 (mm): "))
    y_m2 = float(input("Enter motor Y for R2 (mm): "))

    # Both angles are in Cartesian y-UP convention (profilometer y axis points UP).
    # theta_code is stored in y-UP by main.py; theta_machine is also y-UP since motor y points UP.
    # delta_theta > 0 means the foil is rotated counter-clockwise relative to the motor frame.
    theta_machine = math.atan2(y_m2 - y_m1, x_m2 - x_m1)
    delta_theta = theta_machine - theta_code

    print(f"\n[INFO] Physical foil rotation detected: {math.degrees(delta_theta):.3f}°")

    target_img_name = input("\nEnter target image name (e.g. MoEDAL-057-045.png): ")

    images_dict = data.get("images", {})
    if target_img_name not in images_dict:
        print("Error: this image has no valid craters in the JSON.")
        return

    ellipses_in_image = images_dict[target_img_name].get("ellipses", [])
    if not ellipses_in_image:
        print("No craters found in this image.")
        return

    # Placeholder: selects the first crater in the list
    cible = ellipses_in_image[0]

    # Convert JSON coordinates from µm to mm for motor commands
    x_code = cible["x_um"] / 1000.0
    y_code = cible["y_um"] / 1000.0

    # Rigid-body transformation (scan frame y-DOWN → motor frame y-UP).
    # Scan coordinates (x_code, y_code) use image convention: y points DOWN.
    # Motor frame uses Cartesian convention: y points UP.
    # Step 1 – flip scan y to Cartesian: y_code_cart = -y_code
    # Step 2 – rotate by delta_theta (Cartesian, CCW positive):
    #   x_motor = x_code * cos(dθ) - y_code_cart * sin(dθ)
    #           = x_code * cos(dθ) + y_code * sin(dθ)
    #   y_motor = x_code * sin(dθ) + y_code_cart * cos(dθ)
    #           = x_code * sin(dθ) - y_code * cos(dθ)
    x_cible = x_m1 + x_code * math.cos(delta_theta) + y_code * math.sin(delta_theta)
    y_cible = y_m1 + x_code * math.sin(delta_theta) - y_code * math.cos(delta_theta)
    # --- y-DOWN motor alternative (uncomment if profilometer y points DOWN) ---
    # x_cible = x_m1 + x_code * math.cos(delta_theta) - y_code * math.sin(delta_theta)
    # y_cible = y_m1 + x_code * math.sin(delta_theta) + y_code * math.cos(delta_theta)
    # NOTE: if motor y-DOWN, also negate theta_code above: theta_code = -theta_code

    print("\n=== TARGETING RESULT ===")
    print(f"Selected crater: {target_img_name}  (area: {cible['area_um2']:.1f} µm²)")
    print(f"-> MOVE PROFILOMETER TO:")
    print(f"   X = {x_cible:.4f} mm")
    print(f"   Y = {y_cible:.4f} mm")

if __name__ == "__main__":
    main()