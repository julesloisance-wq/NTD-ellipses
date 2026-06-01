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

    # Rotation angle of the physical foil relative to the scan coordinate frame
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

    # Rigid-body transformation applying the foil rotation:
    # X_target = Xm1 + Xcode*cos(dTheta) + Ycode*sin(dTheta)
    # Y_target = Ym1 + Xcode*sin(dTheta) - Ycode*cos(dTheta)
    x_cible = x_m1 + x_code * math.cos(delta_theta) + y_code * math.sin(delta_theta)
    y_cible = y_m1 + x_code * math.sin(delta_theta) - y_code * math.cos(delta_theta)

    print("\n=== TARGETING RESULT ===")
    print(f"Selected crater: {target_img_name}  (area: {cible['area_um2']:.1f} µm²)")
    print(f"-> MOVE PROFILOMETER TO:")
    print(f"   X = {x_cible:.4f} mm")
    print(f"   Y = {y_cible:.4f} mm")

if __name__ == "__main__":
    main()