"""
update_parent_mosaics.py
------------------------
Patches the `parent_mosaic` field of every image entry in the global JSON
produced by data_export.py, using the same 3×3 block grid logic as
build_mosaics.py (Mosaic_{block_i+1}_{block_j+1}.png).

Usage:
    python update_parent_mosaics.py
    python update_parent_mosaics.py --config path/to/config.json
"""

import os
import re
import json
import argparse


def build_mosaic_mapping(target_dir: str) -> dict[str, str]:
    """
    Reproduces the grid-scanning and block-assignment logic of build_mosaics.py
    and returns a dict  {image_basename: mosaic_name}.

    e.g. {"MoEDAL-034-035.png": "Mosaic_1_1.png", ...}
    """
    pattern = re.compile(r'MoEDAL-(\d{3})-(\d{3})\.png')

    i_values, j_values = [], []
    for filename in os.listdir(target_dir):
        m = pattern.match(filename)
        if m:
            i_values.append(int(m.group(1)))
            j_values.append(int(m.group(2)))

    if not i_values:
        raise FileNotFoundError(f"No MoEDAL-XXX-XXX.png files found in {target_dir}")

    i_min, i_max = min(i_values), max(i_values)
    j_min, j_max = min(j_values), max(j_values)

    num_rows = i_max - i_min + 1
    num_cols = j_max - j_min + 1

    num_row_blocks = num_rows // 3
    num_col_blocks = num_cols // 3

    mapping: dict[str, str] = {}

    for block_i in range(num_row_blocks):
        for block_j in range(num_col_blocks):
            i_start = i_min + block_i * 3
            j_start = j_min + block_j * 3
            mosaic_name = f"Mosaic_{block_i + 1}_{block_j + 1}.png"

            for r in range(3):
                for c in range(3):
                    curr_i = i_start + r
                    curr_j = j_start + c
                    image_name = f"MoEDAL-{curr_i:03}-{curr_j:03}.png"
                    mapping[image_name] = mosaic_name

    return mapping


def patch_json(json_path: str, mapping: dict[str, str]) -> None:
    """
    Loads the JSON at json_path, fills in every `parent_mosaic` field
    using the provided mapping, and writes the result back in-place.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    images: dict = data.get("images", {})
    updated = 0
    not_found = []

    for img_name, img_data in images.items():
        mosaic = mapping.get(img_name)
        if mosaic:
            img_data["parent_mosaic"] = mosaic
            updated += 1
        else:
            not_found.append(img_name)

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"✅ {updated} image(s) updated with their parent mosaic.")
    if not_found:
        print(f"⚠️  {len(not_found)} image(s) had no matching mosaic "
              f"(outside usable 3×3 grid or unrecognised name):")
        for name in not_found:
            print(f"   – {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Fill parent_mosaic fields in the global JSON export."
    )
    parser.add_argument(
        "--config", default="config.json",
        help="Path to config.json (default: config.json in current directory)"
    )
    args = parser.parse_args()

    # --- load config ---
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    element     = config["element"]
    folder_path = config["folder_path"]
    save_folder = config["save_folder"]

    target_dir = os.path.join(folder_path, element)
    json_path  = os.path.join(save_folder, element, f"all_data_{element}.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    print(f"📂 Raw images directory : {target_dir}")
    print(f"📄 JSON file to patch   : {json_path}")

    # --- build mapping and patch ---
    mapping = build_mosaic_mapping(target_dir)
    print(f"🗺️  Mosaic mapping built  : {len(mapping)} image → mosaic associations")
    patch_json(json_path, mapping)


if __name__ == "__main__":
    main()
