import os
import re
import glob
from PIL import Image
from tqdm import tqdm

def process_and_build_mosaics(config):
    folder_path = config["folder_path"]
    element = config["element"]
    save_folder = os.path.join(config["save_folder"], element)
    os.makedirs(save_folder, exist_ok=True)
    
    crop_width_X = config["crop_width_X"]
    crop_height_Y = config["crop_height_Y"]
    step = config["step"]
    num_columns = config["num_columns"]
    num_rows = config["num_rows"]

    
    # Logic from the notebook's "Parameters" cell 

    # Regular expression to extract i and j values
    pattern = re.compile(r'MoEDAL-(\d{3})-(\d{3})\.png') 

    # In order to find the intervals of i and j no matter what the images are
    i_values = []
    j_values = []

    total_width = num_columns * step      # 9 total width of the mosaics
    total_height = num_rows * step        # 12 total height of the mosaics

    # Loop through all files in the folder
    for filename in os.listdir(folder_path + element):
        match = pattern.match(filename)
        if match:
            i = int(match.group(1))
            j = int(match.group(2))
            i_values.append(i)
            j_values.append(j)
            
    # Compute min and max if matching files were found
    if i_values and j_values:
        i_min, i_max = min(i_values), max(i_values)
        j_min, j_max = min(j_values), max(j_values)
        print(f"i_min: {i_min}, i_max: {i_max}")
        print(f"j_min: {j_min}, j_max: {j_max}")
    else:
        print("No matching files found.")
        
    # Important i and j values     
    i_center = int((i_min + i_max) / 2)
    j_center = int((j_min + j_max) / 2)

    # Save the initial intervals for later absolute coordinate reference
    i_initial_min = i_min
    i_initial_max = i_max
    j_initial_min = j_min
    j_initial_max = j_max

    # Define the mosaic borders properly centered around i_center and j_center (i_min and i_max are redefined)
    i_min = i_center - (total_height // 2)
    i_max = i_min + total_height - 1  # to have exactly 12 images

    j_min = j_center - (total_width // 2)
    j_max = j_min + total_width - 1   # to have exactly 9 images

    print(i_min)
    print(i_max)
    print(j_min)
    print(j_max)

    # In case of having a non-multiple-of-3 length interval
    num_row_operations = (i_max + 1 - i_min) // 3
    num_column_operations = (j_max + 1 - j_min) // 3

    new_row_min = i_min
    new_column_min = j_min
    new_row_max = i_min + num_row_operations * 3 - 1
    new_column_max = j_min + num_column_operations * 3 - 1



    # Logic from the notebook's "Loading of Images" cell

    # Paths and images
    list_of_path_lists = []
    list_of_image_lists = []

    # Path generation
    for i in range(new_row_min, new_row_max + 1):
        path_list = [
            f"{folder_path}{element}/MoEDAL-{i:03}-{j:03}.png"  # Only the image name pattern may vary here
            for j in range(new_column_min, new_column_max + 1)
        ]
        list_of_path_lists.append(path_list)

    print(list_of_path_lists)

    # Image loading
    print("\nLoading images...")
    for path_list in tqdm(list_of_path_lists, desc="Rows"):
        images = []
        for path in path_list:
            if os.path.exists(path):
                images.append(Image.open(path))
            else:
                print(f"⚠️ Image not found: {path}")
        list_of_image_lists.append(images)

    print(list_of_image_lists)



    # Logic from the notebook's "Resize" cell

    # List of resized images
    list_of_resized_image_lists = []


    # Get the height of each image and determine the minimum
    heights = [img.size[1] for row in list_of_image_lists for img in row]
    new_height = min(heights)

    # Resize and crop each image
    for idx, row in enumerate(list_of_image_lists):
        resized_row = []
        print(f"Resizing row {idx + 1}...")

        for img in row:
            # In case all of the images don't have the same height, we resize them to the minimum height while keeping the aspect ratio
            new_width = int(img.size[0] * new_height / img.size[1])
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

            # Systematic cropping at the bottom
            resized_img = resized_img.crop((0, 0, resized_img.size[0], new_height - crop_height_Y))

            # Systematic cropping on the left
            resized_img = resized_img.crop((crop_width_X, 0, resized_img.size[0], resized_img.size[1]))

            resized_row.append(resized_img)

        list_of_resized_image_lists.append(resized_row)

    # Update main image list
    list_of_image_lists = list_of_resized_image_lists



    # Logic from the notebook's "Creation of mosaics" cell

    # Delete old mosaics
    for file in glob.glob(os.path.join(save_folder, "Mosaic*.png")):
        os.remove(file)

    # Delete old associated JSON files
    for file in glob.glob(os.path.join(save_folder, "Mosaic*.json")):
        os.remove(file)

    # Build and save mosaics
    for i in range(0, new_row_max - new_row_min + 1, step):
        row_images = [[] for _ in range(step)]

        for j in range(0, new_column_max - new_column_min, step):

            # Build the 3 rows
            for k in range(step):
                img1 = list_of_image_lists[i + k][j]
                img2 = list_of_image_lists[i + k][j + 1]
                img3 = list_of_image_lists[i + k][j + 2]

                target_height = img1.size[1]
                total_width = img1.size[0] + img2.size[0] + img3.size[0]

                new_row = Image.new('RGB', (total_width, target_height), (255, 255, 255))

                widths = [img3.size[0], img2.size[0], img1.size[0]]
                images = [img3, img2, img1]

                x_offset = 0
                x_positions = []
                for img in images:
                    x_positions.append(x_offset)
                    x_offset += img.size[0]

                for idx_img, img in enumerate(images):
                    new_row.paste(img, (x_positions[idx_img], 0))

                row_images[k] = new_row

            print(f"\nFinal vertical assembly of mosaic {int(i/3)+1} ; {int(j/3)+1}")
            total_height = sum(row.size[1] for row in row_images)
            final_width = max(row.size[0] for row in row_images)
            final_image = Image.new('RGB', (final_width, total_height), (255, 255, 255))

            y_offsets = [0, row_images[0].size[1], row_images[0].size[1] + row_images[1].size[1]]
            y_offsets = y_offsets[::-1]  # start from bottom

            for k in range(2, -1, -1):
                final_image.paste(row_images[k], (0, y_offsets[k]))

            # Save image
            filename = f"Mosaic_{int(i/3)+1}_{int(j/3)+1}.png"
            final_image.save(os.path.join(save_folder, filename))



    # The functions returns the path to the folder where the mosaics have been saved, which will be useful for the next steps
    return save_folder
