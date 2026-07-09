import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def create_overlap_visual():
    dir_path = "/Users/julesloisance/Desktop/StageHelsinki/MoEDAL_Data_Apr2025/O1_L8_ME18_UD"
    # Correct horizontal pair: same first index (038), different second index (054 vs 053)
    img_left_path = os.path.join(dir_path, "MoEDAL-038-054.png")
    img_right_path = os.path.join(dir_path, "MoEDAL-038-053.png")
    
    out_dir = "/Users/julesloisance/Desktop/StageHelsinki/NTD-ellipse/report_figures"
    os.makedirs(out_dir, exist_ok=True)
    
    img_left = cv2.cvtColor(cv2.imread(img_left_path), cv2.COLOR_BGR2RGB)
    img_right = cv2.cvtColor(cv2.imread(img_right_path), cv2.COLOR_BGR2RGB)
    
    y_min, y_max = 0, 2748
    x_overlap_size = 664
    step_x = 3840 - 664
    
    roi_left = img_left[y_min:y_max, step_x:3840].copy()
    roi_right = img_right[y_min:y_max, 0:x_overlap_size].copy()
                
    fig, axes = plt.subplots(1, 2, figsize=(8, 16))
    
    axes[0].imshow(roi_left)
    axes[0].set_title("Image Gauche (MoEDAL-038-054)\nBord droit (x=3176 à 3840)")
    axes[0].axis('on')
    
    axes[1].imshow(roi_right)
    axes[1].set_title("Image Droite (MoEDAL-038-053)\nBord gauche (x=0 à 664)")
    axes[1].axis('on')
    
    plt.suptitle("Preuve visuelle de chevauchement sur les cratères\n(Correction: axe horizontal = 2ème nombre du fichier)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overlap_match_visual_craters_fixed.png"), dpi=150)
    plt.close()

create_overlap_visual()
print("Figure generated.")
