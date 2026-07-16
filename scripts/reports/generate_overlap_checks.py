#!/usr/bin/env python3
"""
generate_overlap_checks.py
==========================
Generates the 4 overlap check figures (2 horizontal, 2 vertical)
using overlap parameters loaded dynamically from config.json.
All labels, legends, and titles are in English.
Also deletes the obsolete figures from the report_figures/ directory.

Usage:
    python scripts/generate_overlap_checks.py
"""

import os
import json
import cv2
import matplotlib.pyplot as plt

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config.json"))
    with open(config_path, "r") as f:
        return json.load(f)

def clean_obsolete_files(output_dir):
    obsolete_files = [
        "vertical_overlap_check_324px.png",
        "extra_vertical_check_048_049_324px.png",
        "overlap_match_visual_craters_fixed.png",
        "extra_horizontal_check_054_055.png",
        "vertical_overlap_check_321px.png"
    ]
    for filename in obsolete_files:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"Removed obsolete file: {path}")
            except Exception as e:
                print(f"Warning: Could not remove {path}: {e}")

def main():
    config = load_config()
    
    # Extract crop/overlap sizes from config
    C_x = int(round(config.get("crop_width_X", 667)))
    C_y = int(round(config.get("crop_height_Y", 323.5)))
    
    data_dir = config.get("folder_path", "/Users/julesloisance/Desktop/StageHelsinki/MoEDAL_Data_Apr2025/")
    element = config.get("element", "O1_L8_ME18_UD")
    images_dir = os.path.join(data_dir, element)
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "report_figures"))
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Clean obsolete files
    clean_obsolete_files(output_dir)
    
    # Base dimensions of raw images
    w, h = 3840, 2748
    
    # -------------------------------------------------------------------------
    # FIGURE 1: Vertical Overlap Check (Base Pair: MoEDAL-043-054 vs MoEDAL-042-054)
    # -------------------------------------------------------------------------
    img_top_path = os.path.join(images_dir, "MoEDAL-043-054.png")
    img_bottom_path = os.path.join(images_dir, "MoEDAL-042-054.png")
    
    if os.path.exists(img_top_path) and os.path.exists(img_bottom_path):
        img_top = cv2.cvtColor(cv2.imread(img_top_path), cv2.COLOR_BGR2RGB)
        img_bottom = cv2.cvtColor(cv2.imread(img_bottom_path), cv2.COLOR_BGR2RGB)
        
        roi_top_bottom = img_top[h - C_y:h, 0:w]
        roi_bottom_top = img_bottom[0:C_y, 0:w]
        
        crop_x_start, crop_x_end = 1000, 2500
        roi_top_crop = roi_top_bottom[:, crop_x_start:crop_x_end]
        roi_bottom_crop = roi_bottom_top[:, crop_x_start:crop_x_end]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].imshow(roi_top_crop)
        axes[0].set_title(f"Bottom of Upper Image (MoEDAL-043-054) | Lower boundary (y={h-C_y} to {h})")
        axes[0].axis('on')
        
        axes[1].imshow(roi_bottom_crop)
        axes[1].set_title(f"Top of Lower Image (MoEDAL-042-054) | Upper boundary (y=0 to {C_y})")
        axes[1].axis('on')
        
        plt.suptitle(f"Vertical Overlap Check (C_y = {C_y} px)\n(X region = {crop_x_start} to {crop_x_end})", fontsize=14)
        plt.tight_layout()
        
        out_path1 = os.path.join(output_dir, "vertical_overlap_check.png")
        plt.savefig(out_path1, dpi=150)
        plt.close()
        print(f"Generated: {out_path1}")
    else:
        print("Warning: Vertical check base images not found.")

    # -------------------------------------------------------------------------
    # FIGURE 2: Extra Vertical Overlap Check (MoEDAL-049-054 vs MoEDAL-048-054)
    # -------------------------------------------------------------------------
    img_top_extra_path = os.path.join(images_dir, "MoEDAL-049-054.png")
    img_bottom_extra_path = os.path.join(images_dir, "MoEDAL-048-054.png")
    
    if os.path.exists(img_top_extra_path) and os.path.exists(img_bottom_extra_path):
        img_top_ex = cv2.cvtColor(cv2.imread(img_top_extra_path), cv2.COLOR_BGR2RGB)
        img_bottom_ex = cv2.cvtColor(cv2.imread(img_bottom_extra_path), cv2.COLOR_BGR2RGB)
        
        roi_top_bottom_ex = img_top_ex[h - C_y:h, 0:w]
        roi_bottom_top_ex = img_bottom_ex[0:C_y, 0:w]
        
        crop_x_start, crop_x_end = 1000, 2500
        roi_top_ex_crop = roi_top_bottom_ex[:, crop_x_start:crop_x_end]
        roi_bottom_ex_crop = roi_bottom_top_ex[:, crop_x_start:crop_x_end]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].imshow(roi_top_ex_crop)
        axes[0].set_title(f"Bottom of Upper Image (MoEDAL-049-054) | Lower boundary (y={h-C_y} to {h})")
        axes[0].axis('on')
        
        axes[1].imshow(roi_bottom_ex_crop)
        axes[1].set_title(f"Top of Lower Image (MoEDAL-048-054) | Upper boundary (y=0 to {C_y})")
        axes[1].axis('on')
        
        plt.suptitle(f"Extra Vertical Overlap Check (C_y = {C_y} px)\n(X region = {crop_x_start} to {crop_x_end})", fontsize=14)
        plt.tight_layout()
        
        out_path2 = os.path.join(output_dir, "extra_vertical_check.png")
        plt.savefig(out_path2, dpi=150)
        plt.close()
        print(f"Generated: {out_path2}")
    else:
        print("Warning: Extra vertical check images not found.")

    # -------------------------------------------------------------------------
    # FIGURE 3: Horizontal Overlap Check (Base Pair: MoEDAL-038-054 vs MoEDAL-038-053)
    # -------------------------------------------------------------------------
    img_left_path = os.path.join(images_dir, "MoEDAL-038-054.png")
    img_right_path = os.path.join(images_dir, "MoEDAL-038-053.png")
    
    if os.path.exists(img_left_path) and os.path.exists(img_right_path):
        img_left = cv2.cvtColor(cv2.imread(img_left_path), cv2.COLOR_BGR2RGB)
        img_right = cv2.cvtColor(cv2.imread(img_right_path), cv2.COLOR_BGR2RGB)
        
        roi_left = img_left[0:h, w - C_x:w]
        roi_right = img_right[0:h, 0:C_x]
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 16))
        axes[0].imshow(roi_left)
        axes[0].set_title(f"Left Image (MoEDAL-038-054)\nRight boundary (x={w-C_x} to {w})")
        axes[0].axis('on')
        
        axes[1].imshow(roi_right)
        axes[1].set_title(f"Right Image (MoEDAL-038-053)\nLeft boundary (x=0 to {C_x})")
        axes[1].axis('on')
        
        plt.suptitle(f"Visual Evidence of Crater Overlap (C_x = {C_x} px)\n(Horizontal axis corresponds to 2nd number in file name)", fontsize=14)
        plt.tight_layout()
        
        out_path3 = os.path.join(output_dir, "horizontal_overlap_check.png")
        plt.savefig(out_path3, dpi=150)
        plt.close()
        print(f"Generated: {out_path3}")
    else:
        print("Warning: Horizontal check base images not found.")

    # -------------------------------------------------------------------------
    # FIGURE 4: Extra Horizontal Overlap Check (MoEDAL-048-055 vs MoEDAL-048-054)
    # -------------------------------------------------------------------------
    img_left_extra_path = os.path.join(images_dir, "MoEDAL-048-055.png")
    img_right_extra_path = os.path.join(images_dir, "MoEDAL-048-054.png")
    
    if os.path.exists(img_left_extra_path) and os.path.exists(img_right_extra_path):
        img_left_ex = cv2.cvtColor(cv2.imread(img_left_extra_path), cv2.COLOR_BGR2RGB)
        img_right_ex = cv2.cvtColor(cv2.imread(img_right_extra_path), cv2.COLOR_BGR2RGB)
        
        roi_left_ex = img_left_ex[0:h, w - C_x:w]
        roi_right_ex = img_right_ex[0:h, 0:C_x]
        
        # Zoom in Y to see craters clearly
        crop_y_start, crop_y_end = 1000, 2000
        roi_left_ex_crop = roi_left_ex[crop_y_start:crop_y_end, :]
        roi_right_ex_crop = roi_right_ex[crop_y_start:crop_y_end, :]
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        axes[0].imshow(roi_left_ex_crop)
        axes[0].set_title(f"Right boundary of Left Image (MoEDAL-048-055)\n(x={w-C_x} to {w})")
        axes[0].axis('on')
        
        axes[1].imshow(roi_right_ex_crop)
        axes[1].set_title(f"Left boundary of Right Image (MoEDAL-048-054)\n(x=0 to {C_x})")
        axes[1].axis('on')
        
        plt.suptitle(f"Extra Horizontal Overlap Check (C_x = {C_x} px)\nMoEDAL-048-055 (Left) vs MoEDAL-048-054 (Right)", fontsize=14)
        plt.tight_layout()
        
        out_path4 = os.path.join(output_dir, "extra_horizontal_check.png")
        plt.savefig(out_path4, dpi=150)
        plt.close()
        print(f"Generated: {out_path4}")
    else:
        print("Warning: Extra horizontal check images not found.")

if __name__ == "__main__":
    main()
