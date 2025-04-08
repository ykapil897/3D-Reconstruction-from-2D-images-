#!/usr/bin/env python3
import argparse
import os
import sys
import pathlib
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import functions from Temple.py
from Temple import (
    OpenCVStereoMatcher, 
    rectify_and_show_results, 
    compute_and_show_disparity, 
    reproject_and_save_ply,
    save_point_cloud_multiview,
    read_and_rotate_images
)
from load_camera_info_temple import load_all_camera_parameters_temple, load_all_camera_parameters_dino
from load_ply import save_ply
import collections

def main():
    parser = argparse.ArgumentParser(description='3D Reconstruction from Stereo Images')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--topology', type=str, default='skipping_2', 
                       choices=['360', 'overlapping', 'adjacent', 'skipping_1', 'skipping_2'],
                       help='Camera topology to use')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up the folder path
    folder_path = Path(args.input_dir)
    
    # Get all image files
    image_files = sorted(list(folder_path.glob('*.png')) + list(folder_path.glob('*.jpg')))
    
    if not image_files:
        print(f"No image files found in {args.input_dir}")
        return 1
    
    # Read images
    images = [str(img) for img in image_files]
    images_cv = read_and_rotate_images(images)
    
    if not images_cv:
        print("Failed to load images.")
        return 1
    
    # Get image dimensions
    h, w, d = images_cv[0].shape
    print(f"Image dimensions: {h}x{w}x{d}")
    
    # Define topologies
    topologies = collections.OrderedDict()
    topologies['360'] = tuple(zip((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
                                  (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0)))
    topologies['overlapping'] = tuple(zip((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                                          (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)))
    topologies['adjacent'] = tuple(zip((0, 2, 4, 6, 8, 10),
                                       (1, 3, 5, 7, 9, 11)))
    topologies['skipping_1'] = tuple(zip((0, 3, 6, 9),
                                         (1, 4, 7, 10)))
    topologies['skipping_2'] = tuple(zip((0, 4, 8),
                                         (1, 5, 9)))
    
    # Define options
    StereoRectifyOptions = {
        'imageSize': (w, h),
        'flags': (0, cv2.CALIB_ZERO_DISPARITY)[0],
        'newImageSize': (w, h),
        'alpha': 0.5
    }
    RemapOptions = {
        'interpolation': cv2.INTER_LINEAR
    }
    CameraArrayOptions = {
        'channels': 3,
        'num_cameras': len(images_cv),
        'topology': args.topology
    }
    StereoMatcherOptions = {
        'MinDisparity': 0,
        'NumDisparities': 64,
        'BlockSize': 7,
        'Disp12MaxDiff': 0,
        'PreFilterCap': 0,
        'UniquenessRatio': 15,
        'SpeckleWindowSize': 50,
        'SpeckleRange': 1
    }
    StereoSGBMOptions = {
        'PreFilterCap': 0,
        'UniquenessRatio': 0,
        'P1': 8,
        'P2': 32,
    }
    FinalOptions = {
        'StereoRectify': StereoRectifyOptions,
        'StereoMatcher': StereoMatcherOptions,
        'StereoSGBM': StereoSGBMOptions,
        'CameraArray': CameraArrayOptions,
        'Remap': RemapOptions
    }
    
    # Initialize OpenCVStereoMatcher
    opencv_matcher = OpenCVStereoMatcher(options=FinalOptions, calibration_path=folder_path)
    opencv_matcher.images = images_cv
    
    # Process image pairs
    index = 0
    output_folder = args.output_dir
    
    # Rectification
    print("\nRectification")
    left_image_rectified, right_image_rectified = rectify_and_show_results(
        opencv_matcher, image_index=index, show_image=True, output_folder=output_folder)
    
    # Disparity
    print("\nDisparity")
    disparity_img = compute_and_show_disparity(
        opencv_matcher, left_image_rectified, right_image_rectified, show_image=True)
    plt.savefig(os.path.join(output_folder, 'disparity_map.png'), dpi=300, bbox_inches='tight')
    
    # Project to 3D
    print("\nProject to 3D")
    output_file_path = reproject_and_save_ply(disparity_img, opencv_matcher, index, output_folder)
    
    # Run all image pairs
    xyz = opencv_matcher.run()
    ply_filename = f"{os.path.basename(args.input_dir)}_skipping_2"
    save_ply(xyz, ply_filename, output_folder)
    output_file_path = os.path.join(output_folder, f"{ply_filename}.ply")
    print(f"Saving: {output_file_path}")
    
    # Save point cloud visualizations
    try:
        # Load the saved point cloud
        with open(output_file_path, 'r') as f:
            lines = f.readlines()
            
        # Skip header and read points
        header_end = 0
        for i, line in enumerate(lines):
            if "end_header" in line:
                header_end = i + 1
                break
                
        points_data = []
        for i in range(header_end, len(lines)):
            if lines[i].strip():
                coords = lines[i].strip().split()
                points_data.append([float(coords[0]), float(coords[1]), float(coords[2])])
                
        points = np.array(points_data)
        
        # Save multi-view visualizations
        save_point_cloud_multiview(points, output_folder)
    except Exception as e:
        print(f"Failed to save point cloud visualizations: {e}")
    
    print("\nProcessing completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())