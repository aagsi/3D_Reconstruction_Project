#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo Camera Calibration and Real-Time Depth Mapping

This script performs real-time stereo image processing to compute and display a disparity (depth) map.
It utilizes pre-calibrated stereo camera parameters to rectify images and compute depth information.

Dependencies:
- OpenCV
- NumPy
- Matplotlib (optional, for plotting)
- Custom modules:
    - Camera.jetsonCam (ensure this module is available)

Usage:
- Ensure that calibration files (`jetson_stereo_8MP_calibration.npz`, `jetson_stereo_8MPc1.npz`, `jetson_stereo_8MPc2.npz`) are available.
- Run the script to start real-time depth mapping.

Author: akhil_kk
Created on: Fri Apr 14 23:50:30 2023
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob

# Import custom camera module
# Ensure that Camera.jetsonCam is accessible in your Python path
import Camera.jetsonCam as jetCam

# ----------------------- Utility Functions ----------------------- #

def getStereoCameraParameters(file_name):
    """
    Load stereo camera parameters from a .npz file.

    Parameters:
    - file_name: Path to the .npz file containing stereo parameters

    Returns:
    - Tuple containing camera matrices, distortion coefficients, rotation, and translation
    """
    loaded_data = np.load(file_name)
    return (
        loaded_data['k1'],      # Camera Matrix Left
        loaded_data['d1'],      # Distortion Coeffs Left
        loaded_data['k2'],      # Camera Matrix Right
        loaded_data['d2'],      # Distortion Coeffs Right
        loaded_data['R'],       # Rotation Matrix
        loaded_data['T']        # Translation Vector
    )

def getStereoSingleCameraParameters(file_name):
    """
    Load individual camera parameters from a .npz file.

    Parameters:
    - file_name: Path to the .npz file containing individual camera parameters

    Returns:
    - Tuple containing camera matrix, distortion coefficients, rotation, and translation
    """
    loaded_data = np.load(file_name)
    return (
        loaded_data['k'],  # Camera Matrix
        loaded_data['d'],  # Distortion Coefficients
        loaded_data['r'],  # Rotation Vector
        loaded_data['t']   # Translation Vector
    )

def draw_lines(img):
    """
    Draw horizontal blue lines on the image at every 30 pixels.

    Parameters:
    - img: Input image

    Returns:
    - Image with lines drawn
    """
    for i in range(0, img.shape[0], 30):
        cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0), 1)
    return img  

# ----------------------- Main Processing Function ----------------------- #

def main():
    # Initialize cameras
    cam1 = jetCam.jetsonCam()
    cam2 = jetCam.jetsonCam()
    
    # Open left camera
    cam1.open(
        sensor_id=1,
        sensor_mode=3,
        flip_method=0,
        display_height=540,
        display_width=960
    )
    
    # Open right camera
    cam2.open(
        sensor_id=0,
        sensor_mode=3,
        flip_method=0,
        display_height=540,
        display_width=960
    )
    
    cam1.start()
    cam2.start()
    
    # Load stereo calibration parameters
    calibration_file = 'jetson_stereo_8MP.npz'  # Updated calibration file name
    calibration_file_c1 = 'jetson_stereo_8MPc1.npz'
    calibration_file_c2 = 'jetson_stereo_8MPc2.npz'
    
    # Check if calibration files exist
    if not (os.path.exists(calibration_file) and 
            os.path.exists(calibration_file_c1) and 
            os.path.exists(calibration_file_c2)):
        print("Calibration files not found. Please ensure that 'jetson_stereo_8MP.npz', 'jetson_stereo_8MPc1.npz', and 'jetson_stereo_8MPc2.npz' are available.")
        cam1.stop()
        cam2.stop()
        cam1.release()
        cam2.release()
        return
    
    # Load calibration data
    try:
        lod_data = getStereoCameraParameters(calibration_file)
        lod_datac1 = getStereoSingleCameraParameters(calibration_file_c1)
        lod_datac2 = getStereoSingleCameraParameters(calibration_file_c2)
    except KeyError as e:
        print(f"KeyError while loading calibration data: {e}")
        print("Please ensure that the calibration files contain the correct keys.")
        cam1.stop()
        cam2.stop()
        cam1.release()
        cam2.release()
        return
    
    print("Individual Camera 1 Matrix:\n", lod_datac1[0])
    print("Individual Camera 2 Matrix:\n", lod_datac2[0])
    print("Individual Camera 1 Dist Coefs:\n", lod_datac1[1])
    print("Individual Camera 2 Dist Coefs:\n", lod_datac2[1])
    
    camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T = lod_data
    
    print("Stereo Camera Matrix Left:\n", camera_matrix_left)
    print("Stereo Camera Matrix Right:\n", camera_matrix_right)
    
    # Capture a single frame to get image size
    ret, img_s = cam1.read()
    if ret:
        image_size = (img_s.shape[1], img_s.shape[0])
        print("Image size:", image_size)
    else:
        print("Failed to capture image from camera 1.")
        cam1.stop()
        cam2.stop()
        cam1.release()
        cam2.release()
        exit()
    
    # Stereo Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        camera_matrix_left, dist_coeffs_left, 
        camera_matrix_right, dist_coeffs_right, 
        image_size, R, T
    )
    
    # Initialize StereoBM matcher
    block_s = 5
    num_disp = 16
    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_s)
    
    # Create rectification maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2
    )
    
    print("Starting real-time stereo processing. Press 'x' to exit.")
    print("Press 'q/a' to increase/decrease block size.")
    print("Press 'w/s' to increase/decrease number of disparities.")
    
    while True:
        # Read stereo images
        ret_left, image_left = cam1.read()
        ret_right, image_right = cam2.read()

        if not ret_left or not ret_right:
            print("Failed to read from cameras.")
            break

        # Remap the images using rectification maps
        rectified_left = cv2.remap(image_left, map1_left, map2_left, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(image_right, map1_right, map2_right, cv2.INTER_LINEAR)

        # Convert images to grayscale
        gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

        # Compute disparity map
        disparity = stereo.compute(gray_left, gray_right)

        # Normalize the disparity map to the range [0, 1] for visualization
        normalized_disparity_map = cv2.normalize(
            disparity, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F
        )

        # Apply a colormap for better visualization
        colormap_image = cv2.applyColorMap(
            np.uint8(normalized_disparity_map * 255), cv2.COLORMAP_JET
        )

        # Combine rectified images side by side with horizontal lines
        combined_images = cv2.hconcat([rectified_left, rectified_right])
        combined_images = draw_lines(combined_images)

        # Resize images for display
        display_combined = cv2.resize(combined_images, (0, 0), fx=0.6, fy=0.6)
        display_disparity = cv2.resize(colormap_image, (0, 0), fx=0.6, fy=0.6)

        # Show the combined image and disparity map
        cv2.imshow('Stereo Images and Disparity Map', 
                   cv2.hconcat([display_combined, display_disparity]))
        
        # Handle key events
        k = cv2.waitKey(33)
        if k == ord('x'):
            print("Exiting...")
            break
        elif k == ord('q'):
            # Increase block size
            block_s += 2
            print(f"Block size increased to: {block_s}")
            stereo.setBlockSize(block_s)
        elif k == ord('a'):
            # Decrease block size
            block_s = max(block_s - 2, 5)
            print(f"Block size decreased to: {block_s}")
            stereo.setBlockSize(block_s)
        elif k == ord('w'):
            # Increase number of disparities
            num_disp += 16
            # Ensure num_disp is divisible by 16
            num_disp = (num_disp // 16) * 16
            print(f"Number of disparities increased to: {num_disp}")
            stereo.setNumDisparities(num_disp)
        elif k == ord('s'):
            # Decrease number of disparities
            num_disp = max(16, num_disp - 16)
            print(f"Number of disparities decreased to: {num_disp}")
            stereo.setNumDisparities(num_disp)

    # Cleanup
    cv2.destroyAllWindows()
    cam1.stop()
    cam2.stop()
    cam1.release()
    cam2.release()

if __name__ == "__main__":
    main()

