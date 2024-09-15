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

Improved by: [Your Name]
Date: [Current Date]
"""

import cv2
import numpy as np
import os
import time

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

def initialize_cameras():
    """
    Initialize and start the left and right cameras.

    Returns:
    - cam1: Left camera object
    - cam2: Right camera object
    """
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
    return cam1, cam2

def load_calibration_data(calibration_file, calibration_file_c1, calibration_file_c2):
    """
    Load stereo calibration data from files.

    Parameters:
    - calibration_file: Path to stereo calibration file
    - calibration_file_c1: Path to left camera calibration file
    - calibration_file_c2: Path to right camera calibration file

    Returns:
    - lod_data: Stereo calibration data
    - lod_datac1: Left camera calibration data
    - lod_datac2: Right camera calibration data
    """
    # Check if calibration files exist
    if not (os.path.exists(calibration_file) and 
            os.path.exists(calibration_file_c1) and 
            os.path.exists(calibration_file_c2)):
        raise FileNotFoundError("Calibration files not found. Please ensure that calibration files are available.")
    
    # Load calibration data
    lod_data = getStereoCameraParameters(calibration_file)
    lod_datac1 = getStereoSingleCameraParameters(calibration_file_c1)
    lod_datac2 = getStereoSingleCameraParameters(calibration_file_c2)
    return lod_data, lod_datac1, lod_datac2

def compute_rectification_maps(camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, image_size):
    """
    Compute rectification maps for stereo images.

    Parameters:
    - camera_matrix_left: Left camera matrix
    - dist_coeffs_left: Left camera distortion coefficients
    - camera_matrix_right: Right camera matrix
    - dist_coeffs_right: Right camera distortion coefficients
    - R: Rotation matrix
    - T: Translation vector
    - image_size: Size of the images

    Returns:
    - map1_left, map2_left: Rectification maps for left image
    - map1_right, map2_right: Rectification maps for right image
    - Q: Disparity-to-depth mapping matrix
    """
    # Stereo Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        camera_matrix_left, dist_coeffs_left, 
        camera_matrix_right, dist_coeffs_right, 
        image_size, R, T
    )
    
    # Create rectification maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2
    )
    
    return map1_left, map2_left, map1_right, map2_right, Q

def initialize_stereo_matcher_sgbm(num_disp=16, block_s=5):
    """
    Initialize the StereoSGBM matcher and WLS filter for disparity computation.

    Parameters:
    - num_disp: Number of disparities
    - block_s: Block size

    Returns:
    - matcher_left: Left stereo matcher
    - matcher_right: Right stereo matcher
    - wls_filter: WLS filter for disparity refinement
    """
    # Create left and right matchers for WLS filtering
    min_disp = 0
    num_disp = num_disp
    block_s = block_s
    matcher_left = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_s,
        P1=8 * 3 * block_s ** 2,
        P2=32 * 3 * block_s ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    matcher_right = cv2.ximgproc.createRightMatcher(matcher_left)
    
    # Create WLS filter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.2)
    
    return matcher_left, matcher_right, wls_filter

def handle_key_events(k, stereo, wls_filter):
    """
    Handle key events for adjusting stereo parameters.

    Parameters:
    - k: Key pressed
    - stereo: Stereo matcher object
    - wls_filter: WLS filter object

    Returns:
    - True to continue, False to exit
    """
    if k == ord('x'):
        return False
    elif k == ord('q'):
        # Increase block size
        block_s = stereo.getBlockSize() + 2
        if block_s % 2 == 0:
            block_s += 1
        if block_s > 11:
            block_s = 11
        stereo.setBlockSize(block_s)
        print(f"Block size increased to: {block_s}")
    elif k == ord('a'):
        # Decrease block size
        block_s = max(3, stereo.getBlockSize() - 2)
        if block_s % 2 == 0:
            block_s -= 1
        stereo.setBlockSize(block_s)
        print(f"Block size decreased to: {block_s}")
    elif k == ord('w'):
        # Increase number of disparities
        num_disp = stereo.getNumDisparities() + 16
        num_disp = (num_disp // 16) * 16
        if num_disp > 256:
            num_disp = 256
        stereo.setNumDisparities(num_disp)
        print(f"Number of disparities increased to: {num_disp}")
    elif k == ord('s'):
        # Decrease number of disparities
        num_disp = max(16, stereo.getNumDisparities() - 16)
        stereo.setNumDisparities(num_disp)
        print(f"Number of disparities decreased to: {num_disp}")
    elif k == ord('e'):
        # Increase WLS lambda
        current_lambda = wls_filter.getLambda()
        wls_filter.setLambda(current_lambda + 10000)
        print(f"WLS Lambda increased to: {wls_filter.getLambda()}")
    elif k == ord('d'):
        # Decrease WLS lambda
        current_lambda = wls_filter.getLambda()
        wls_filter.setLambda(max(0, current_lambda - 10000))
        print(f"WLS Lambda decreased to: {wls_filter.getLambda()}")
    elif k == ord('r'):
        # Increase WLS sigma color
        current_sigma = wls_filter.getSigmaColor()
        wls_filter.setSigmaColor(current_sigma + 0.1)
        print(f"WLS Sigma Color increased to: {wls_filter.getSigmaColor()}")
    elif k == ord('f'):
        # Decrease WLS sigma color
        current_sigma = wls_filter.getSigmaColor()
        wls_filter.setSigmaColor(max(0.1, current_sigma - 0.1))
        print(f"WLS Sigma Color decreased to: {wls_filter.getSigmaColor()}")
    return True

def process_frames(cam1, cam2, map1_left, map2_left, map1_right, map2_right, matcher_left, matcher_right, wls_filter, Q):
    """
    Main loop for processing frames and displaying results.

    Parameters:
    - cam1: Left camera object
    - cam2: Right camera object
    - map1_left, map2_left: Rectification maps for left image
    - map1_right, map2_right: Rectification maps for right image
    - matcher_left: Left stereo matcher
    - matcher_right: Right stereo matcher
    - wls_filter: WLS filter for disparity refinement
    - Q: Disparity-to-depth mapping matrix
    """
    print("Starting real-time stereo processing. Press 'x' to exit.")
    print("Press 'q/a' to increase/decrease block size.")
    print("Press 'w/s' to increase/decrease number of disparities.")
    print("Press 'e/d' to increase/decrease WLS lambda.")
    print("Press 'r/f' to increase/decrease WLS sigma color.")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        try:
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

            # Compute disparity maps for left and right images
            disparity_left = matcher_left.compute(gray_left, gray_right).astype(np.int16)
            disparity_right = matcher_right.compute(gray_right, gray_left).astype(np.int16)
            
            # Apply WLS filter
            filtered_disparity = wls_filter.filter(disparity_left, gray_left, None, disparity_right)
            
            # Normalize the filtered disparity map for visualization
            normalized_disparity_map = cv2.normalize(
                src=filtered_disparity,
                dst=None,
                beta=0,
                alpha=255,
                norm_type=cv2.NORM_MINMAX
            )
            normalized_disparity_map = np.uint8(normalized_disparity_map)
            
            # Apply colormap
            colormap_image = cv2.applyColorMap(normalized_disparity_map, cv2.COLORMAP_JET)

            # Combine rectified images side by side with horizontal lines
            combined_images = cv2.hconcat([rectified_left, rectified_right])
            combined_images = draw_lines(combined_images)

            # Resize images for display
            display_combined = cv2.resize(combined_images, (0, 0), fx=0.6, fy=0.6)
            display_disparity = cv2.resize(colormap_image, (0, 0), fx=0.6, fy=0.6)

            # Calculate fps
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Put fps on the image
            cv2.putText(display_disparity, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show the combined image and disparity map
            cv2.imshow('Stereo Images and Disparity Map', 
                       cv2.hconcat([display_combined, display_disparity]))
            
            # Handle key events
            k = cv2.waitKey(1) & 0xFF  # Use waitKey(1) for faster response

            if not handle_key_events(k, matcher_left, wls_filter):
                print("Exiting...")
                break
        except Exception as e:
            print(f"An error occurred during processing: {e}")
            break

    # Cleanup
    cv2.destroyAllWindows()

# ----------------------- Main Processing Function ----------------------- #

def main():
    try:
        # Optimize OpenCV settings
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)  # Adjust this based on your system's capabilities
        
        # Initialize cameras
        cam1, cam2 = initialize_cameras()
        
        # Load calibration data
        calibration_file = 'jetson_stereo_8MP.npz'  # Updated calibration file name
        calibration_file_c1 = 'jetson_stereo_8MPc1.npz'
        calibration_file_c2 = 'jetson_stereo_8MPc2.npz'
        
        lod_data, lod_datac1, lod_datac2 = load_calibration_data(calibration_file, calibration_file_c1, calibration_file_c2)
        
        # Unpack data
        camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T = lod_data

        # Capture a single frame to get image size
        ret, img_s = cam1.read()
        if ret:
            image_size = (img_s.shape[1], img_s.shape[0])
            print("Image size:", image_size)
        else:
            print("Failed to capture image from camera 1.")
            raise RuntimeError("Failed to capture image from camera 1.")
        
        # Compute rectification maps
        map1_left, map2_left, map1_right, map2_right, Q = compute_rectification_maps(
            camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, image_size
        )
        
        # Initialize stereo matcher
        matcher_left, matcher_right, wls_filter = initialize_stereo_matcher_sgbm()
        
        # Process frames
        process_frames(cam1, cam2, map1_left, map2_left, map1_right, map2_right, matcher_left, matcher_right, wls_filter, Q)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cleanup
        cam1.stop()
        cam2.stop()
        cam1.release()
        cam2.release()

if __name__ == "__main__":
    main()

