#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo Camera Calibration and Real-Time Depth Mapping

This script performs real-time stereo image processing to compute and display a disparity (depth) map.
It utilizes pre-calibrated stereo camera parameters to rectify images and compute depth information.

Dependencies:
- OpenCV
- NumPy
- Custom modules:
    - Camera.jetsonCam (ensure this module is available)
    - cv2.ximgproc (for advanced stereo algorithms)

Usage:
- Ensure that calibration files (`jetson_stereo_8MP_stereo.npz`, `jetson_stereo_8MP_c1.npz`, `jetson_stereo_8MP_c2.npz`) are available.
- Run the script to start real-time depth mapping.

Author: akhil_kk
Improved by: [Your Name]
Date: [Current Date]
"""

import cv2
import numpy as np
import os
import time

# Import custom camera module
import Camera.jetsonCam as jetCam

# ----------------------- Utility Functions ----------------------- #

def getStereoCameraParameters(file_name):
    """
    Load stereo camera parameters from a .npz file.

    Parameters:
    - file_name: Path to the .npz file containing stereo parameters

    Returns:
    - Dictionary containing all stereo calibration parameters
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Calibration file '{file_name}' not found.")

    loaded_data = np.load(file_name)
    required_keys = ['mtx1', 'dist1', 'mtx2', 'dist2', 'R', 'T', 'E', 'F', 'R1', 'R2', 'P1', 'P2', 'Q']
    if not all(key in loaded_data for key in required_keys):
        raise ValueError("Calibration file is missing some required parameters.")

    return {
        'mtx1': loaded_data['mtx1'],
        'dist1': loaded_data['dist1'],
        'mtx2': loaded_data['mtx2'],
        'dist2': loaded_data['dist2'],
        'R': loaded_data['R'],
        'T': loaded_data['T'],
        'E': loaded_data['E'],
        'F': loaded_data['F'],
        'R1': loaded_data['R1'],
        'R2': loaded_data['R2'],
        'P1': loaded_data['P1'],
        'P2': loaded_data['P2'],
        'Q': loaded_data['Q']
    }

def initialize_cameras():
    """
    Initialize and start the left and right cameras.

    Returns:
    - cam_left: Left camera object
    - cam_right: Right camera object
    """
    cam_left = jetCam.jetsonCam()
    cam_right = jetCam.jetsonCam()
    
    # Open left camera (sensor_id=1)
    cam_left.open(
        sensor_id=1,
        sensor_mode=3,
        flip_method=0,
        display_height=540,
        display_width=960
    )
    
    # Open right camera (sensor_id=0)
    cam_right.open(
        sensor_id=0,
        sensor_mode=3,
        flip_method=0,
        display_height=540,
        display_width=960
    )
    
    cam_left.start()
    cam_right.start()
    return cam_left, cam_right

def compute_rectification_maps(calib_params, image_size):
    """
    Compute rectification maps for stereo images.

    Parameters:
    - calib_params: Dictionary containing calibration parameters
    - image_size: Tuple (width, height) of the images

    Returns:
    - map_left_x, map_left_y: Rectification maps for the left image
    - map_right_x, map_right_y: Rectification maps for the right image
    """
    # Extract parameters
    mtx1 = calib_params['mtx1']
    dist1 = calib_params['dist1']
    mtx2 = calib_params['mtx2']
    dist2 = calib_params['dist2']
    R1 = calib_params['R1']
    R2 = calib_params['R2']
    P1 = calib_params['P1']
    P2 = calib_params['P2']

    # Compute rectification maps
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        mtx1, dist1, R1, P1, image_size, cv2.CV_16SC2)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        mtx2, dist2, R2, P2, image_size, cv2.CV_16SC2)
    return map_left_x, map_left_y, map_right_x, map_right_y

def initialize_stereo_matcher():
    """
    Initialize the StereoSGBM matcher and WLS filter for disparity computation.

    Returns:
    - stereo_matcher: StereoSGBM matcher object
    - wls_filter: WLS filter for disparity refinement
    """
    # StereoSGBM parameters
    min_disp = 0
    num_disp = 128  # Must be divisible by 16
    block_size = 5
    P1 = 8 * 3 * block_size ** 2
    P2 = 32 * 3 * block_size ** 2

    stereo_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Create right matcher for WLS filter
    right_matcher = cv2.ximgproc.createRightMatcher(stereo_matcher)

    # WLS filter parameters
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_matcher)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)

    return stereo_matcher, right_matcher, wls_filter

def draw_horizontal_lines(img, line_interval=30, color=(255, 0, 0)):
    """
    Draw horizontal lines on the image at specified intervals.

    Parameters:
    - img: Input image
    - line_interval: Interval between lines in pixels
    - color: Color of the lines (B, G, R)

    Returns:
    - img_with_lines: Image with lines drawn
    """
    img_with_lines = img.copy()
    for i in range(0, img.shape[0], line_interval):
        cv2.line(img_with_lines, (0, i), (img.shape[1], i), color, 1)
    return img_with_lines

def display_help():
    """
    Display help information for adjusting stereo parameters.
    """
    print("\nControls:")
    print("Press 'x' to exit.")
    print("Press 'q/a' to increase/decrease block size.")
    print("Press 'w/s' to increase/decrease number of disparities.")
    print("Press 'e/d' to increase/decrease WLS lambda.")
    print("Press 'r/f' to increase/decrease WLS sigma color.")
    print("Press 'h' to display this help message again.\n")

# ----------------------- Main Processing Function ----------------------- #

def main():
    try:
        # Optimize OpenCV settings
        cv2.setUseOptimized(True)
        cv2.setNumThreads(cv2.getNumberOfCPUs())

        # Initialize cameras
        cam_left, cam_right = initialize_cameras()
        
        # Load calibration data
        calibration_file = 'jetson_stereo_8MP_stereo.npz'
        calib_params = getStereoCameraParameters(calibration_file)

        # Capture a single frame to get image size
        ret_left, img_left_sample = cam_left.read()
        ret_right, img_right_sample = cam_right.read()
        if not ret_left or not ret_right:
            raise RuntimeError("Failed to capture images from cameras.")

        image_size = (img_left_sample.shape[1], img_left_sample.shape[0])
        print(f"Image size: {image_size}")

        # Compute rectification maps
        map_left_x, map_left_y, map_right_x, map_right_y = compute_rectification_maps(calib_params, image_size)

        # Initialize stereo matcher and WLS filter
        stereo_matcher, right_matcher, wls_filter = initialize_stereo_matcher()

        # Display help information
        display_help()

        # Main loop for processing frames
        while True:
            # Capture frames from both cameras
            ret_left, frame_left = cam_left.read()
            ret_right, frame_right = cam_right.read()

            if not ret_left or not ret_right:
                print("Failed to read from cameras. Exiting...")
                break

            # Rectify images
            rectified_left = cv2.remap(frame_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
            rectified_right = cv2.remap(frame_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

            # Convert to grayscale
            gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

            # Compute disparity maps
            disparity_left = stereo_matcher.compute(gray_left, gray_right).astype(np.int16)
            disparity_right = right_matcher.compute(gray_right, gray_left).astype(np.int16)

            # Apply WLS filter to refine disparity map
            filtered_disparity = wls_filter.filter(disparity_left, gray_left, None, disparity_right)
            filtered_disparity = cv2.normalize(filtered_disparity, None, 0, 255, cv2.NORM_MINMAX)
            filtered_disparity = np.uint8(filtered_disparity)

            # Apply colormap for visualization
            depth_colormap = cv2.applyColorMap(filtered_disparity, cv2.COLORMAP_JET)

            # Draw horizontal lines on rectified images for visualization
            rectified_combined = cv2.hconcat([rectified_left, rectified_right])
            rectified_with_lines = draw_horizontal_lines(rectified_combined)

            # Resize images for display
            display_rectified = cv2.resize(rectified_with_lines, (0, 0), fx=0.6, fy=0.6)
            display_depth = cv2.resize(depth_colormap, (0, 0), fx=0.6, fy=0.6)

            # Concatenate rectified images and depth map side by side
            display_combined = cv2.hconcat([display_rectified, display_depth])

            # Display the combined image
            cv2.imshow('Stereo Rectification and Depth Map', display_combined)

            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('x'):
                print("Exiting...")
                break
            elif key == ord('q'):
                # Increase block size
                block_size = stereo_matcher.getBlockSize() + 2
                if block_size % 2 == 0:
                    block_size += 1
                if block_size > 11:
                    block_size = 11
                stereo_matcher.setBlockSize(block_size)
                print(f"Block size increased to: {block_size}")
            elif key == ord('a'):
                # Decrease block size
                block_size = stereo_matcher.getBlockSize() - 2
                if block_size % 2 == 0:
                    block_size -= 1
                if block_size < 3:
                    block_size = 3
                stereo_matcher.setBlockSize(block_size)
                print(f"Block size decreased to: {block_size}")
            elif key == ord('w'):
                # Increase number of disparities
                num_disp = stereo_matcher.getNumDisparities() + 16
                if num_disp > 256:
                    num_disp = 256
                stereo_matcher.setNumDisparities(num_disp)
                print(f"Number of disparities increased to: {num_disp}")
            elif key == ord('s'):
                # Decrease number of disparities
                num_disp = stereo_matcher.getNumDisparities() - 16
                if num_disp < 16:
                    num_disp = 16
                stereo_matcher.setNumDisparities(num_disp)
                print(f"Number of disparities decreased to: {num_disp}")
            elif key == ord('e'):
                # Increase WLS lambda
                current_lambda = wls_filter.getLambda()
                wls_filter.setLambda(current_lambda + 1000)
                print(f"WLS Lambda increased to: {wls_filter.getLambda()}")
            elif key == ord('d'):
                # Decrease WLS lambda
                current_lambda = wls_filter.getLambda()
                new_lambda = max(0, current_lambda - 1000)
                wls_filter.setLambda(new_lambda)
                print(f"WLS Lambda decreased to: {wls_filter.getLambda()}")
            elif key == ord('r'):
                # Increase WLS sigma color
                current_sigma = wls_filter.getSigmaColor()
                wls_filter.setSigmaColor(current_sigma + 0.1)
                print(f"WLS Sigma Color increased to: {wls_filter.getSigmaColor()}")
            elif key == ord('f'):
                # Decrease WLS sigma color
                current_sigma = wls_filter.getSigmaColor()
                new_sigma = max(0.1, current_sigma - 0.1)
                wls_filter.setSigmaColor(new_sigma)
                print(f"WLS Sigma Color decreased to: {wls_filter.getSigmaColor()}")
            elif key == ord('h'):
                # Display help
                display_help()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release resources
        cv2.destroyAllWindows()
        cam_left.stop()
        cam_right.stop()
        cam_left.release()
        cam_right.release()
        print("Camera resources released.")

if __name__ == "__main__":
    main()

