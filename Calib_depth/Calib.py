import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob

# Import your Jetson camera module
import Camera.jetsonCam as jetCam

def stereoCalibrateCamera(camera_c1, camera_c2, camera_name, chessboard_box_size=47, 
                          chessboard_grid_size=(7,7), number_of_frames=50):
    """
    Perform stereo camera calibration with interactive control for block size and min disparity.

    Parameters:
    - camera_c1, camera_c2: camera objects
    - camera_name: base name for saving calibration files
    - chessboard_box_size: size of a chessboard square in mm (default: 47 mm)
    - chessboard_grid_size: number of inner corners per a chessboard row and column (rows, columns) (default: (7,7))
    - number_of_frames: number of calibration frames to capture (default: 50)
    """
    # Define the dimensions of the checkerboard
    CHECKERBOARD = chessboard_grid_size

    # Chessboard square size in mm
    square_size = chessboard_box_size  

    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + 
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Flags for stereo calibration
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC     

    # Vectors to store 3D points and 2D points from both cameras
    threedpoints = []
    twodpoints_c1 = []
    twodpoints_c2 = []
   
    # 3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                   0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objectp3d *= square_size

    img_list_c1 =[] 
    img_list_c2 =[] 

    # Initialize adjustable parameters
    block_size = 11  # Default block size for cornerSubPix
    min_disparity = 160  # Default minimum disparity

    print("Calibration Setup:")
    print(f"Chessboard Grid Size (Inner Corners): {CHECKERBOARD[0]}x{CHECKERBOARD[1]}")
    print(f"Chessboard Square Size: {square_size} mm")
    print(f"Default Block Size: {block_size}")
    print(f"Default Minimum Disparity: {min_disparity}")
    print("\nControls during image capture:")
    print("Press 'c' to capture an image pair.")
    print("Press 'x' to abort the calibration process.")
    print("Press 'q' to increase block size.")
    print("Press 'a' to decrease block size.")
    print("Press 'w' to increase minimum disparity.")
    print("Press 's' to decrease minimum disparity.\n")

    img_count = 0
    while img_count < number_of_frames:
        # Capture frames from both cameras
        dat1_rcved, img1 = camera_c1.read()  
        dat2_rcved, img2 = camera_c2.read()      
        
        if not dat1_rcved or not dat2_rcved:
            print("Failed to grab frames from cameras.")
            break
        
        # Resize images for display
        img1_r = cv2.resize(img1, (0,0), fx=0.6, fy=0.6)
        img2_r = cv2.resize(img2, (0,0), fx=0.6, fy=0.6)
        
        # Concatenate images horizontally
        concatenated = cv2.hconcat([img1_r, img2_r])

        # Display current parameters on the image
        display_img = concatenated.copy()
        cv2.putText(display_img, f"Block Size: {block_size}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(display_img, f"Min Disparity: {min_disparity}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(display_img, f"Captured: {img_count}/{number_of_frames}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Camera Left - Camera Right", display_img)
        k = cv2.waitKey(10) & 0xFF

        if k == ord('c'):
            img_list_c1.append(img1)
            img_list_c2.append(img2)
            img_count +=1
            print(f"{img_count} image(s) captured.")
            
            # Save captured images for inspection
            cv2.imwrite(f"captured_c1_image_{img_count}.png", img1)
            cv2.imwrite(f"captured_c2_image_{img_count}.png", img2)
            print(f"Captured images saved as 'captured_c1_image_{img_count}.png' and 'captured_c2_image_{img_count}.png'.")
        
        elif k == ord('x'):
            cv2.destroyAllWindows()
            print('Capture terminated. Aborting calibration.')
            return
        elif k == ord('q'):
            block_size += 2
            if block_size > 31:
                block_size = 31  # Maximum block size
            print(f"Block Size increased to {block_size}.")
        elif k == ord('a'):
            block_size -= 2
            if block_size < 3:
                block_size = 3  # Minimum block size
            print(f"Block Size decreased to {block_size}.")
        elif k == ord('w'):
            min_disparity += 10
            if min_disparity > 500:
                min_disparity = 500  # Maximum min disparity
            print(f"Minimum Disparity increased to {min_disparity}.")
        elif k == ord('s'):
            min_disparity -= 10
            if min_disparity < 0:
                min_disparity = 0  # Minimum min disparity
            print(f"Minimum Disparity decreased to {min_disparity}.")

    cv2.destroyAllWindows()
    
    # Process each pair of captured images
    for idx, (image1, image2) in enumerate(zip(img_list_c1, img_list_c2), start=1):
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Optional: Enhance image contrast and reduce noise
        gray1 = cv2.equalizeHist(gray1)
        gray2 = cv2.equalizeHist(gray2)
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

        # Find chessboard corners in both images
        ret1, corners1 = cv2.findChessboardCorners(
                        gray1, CHECKERBOARD, 
                        cv2.CALIB_CB_ADAPTIVE_THRESH + 
                        cv2.CALIB_CB_FAST_CHECK + 
                        cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret2, corners2 = cv2.findChessboardCorners(
                        gray2, CHECKERBOARD, 
                        cv2.CALIB_CB_ADAPTIVE_THRESH + 
                        cv2.CALIB_CB_FAST_CHECK + 
                        cv2.CALIB_CB_NORMALIZE_IMAGE)
      
        # If corners are found in both images
        if ret1 and ret2:
            threedpoints.append(objectp3d)
      
            # Refine corner positions using the current block size
            cornersf_1 = cv2.cornerSubPix(
                gray1, corners1, (block_size, block_size), (-1, -1), criteria)
            cornersf_2 = cv2.cornerSubPix(
                gray2, corners2, (block_size, block_size), (-1, -1), criteria)
      
            twodpoints_c1.append(cornersf_1)
            twodpoints_c2.append(cornersf_2)
      
            # Draw and display the corners
            image1_drawn = cv2.drawChessboardCorners(image1.copy(), 
                                              CHECKERBOARD, 
                                              cornersf_1, ret1)

            image2_drawn = cv2.drawChessboardCorners(image2.copy(), 
                                              CHECKERBOARD, 
                                              cornersf_2, ret2)
      
            # Concatenate corner images for display
            concatenated_corners = cv2.hconcat([
                cv2.resize(image1_drawn, (0,0), fx=0.6, fy=0.6),
                cv2.resize(image2_drawn, (0,0), fx=0.6, fy=0.6)
            ])
            cv2.imshow("Corners Detected - Left and Right", concatenated_corners)
            cv2.waitKey(100)  # Display for a short duration
            print(f"Processed image pair {idx}.")
        else:
            print(f"Chessboard corners not found in image pair {idx}. Skipping.")
            # Save the problematic images for inspection
            cv2.imwrite(f"failed_c1_image_{idx}.png", image1)
            cv2.imwrite(f"failed_c2_image_{idx}.png", image2)
            print(f"Failed images saved as 'failed_c1_image_{idx}.png' and 'failed_c2_image_{idx}.png'.")

    cv2.destroyAllWindows()
    
    if not threedpoints or not twodpoints_c1 or not twodpoints_c2:
        print("Insufficient valid image pairs for calibration.")
        return
    
    # Assume all images have the same size
    width = img_list_c1[0].shape[1]
    height = img_list_c1[0].shape[0]

    # Perform camera calibration for each camera individually
    ret_1, k1, d1, r_1, t_1 = cv2.calibrateCamera(
        threedpoints, twodpoints_c1, (width, height), None, None)

    ret_2, k2, d2, r_2, t_2 = cv2.calibrateCamera(
        threedpoints, twodpoints_c2, (width, height), None, None)

    # Save individual camera parameters
    np.savez(f"{camera_name}c1.npz", k=k1, d=d1, r=r_1, t=t_1)
    np.savez(f"{camera_name}c2.npz", k=k2, d=d2, r=r_2, t=t_2)
    
    # Perform stereo calibration
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(
        threedpoints, twodpoints_c1, twodpoints_c2, 
        k1, d1, k2, d2, (width, height), 
        criteria=criteria, flags=stereocalibration_flags)
    
    # Save stereo calibration parameters
    np.savez(f"{camera_name}.npz", k1=k1, d1=d1, k2=k2, d2=d2, R=R, T=T)    
    print("Stereo calibration completed and parameters saved.")

def getStereoCameraParameters(file_name):
    """
    Extract the stereo camera parameters from the saved .npz file.

    Returns:
    - k1, d1: Camera matrix and distortion coefficients for camera 1
    - k2, d2: Camera matrix and distortion coefficients for camera 2
    - R, T: Rotation and translation between the cameras
    """
    loaded_data = np.load(file_name)
    return loaded_data['k1'], loaded_data['d1'], loaded_data['k2'], loaded_data['d2'], loaded_data['R'], loaded_data['T']

def getStereoSingleCameraParameters(file_name):
    """
    Extract individual camera parameters from the saved .npz file.

    Returns:
    - k: Camera matrix
    - d: Distortion coefficients
    - r: Rotation vectors
    - t: Translation vectors
    """
    loaded_data = np.load(file_name)
    return loaded_data['k'], loaded_data['d'], loaded_data['r'], loaded_data['t']

def main():
    # Initialize cameras
    cam1 = jetCam.jetsonCam()
    cam2 = jetCam.jetsonCam()

    # Open cameras with specified settings
    cam1.open(sensor_id=1,
              sensor_mode=3,
              flip_method=0,
              display_height=540,
              display_width=960)
    cam2.open(sensor_id=0,
              sensor_mode=3,
              flip_method=0,
              display_height=540,
              display_width=960)

    # Start camera streams
    cam1.start()
    cam2.start()
    
    try:
        # Perform stereo calibration
        stereoCalibrateCamera(cam1, cam2, 'jetson_stereo_8MP', 
                              chessboard_box_size=47, 
                              chessboard_grid_size=(7,7))
        
        # Load calibration parameters
        lod_data = getStereoCameraParameters('jetson_stereo_8MP.npz')
        lod_datac1 = getStereoSingleCameraParameters('jetson_stereo_8MPc1.npz')
        lod_datac2 = getStereoSingleCameraParameters('jetson_stereo_8MPc2.npz')

        # Extract parameters
        camera_matrix_left = lod_data[0]
        dist_coeffs_left =  lod_data[1]
        camera_matrix_right =  lod_data[2]
        dist_coeffs_right =  lod_data[3]
        R =  lod_data[4]
        T =  lod_data[5]

        # Print camera matrices and distortion coefficients
        print("\n--- Calibration Results ---")
        print("Camera Matrix Left:\n", camera_matrix_left)
        print("Camera Matrix Right:\n", camera_matrix_right)
        print("Distortion Coefficients Left:\n", dist_coeffs_left)
        print("Distortion Coefficients Right:\n", dist_coeffs_right)
        print("Rotation Matrix (R):\n", R)
        print("Translation Vector (T):\n", T)
        print("----------------------------\n")
        
    except Exception as e:
        print(f"An error occurred during calibration: {e}")
    finally:
        # Release resources
        cv2.destroyAllWindows()
        cam1.stop()
        cam2.stop()
        cam1.release()
        cam2.release()
        print("Camera resources released.")

if __name__ == "__main__":
    main()

