import cv2
import numpy as np
import os
import threading

class jetsonCam:

    def __init__ (self) :
        # Initialize instance variables
        self.video_capture = None
        self.frame = None
        self.grabbed = False
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    def open(self, **arg):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline(**arg), cv2.CAP_GSTREAMER
            )
            print(f"Camera opened with sensor_id={arg.get('sensor_id', 0)}")
        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            return
        # Grab the first frame to start the video capturing
        self.grabbed, self.frame = self.video_capture.read()

    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # Create a thread to read the camera image
        if self.video_capture is not None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
            print("Camera stream started.")
        return self

    def stop(self):
        self.running = False
        if self.read_thread is not None:
            self.read_thread.join()
            print("Camera stream stopped.")

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
            print("Camera released.")
        # Now kill the thread
        if self.read_thread is not None:
            self.read_thread.join()

# GStreamer pipeline function for capturing from the CSI camera
def gstreamer_pipeline(
    sensor_id=0,
    sensor_mode=3,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def stereoCalibrateCamera(camera_c1, camera_c2, camera_name, chessboard_box_size=47, 
                          chessboard_grid_size=(7,7), number_of_frames=50):
    """
    Perform stereo camera calibration with the given parameters.

    Parameters:
    - camera_c1, camera_c2: Camera objects.
    - camera_name: Base name for saving calibration files.
    - chessboard_box_size: Size of a chessboard square in mm (default: 47 mm).
    - chessboard_grid_size: Number of inner corners per a chessboard row and column (rows, columns) (default: (7,7)).
    - number_of_frames: Number of calibration frames to capture (default: 10).
    """

    # =============================================================================
    # Calibration Pattern Settings
    # =============================================================================

    # Define the dimensions of the checkerboard
    CHECKERBOARD = chessboard_grid_size

    # Chessboard square size in mm
    square_size = chessboard_box_size  

    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + 
                cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    # Flags for stereo calibration
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC

    # Prepare object points based on the calibration pattern
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0],
                           0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all images
    objpoints = []       # 3D points in real-world space
    imgpoints_c1 = []    # 2D points in image plane for camera 1
    imgpoints_c2 = []    # 2D points in image plane for camera 2

    # Lists to store captured images for calibration
    img_list_c1 = []
    img_list_c2 = []

    # =============================================================================
    # Capture Calibration Images
    # =============================================================================

    print("Calibration Setup:")
    print(f"Chessboard Grid Size (Inner Corners): {CHECKERBOARD[0]}x{CHECKERBOARD[1]}")
    print(f"Chessboard Square Size: {square_size} mm")
    print("\nControls during image capture:")
    print("Press 'c' to capture an image pair.")
    print("Press 'x' to abort the calibration process.\n")

    img_count = 0
    while img_count < number_of_frames:
        # Capture frames from both cameras
        dat1_rcved, img1 = camera_c1.read()  
        dat2_rcved, img2 = camera_c2.read()      
        
        if not dat1_rcved or not dat2_rcved:
            print("Failed to grab frames from cameras.")
            break
        
        # Use CUDA for image resizing
        try:
            # Upload images to GPU
            img1_gpu = cv2.cuda_GpuMat()
            img2_gpu = cv2.cuda_GpuMat()
            img1_gpu.upload(img1)
            img2_gpu.upload(img2)

            # Resize images
            new_size = (int(img1.shape[1]*0.6), int(img1.shape[0]*0.6))
            img1_r_gpu = cv2.cuda.resize(img1_gpu, new_size)
            img2_r_gpu = cv2.cuda.resize(img2_gpu, new_size)

            # Download images back to CPU
            img1_r = img1_r_gpu.download()
            img2_r = img2_r_gpu.download()
        except Exception as e:
            print(f"CUDA error during image resizing: {e}")
            # Fallback to CPU resizing
            img1_r = cv2.resize(img1, (0,0), fx=0.6, fy=0.6)
            img2_r = cv2.resize(img2, (0,0), fx=0.6, fy=0.6)
        
        # Concatenate images horizontally
        concatenated = cv2.hconcat([img1_r, img2_r])

        # Display captured count on the image
        display_img = concatenated.copy()
        cv2.putText(display_img, f"Captured: {img_count}/{number_of_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Camera Left - Camera Right", display_img)
        k = cv2.waitKey(10) & 0xFF

        if k == ord('c'):
            img_list_c1.append(img1)
            img_list_c2.append(img2)
            img_count += 1
            print(f"{img_count} image(s) captured.")
            
            # Optionally save captured images for inspection
            cv2.imwrite(f"captured_c1_image_{img_count}.png", img1)
            cv2.imwrite(f"captured_c2_image_{img_count}.png", img2)
        elif k == ord('x'):
            cv2.destroyAllWindows()
            print('Capture terminated. Aborting calibration.')
            return

    cv2.destroyAllWindows()

    # =============================================================================
    # Find Chessboard Corners and Collect Points
    # =============================================================================

    print("\nProcessing captured images to find chessboard corners...")
    for idx, (image1, image2) in enumerate(zip(img_list_c1, img_list_c2), start=1):
        # Grayscale conversion using CPU
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners in both images
        ret1, corners1 = cv2.findChessboardCorners(
            gray1, CHECKERBOARD, None)
        ret2, corners2 = cv2.findChessboardCorners(
            gray2, CHECKERBOARD, None)
      
        if ret1 and ret2:
            objpoints.append(objp)

            # Refine corner positions
            corners1 = cv2.cornerSubPix(
                gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(
                gray2, corners2, (11, 11), (-1, -1), criteria)

            imgpoints_c1.append(corners1)
            imgpoints_c2.append(corners2)

            # Draw and display the corners
            image1_drawn = cv2.drawChessboardCorners(
                image1.copy(), CHECKERBOARD, corners1, ret1)
            image2_drawn = cv2.drawChessboardCorners(
                image2.copy(), CHECKERBOARD, corners2, ret2)

            concatenated_corners = cv2.hconcat([
                cv2.resize(image1_drawn, (0,0), fx=0.6, fy=0.6),
                cv2.resize(image2_drawn, (0,0), fx=0.6, fy=0.6)
            ])
            cv2.imshow("Corners Detected - Left and Right", concatenated_corners)
            cv2.waitKey(100)
            print(f"Processed image pair {idx}.")
        else:
            print(f"Chessboard corners not found in image pair {idx}. Skipping.")
            # Optionally save problematic images
            cv2.imwrite(f"failed_c1_image_{idx}.png", image1)
            cv2.imwrite(f"failed_c2_image_{idx}.png", image2)

    cv2.destroyAllWindows()
    
    if not objpoints or not imgpoints_c1 or not imgpoints_c2:
        print("Insufficient valid image pairs for calibration.")
        return

    # =============================================================================
    # Camera Calibration
    # =============================================================================

    print("\nStarting camera calibration...")
    # Image resolution
    image_size = (img_list_c1[0].shape[1], img_list_c1[0].shape[0])

    # Calibrate individual cameras
    ret_1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(
        objpoints, imgpoints_c1, image_size, None, None)
    print("Camera 1 calibration complete.")

    ret_2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
        objpoints, imgpoints_c2, image_size, None, None)
    print("Camera 2 calibration complete.")

    # Save individual camera parameters
    np.savez(f"{camera_name}_c1.npz", mtx=mtx1, dist=dist1, rvecs=rvecs1, tvecs=tvecs1)
    np.savez(f"{camera_name}_c2.npz", mtx=mtx2, dist=dist2, rvecs=rvecs2, tvecs=tvecs2)
    print("Individual camera parameters saved.")

    # Stereo calibration to compute extrinsic parameters
    print("Starting stereo calibration...")
    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_c1, imgpoints_c2, mtx1, dist1, mtx2, dist2,
        image_size, criteria=criteria, flags=stereocalibration_flags)
    print("Stereo calibration complete.")

    # Compute rectification transforms
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx1, dist1, mtx2, dist2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY)
    print("Rectification transforms computed.")

    # =============================================================================
    # Compute Reprojection Error
    # =============================================================================

    print("Computing reprojection errors...")
    def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        mean_error = total_error / len(objpoints)
        return mean_error

    error_c1 = compute_reprojection_error(
        objpoints, imgpoints_c1, rvecs1, tvecs1, mtx1, dist1)
    error_c2 = compute_reprojection_error(
        objpoints, imgpoints_c2, rvecs2, tvecs2, mtx2, dist2)
    print(f"Mean Reprojection Error (Camera 1): {error_c1}")
    print(f"Mean Reprojection Error (Camera 2): {error_c2}")

    # =============================================================================
    # Save Calibration Parameters
    # =============================================================================

    # Save stereo calibration parameters
    np.savez(f"{camera_name}_stereo.npz",
             mtx1=mtx1, dist1=dist1, mtx2=mtx2, dist2=dist2,
             R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)
    print("Stereo calibration parameters saved.")

    # Write detailed calibration report
    with open(f"{camera_name}_calibration_report.txt", 'w') as f:
        f.write("===== Intrinsic Parameters (Camera 1) =====\n")
        f.write(f"Focal Lengths: fx={mtx1[0,0]}, fy={mtx1[1,1]}\n")
        f.write(f"Principal Point: cx={mtx1[0,2]}, cy={mtx1[1,2]}\n")
        f.write(f"Skew Coefficient: s={mtx1[0,1]}\n")
        f.write(f"Distortion Coefficients: {dist1.ravel()}\n\n")

        f.write("===== Intrinsic Parameters (Camera 2) =====\n")
        f.write(f"Focal Lengths: fx={mtx2[0,0]}, fy={mtx2[1,1]}\n")
        f.write(f"Principal Point: cx={mtx2[0,2]}, cy={mtx2[1,2]}\n")
        f.write(f"Skew Coefficient: s={mtx2[0,1]}\n")
        f.write(f"Distortion Coefficients: {dist2.ravel()}\n\n")

        f.write("===== Extrinsic Parameters =====\n")
        f.write(f"Rotation Matrix (R):\n{R}\n")
        f.write(f"Translation Vector (T):\n{T}\n")
        baseline = np.linalg.norm(T)
        f.write(f"Baseline (B): {baseline} mm\n\n")

        f.write("===== Essential and Fundamental Matrices =====\n")
        f.write(f"Essential Matrix (E):\n{E}\n\n")
        f.write(f"Fundamental Matrix (F):\n{F}\n\n")

        f.write("===== Rectification Parameters =====\n")
        f.write(f"Rectification Rotation Matrix (Camera 1, R1):\n{R1}\n")
        f.write(f"Rectification Rotation Matrix (Camera 2, R2):\n{R2}\n")
        f.write(f"Projection Matrix (Camera 1, P1):\n{P1}\n")
        f.write(f"Projection Matrix (Camera 2, P2):\n{P2}\n")
        f.write(f"Disparity-to-Depth Mapping Matrix (Q):\n{Q}\n\n")

        f.write("===== Reprojection Error Metrics =====\n")
        f.write(f"Mean Reprojection Error (Camera 1): {error_c1}\n")
        f.write(f"Mean Reprojection Error (Camera 2): {error_c2}\n")

    print("Stereo calibration completed and parameters saved.")

def getStereoCameraParameters(file_name):
    """
    Extract the stereo camera parameters from the saved .npz file.

    Returns:
    - mtx1, dist1: Camera matrix and distortion coefficients for camera 1.
    - mtx2, dist2: Camera matrix and distortion coefficients for camera 2.
    - R, T: Rotation and translation between the cameras.
    - E, F: Essential and Fundamental matrices.
    - R1, R2, P1, P2, Q: Rectification and projection parameters.
    """
    loaded_data = np.load(file_name)
    return (loaded_data['mtx1'], loaded_data['dist1'],
            loaded_data['mtx2'], loaded_data['dist2'],
            loaded_data['R'], loaded_data['T'],
            loaded_data['E'], loaded_data['F'],
            loaded_data['R1'], loaded_data['R2'],
            loaded_data['P1'], loaded_data['P2'],
            loaded_data['Q'])

def main():
    # =============================================================================
    # Initialize Cameras
    # =============================================================================

    print("Initializing cameras...")
    cam1 = jetsonCam()
    cam2 = jetsonCam()

    # Open cameras with specified settings
    print("Opening camera 1...")
    cam1.open(sensor_id=1,
              sensor_mode=3,
              flip_method=0,
              display_height=540,
              display_width=960)
    print("Opening camera 2...")
    cam2.open(sensor_id=0,
              sensor_mode=3,
              flip_method=0,
              display_height=540,
              display_width=960)

    # Start camera streams
    print("Starting camera 1 stream...")
    cam1.start()
    print("Starting camera 2 stream...")
    cam2.start()
    
    try:
        # Perform stereo calibration
        stereoCalibrateCamera(cam1, cam2, 'jetson_stereo_8MP', 
                              chessboard_box_size=47, 
                              chessboard_grid_size=(7,7))
        
        # Load calibration parameters
        params = getStereoCameraParameters('jetson_stereo_8MP_stereo.npz')

        # Extract parameters
        (mtx1, dist1, mtx2, dist2, R, T, E, F, R1, R2, P1, P2, Q) = params

        # Print calibration parameters
        print("\n--- Calibration Results ---")
        print("Camera Matrix 1 (Intrinsic Parameters):\n", mtx1)
        print("Distortion Coefficients 1:\n", dist1.ravel())
        print("Camera Matrix 2 (Intrinsic Parameters):\n", mtx2)
        print("Distortion Coefficients 2:\n", dist2.ravel())
        print("Rotation Matrix (R):\n", R)
        print("Translation Vector (T):\n", T)
        print("Essential Matrix (E):\n", E)
        print("Fundamental Matrix (F):\n", F)
        print("Rectification Rotation Matrix 1 (R1):\n", R1)
        print("Rectification Rotation Matrix 2 (R2):\n", R2)
        print("Projection Matrix 1 (P1):\n", P1)
        print("Projection Matrix 2 (P2):\n", P2)
        print("Disparity-to-Depth Mapping Matrix (Q):\n", Q)
        print("----------------------------\n")
        
    except Exception as e:
        print(f"An error occurred during calibration: {e}")
    finally:
        # Release resources
        cv2.destroyAllWindows()
        print("Stopping camera 1 stream...")
        cam1.stop()
        print("Stopping camera 2 stream...")
        cam2.stop()
        print("Releasing camera 1 resources...")
        cam1.release()
        print("Releasing camera 2 resources...")
        cam2.release()
        print("Camera resources released.")

if __name__ == "__main__":
    main()

