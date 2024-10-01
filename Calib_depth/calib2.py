import cv2
import numpy as np
import threading

# ----------------------- Jetson Camera Implementation ----------------------- #

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
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

class JetsonCamera:
    def __init__(self, **kwargs):
        self.pipeline = gstreamer_pipeline(**kwargs)
        self.video_capture = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.video_capture.isOpened():
            raise RuntimeError("Could not open video device")
        self.grabbed, self.frame = self.video_capture.read()
        self.read_lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            grabbed, frame = self.video_capture.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            if not grabbed:
                break

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.grabbed else None
        return self.grabbed, frame

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        self.video_capture.release()

# ----------------------- Stereo Calibration Function ----------------------- #

def stereo_calibrate_camera(cam_left, cam_right, camera_name, chessboard_box_size=47,
                            chessboard_grid_size=(7, 7), number_of_frames=50):
    """
    Perform stereo camera calibration with the given parameters.
    """
    # Calibration pattern settings
    CHECKERBOARD = chessboard_grid_size
    square_size = chessboard_box_size

    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    # Prepare object points
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real-world space
    imgpoints_left = []  # 2D points in image plane for left camera
    imgpoints_right = []  # 2D points in image plane for right camera

    # Start capturing calibration images
    print("Calibration Setup:")
    print(f"Chessboard Grid Size (Inner Corners): {CHECKERBOARD[0]}x{CHECKERBOARD[1]}")
    print(f"Chessboard Square Size: {square_size} mm")
    print("\nControls during image capture:")
    print("Press 'c' to capture an image pair.")
    print("Press 'x' to abort the calibration process.\n")

    img_count = 0
    while img_count < number_of_frames:
        ret_left, frame_left = cam_left.read()
        ret_right, frame_right = cam_right.read()

        if not ret_left or not ret_right:
            print("Failed to grab frames from cameras.")
            break

        # Resize images for display
        frame_left_resized = cv2.resize(frame_left, None, fx=0.6, fy=0.6)
        frame_right_resized = cv2.resize(frame_right, None, fx=0.6, fy=0.6)

        # Concatenate images horizontally
        concatenated = cv2.hconcat([frame_left_resized, frame_right_resized])

        # Display captured count on the image
        cv2.putText(concatenated, f"Captured: {img_count}/{number_of_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Camera Left - Camera Right", concatenated)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('c'):
            img_count += 1
            imgpoints_left.append(frame_left)
            imgpoints_right.append(frame_right)
            print(f"{img_count} image(s) captured.")
        elif key == ord('x'):
            cv2.destroyAllWindows()
            print('Capture terminated. Aborting calibration.')
            return
    cv2.destroyAllWindows()

    if img_count < number_of_frames:
        print("Insufficient images captured for calibration.")
        return

    # Find chessboard corners and collect points
    valid_pairs = 0
    indices_to_remove = []
    for idx, (img_left, img_right) in enumerate(zip(imgpoints_left, imgpoints_right)):
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

        if ret_left and ret_right:
            objpoints.append(objp)

            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

            imgpoints_left[idx] = corners_left
            imgpoints_right[idx] = corners_right

            valid_pairs += 1
            print(f"Chessboard corners found in image pair {idx + 1}.")
        else:
            print(f"Chessboard corners not found in image pair {idx + 1}. Skipping.")
            indices_to_remove.append(idx)

    # Remove invalid image pairs
    for idx in sorted(indices_to_remove, reverse=True):
        del imgpoints_left[idx]
        del imgpoints_right[idx]

    if valid_pairs < 10:
        print("Not enough valid image pairs for calibration.")
        return

    # Camera calibration
    image_size = (img_left.shape[1], img_left.shape[0])

    # Calibrate individual cameras
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, image_size, None, None
    )
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, image_size, None, None
    )

    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left, mtx_right, dist_right,
        image_size, criteria=criteria_stereo, flags=flags
    )

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_left, dist_left, mtx_right, dist_right, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY
    )

    # Save parameters
    np.savez(f"{camera_name}_stereo.npz",
             mtx1=mtx_left, dist1=dist_left, mtx2=mtx_right, dist2=dist_right,
             R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

    # Compute reprojection error
    def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        mean_error = total_error / len(objpoints)
        return mean_error

    error_left = compute_reprojection_error(objpoints, imgpoints_left, rvecs_left, tvecs_left, mtx_left, dist_left)
    error_right = compute_reprojection_error(objpoints, imgpoints_right, rvecs_right, tvecs_right, mtx_right, dist_right)

    # Save calibration report
    with open(f"{camera_name}_calibration_report.txt", 'w') as f:
        f.write("===== Calibration Report =====\n")
        f.write(f"Mean Reprojection Error (Left Camera): {error_left}\n")
        f.write(f"Mean Reprojection Error (Right Camera): {error_right}\n")
        f.write("\nIntrinsic Parameters Left Camera:\n")
        f.write(f"{mtx_left}\n")
        f.write("\nIntrinsic Parameters Right Camera:\n")
        f.write(f"{mtx_right}\n")
        f.write("\nDistortion Coefficients Left Camera:\n")
        f.write(f"{dist_left.ravel()}\n")
        f.write("\nDistortion Coefficients Right Camera:\n")
        f.write(f"{dist_right.ravel()}\n")
        f.write("\nRotation Matrix (R):\n")
        f.write(f"{R}\n")
        f.write("\nTranslation Vector (T):\n")
        f.write(f"{T}\n")
        f.write("\nEssential Matrix (E):\n")
        f.write(f"{E}\n")
        f.write("\nFundamental Matrix (F):\n")
        f.write(f"{F}\n")
        f.write("\nRectification Matrices (R1, R2):\n")
        f.write(f"{R1}\n")
        f.write(f"{R2}\n")
        f.write("\nProjection Matrices (P1, P2):\n")
        f.write(f"{P1}\n")
        f.write(f"{P2}\n")
        f.write("\nDisparity-to-Depth Mapping Matrix (Q):\n")
        f.write(f"{Q}\n")

    print("Stereo calibration completed and parameters saved.")
    print(f"Mean Reprojection Error (Left Camera): {error_left}")
    print(f"Mean Reprojection Error (Right Camera): {error_right}")

# ----------------------- Main Function ----------------------- #

def main():
    # Initialize cameras
    cam_left = JetsonCamera(
        sensor_id=1,
        sensor_mode=3,
        flip_method=0,
        display_height=540,
        display_width=960
    )
    cam_right = JetsonCamera(
        sensor_id=0,
        sensor_mode=3,
        flip_method=0,
        display_height=540,
        display_width=960
    )
    cam_left.start()
    cam_right.start()

    try:
        stereo_calibrate_camera(
            cam_left,
            cam_right,
            camera_name='jetson_stereo_8MP',
            chessboard_box_size=47,
            chessboard_grid_size=(7, 7),
            number_of_frames=50
        )
    except Exception as e:
        print(f"An error occurred during calibration: {e}")
    finally:
        # Release resources
        cam_left.stop()
        cam_right.stop()
        print("Camera resources released.")

if __name__ == "__main__":
    main()

