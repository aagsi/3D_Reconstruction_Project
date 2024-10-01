import cv2
import numpy as np
import threading
import os
import glob
from PyQt5 import QtWidgets, QtGui, QtCore
import sys

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
                            chessboard_grid_size=(7, 7), number_of_frames=50,
                            save_images=False, images_folder="calibration_images",
                            load_images=False):
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

    if not load_images:
        print("\nControls during image capture:")
        print("Press 'c' to capture an image pair.")
        print("Press 'x' to abort the calibration process.\n")

        if save_images and not os.path.exists(images_folder):
            os.makedirs(images_folder)

        img_count = 0
        while img_count < number_of_frames:
            try:
                ret_left, frame_left = cam_left.read()
                ret_right, frame_right = cam_right.read()
            except Exception as e:
                print(f"Error reading from cameras: {e}")
                break

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

                if save_images:
                    cv2.imwrite(os.path.join(images_folder, f"left_{img_count}.png"), frame_left)
                    cv2.imwrite(os.path.join(images_folder, f"right_{img_count}.png"), frame_right)
            elif key == ord('x'):
                cv2.destroyAllWindows()
                print('Capture terminated. Aborting calibration.')
                return
        cv2.destroyAllWindows()

    else:
        # Load images from the images_folder
        print("Loading images from folder:", images_folder)
        left_images = sorted(glob.glob(os.path.join(images_folder, "left_*.png")))
        right_images = sorted(glob.glob(os.path.join(images_folder, "right_*.png")))

        if len(left_images) != len(right_images):
            print("Number of left and right images does not match.")
            return

        if len(left_images) == 0:
            print("No images found in the folder.")
            return

        for left_image_path, right_image_path in zip(left_images, right_images):
            img_left = cv2.imread(left_image_path)
            img_right = cv2.imread(right_image_path)

            if img_left is None or img_right is None:
                print(f"Failed to load images: {left_image_path}, {right_image_path}")
                continue

            imgpoints_left.append(img_left)
            imgpoints_right.append(img_right)

        img_count = len(imgpoints_left)
        print(f"Loaded {img_count} image pairs.")

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

            # Visualization: Draw and display the corners
            img_left_corners = cv2.drawChessboardCorners(img_left.copy(), CHECKERBOARD, corners_left, ret_left)
            img_right_corners = cv2.drawChessboardCorners(img_right.copy(), CHECKERBOARD, corners_right, ret_right)
            concatenated = cv2.hconcat([img_left_corners, img_right_corners])
            concatenated_resized = cv2.resize(concatenated, None, fx=0.5, fy=0.5)
            cv2.imshow("Corners", concatenated_resized)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
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

# ----------------------- GUI Implementation ----------------------- #

class CalibrationGUI(QtWidgets.QMainWindow):
    def __init__(self, cam_left, cam_right):
        super().__init__()
        self.cam_left = cam_left
        self.cam_right = cam_right
        self.initUI()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)
        self.captured_pairs = []
        self.images_folder = "calibration_images"
        self.save_images = False

    def initUI(self):
        self.setWindowTitle('Stereo Calibration GUI')
        self.setGeometry(100, 100, 1200, 600)

        # Create central widget
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        # Create layout
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Create image display
        self.label_display = QtWidgets.QLabel()
        self.layout.addWidget(self.label_display)

        # Create control buttons
        self.btn_capture = QtWidgets.QPushButton('Capture Image Pair')
        self.btn_calibrate = QtWidgets.QPushButton('Start Calibration')
        self.btn_load_images = QtWidgets.QPushButton('Load Images')
        self.btn_save_images = QtWidgets.QPushButton('Toggle Save Images (Off)')

        self.btn_capture.clicked.connect(self.capture_images)
        self.btn_calibrate.clicked.connect(self.start_calibration)
        self.btn_load_images.clicked.connect(self.load_images)
        self.btn_save_images.clicked.connect(self.toggle_save_images)

        self.layout.addWidget(self.btn_capture)
        self.layout.addWidget(self.btn_calibrate)
        self.layout.addWidget(self.btn_load_images)
        self.layout.addWidget(self.btn_save_images)

        # Status bar
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.captured_count = 0

    def update_frames(self):
        ret_left, frame_left = self.cam_left.read()
        ret_right, frame_right = self.cam_right.read()

        if not ret_left or not ret_right:
            self.status_bar.showMessage("Failed to grab frames from cameras.")
            return

        # Resize images for display
        frame_left_resized = cv2.resize(frame_left, (480, 360))
        frame_right_resized = cv2.resize(frame_right, (480, 360))

        # Concatenate images horizontally
        concatenated = cv2.hconcat([frame_left_resized, frame_right_resized])

        # Convert image to Qt format
        qt_img = self.convert_cv_qt(concatenated)
        self.label_display.setPixmap(qt_img)

    def capture_images(self):
        ret_left, frame_left = self.cam_left.read()
        ret_right, frame_right = self.cam_right.read()

        if not ret_left or not ret_right:
            self.status_bar.showMessage("Failed to grab frames from cameras.")
            return

        self.captured_pairs.append((frame_left.copy(), frame_right.copy()))
        self.captured_count += 1
        self.status_bar.showMessage(f"Captured {self.captured_count} image pairs.")

        if self.save_images:
            if not os.path.exists(self.images_folder):
                os.makedirs(self.images_folder)
            cv2.imwrite(os.path.join(self.images_folder, f"left_{self.captured_count}.png"), frame_left)
            cv2.imwrite(os.path.join(self.images_folder, f"right_{self.captured_count}.png"), frame_right)

    def start_calibration(self):
        if len(self.captured_pairs) == 0:
            self.status_bar.showMessage("No images captured for calibration.")
            return

        # Stop the camera updates
        self.timer.stop()

        imgpoints_left = [pair[0] for pair in self.captured_pairs]
        imgpoints_right = [pair[1] for pair in self.captured_pairs]

        # Save images if needed
        if self.save_images:
            for idx, (img_left, img_right) in enumerate(self.captured_pairs, 1):
                cv2.imwrite(os.path.join(self.images_folder, f"left_{idx}.png"), img_left)
                cv2.imwrite(os.path.join(self.images_folder, f"right_{idx}.png"), img_right)

        # Call the calibration function
        stereo_calibrate_camera(None, None, camera_name='jetson_stereo_8MP',
                                chessboard_box_size=47, chessboard_grid_size=(7, 7),
                                number_of_frames=len(self.captured_pairs),
                                save_images=False, images_folder=self.images_folder,
                                load_images=False)
        self.status_bar.showMessage("Calibration completed.")

        # Restart the camera updates
        self.timer.start(30)

    def load_images(self):
        options = QtWidgets.QFileDialog.Options()
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Calibration Images Folder", options=options)
        if folder:
            self.images_folder = folder
            self.status_bar.showMessage(f"Selected folder: {folder}")
            # Call the calibration function with load_images=True
            stereo_calibrate_camera(None, None, camera_name='jetson_stereo_8MP',
                                    chessboard_box_size=47, chessboard_grid_size=(7, 7),
                                    number_of_frames=50,
                                    save_images=False, images_folder=self.images_folder,
                                    load_images=True)
            self.status_bar.showMessage("Calibration completed using loaded images.")

    def toggle_save_images(self):
        self.save_images = not self.save_images
        status = "On" if self.save_images else "Off"
        self.btn_save_images.setText(f"Toggle Save Images ({status})")

    def convert_cv_qt(self, cv_img):
        """Convert from an OpenCV image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(960, 360, QtCore.Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)

# ----------------------- Main Function ----------------------- #

def main():
    # Initialize cameras
    try:
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
    except Exception as e:
        print(f"Error initializing cameras: {e}")
        return

    try:
        # Create an instance of the GUI application
        app = QtWidgets.QApplication(sys.argv)
        window = CalibrationGUI(cam_left, cam_right)
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"An error occurred during calibration: {e}")
    finally:
        # Release resources
        cam_left.stop()
        cam_right.stop()
        print("Camera resources released.")

if __name__ == "__main__":
    main()

