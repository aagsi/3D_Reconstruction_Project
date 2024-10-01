import cv2
import numpy as np
import threading
import os
import glob
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

# ----------------------- Camera Class with Threading ----------------------- #

class Camera:
    """
    Camera class that captures frames using threading to prevent lag.
    """
    def __init__(self, sensor_id=0, capture_width=1280, capture_height=720,
                 display_width=1280, display_height=720, framerate=30, flip_method=0):
        self.sensor_id = sensor_id
        self.gstreamer_pipeline_string = self.gstreamer_pipeline(
            sensor_id=sensor_id,
            capture_width=capture_width,
            capture_height=capture_height,
            display_width=display_width,
            display_height=display_height,
            framerate=framerate,
            flip_method=flip_method
        )
        self.video_capture = cv2.VideoCapture(
            self.gstreamer_pipeline_string, cv2.CAP_GSTREAMER)
        if not self.video_capture.isOpened():
            raise RuntimeError(f"Could not open video device {sensor_id}")
        self.grabbed, self.frame = self.video_capture.read()
        self.read_lock = threading.Lock()
        self.running = False
        self.thread = None

    def gstreamer_pipeline(
        self,
        sensor_id=0,
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=30,
        flip_method=0,
    ):
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
            f"framerate=(fraction){framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! appsink"
        )

    def start(self):
        if self.running:
            print(f'Camera {self.sensor_id} is already running')
            return
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            grabbed, frame = self.video_capture.read()
            if not grabbed:
                print(f"Camera {self.sensor_id}: Failed to grab frame")
                break
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
        self.running = False

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

def stereo_calibrate_camera(imgpoints_left, imgpoints_right, camera_name, chessboard_box_size=47,
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

    img_count = len(imgpoints_left)

    if img_count < number_of_frames:
        print("Insufficient images captured for calibration.")
        return

    # Find chessboard corners and collect points
    valid_pairs = 0
    indices_to_remove = []
    for idx, (img_left, img_right) in enumerate(zip(imgpoints_left, imgpoints_right)):
        # Convert to grayscale using CUDA if available
        try:
            cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        except AttributeError:
            cuda_available = False

        if cuda_available:
            # Use CUDA for grayscale conversion
            img_left_gpu = cv2.cuda_GpuMat()
            img_right_gpu = cv2.cuda_GpuMat()
            img_left_gpu.upload(img_left)
            img_right_gpu.upload(img_right)
            gray_left_gpu = cv2.cuda.cvtColor(img_left_gpu, cv2.COLOR_BGR2GRAY)
            gray_right_gpu = cv2.cuda.cvtColor(img_right_gpu, cv2.COLOR_BGR2GRAY)
            gray_left = gray_left_gpu.download()
            gray_right = gray_right_gpu.download()
        else:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

        if ret_left and ret_right:
            objpoints.append(objp)

            # Refine corner positions
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
    image_size = (imgpoints_left[0].shape[1], imgpoints_left[0].shape[0])

    # Calibrate individual cameras
    print("Calibrating left camera...")
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, image_size, None, None
    )
    print("Calibrating right camera...")
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, image_size, None, None
    )

    # Stereo calibration
    print("Performing stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left, mtx_right, dist_right,
        image_size, criteria=criteria_stereo, flags=flags
    )

    # Stereo rectification
    print("Computing rectification transforms...")
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
        f.write(f"\nDisparity-to-Depth Mapping Matrix (Q):\n")
        f.write(f"{Q}\n")

    print("Stereo calibration completed and parameters saved.")
    print(f"Mean Reprojection Error (Left Camera): {error_left}")
    print(f"Mean Reprojection Error (Right Camera): {error_right}")

# ----------------------- GUI Implementation using tkinter ----------------------- #

class CalibrationGUI:
    """
    GUI for capturing images and performing stereo calibration.
    """
    def __init__(self, root, cam_left, cam_right):
        self.root = root
        self.cam_left = cam_left
        self.cam_right = cam_right
        self.captured_pairs = []
        self.images_folder = "calibration_images"
        self.save_images = False
        self.initUI()
        self.update_frames_id = None
        self.update_frames()

    def initUI(self):
        self.root.title('Stereo Calibration GUI')

        # Create image display label
        self.label_display = Label(self.root)
        self.label_display.pack()

        # Create control buttons
        self.btn_capture = Button(self.root, text='Capture Image Pair', command=self.capture_images)
        self.btn_calibrate = Button(self.root, text='Start Calibration', command=self.start_calibration)
        self.btn_load_images = Button(self.root, text='Load Images', command=self.load_images)
        self.btn_save_images = Button(self.root, text='Toggle Save Images (Off)', command=self.toggle_save_images)
        self.btn_exit = Button(self.root, text='Exit', command=self.on_closing)

        self.btn_capture.pack()
        self.btn_calibrate.pack()
        self.btn_load_images.pack()
        self.btn_save_images.pack()
        self.btn_exit.pack()

        # Status label
        self.status_label = Label(self.root, text='Status: Ready')
        self.status_label.pack()

        self.captured_count = 0

        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_frames(self):
        ret_left, frame_left = self.cam_left.read()
        ret_right, frame_right = self.cam_right.read()

        if not ret_left or not ret_right:
            self.status_label.config(text="Failed to grab frames from cameras.")
            return

        # Resize images for display
        frame_left_resized = cv2.resize(frame_left, (480, 360))
        frame_right_resized = cv2.resize(frame_right, (480, 360))

        # Concatenate images horizontally
        concatenated = cv2.hconcat([frame_left_resized, frame_right_resized])

        # Convert image to PIL format
        image = cv2.cvtColor(concatenated, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.label_display.imgtk = image
        self.label_display.configure(image=image)

        # Schedule next frame update
        self.update_frames_id = self.root.after(30, self.update_frames)

    def capture_images(self):
        ret_left, frame_left = self.cam_left.read()
        ret_right, frame_right = self.cam_right.read()

        if not ret_left or not ret_right:
            self.status_label.config(text="Failed to grab frames from cameras.")
            return

        self.captured_pairs.append((frame_left.copy(), frame_right.copy()))
        self.captured_count += 1
        self.status_label.config(text=f"Captured {self.captured_count} image pairs.")

        if self.save_images:
            if not os.path.exists(self.images_folder):
                os.makedirs(self.images_folder)
            cv2.imwrite(os.path.join(self.images_folder, f"left_{self.captured_count}.png"), frame_left)
            cv2.imwrite(os.path.join(self.images_folder, f"right_{self.captured_count}.png"), frame_right)

    def start_calibration(self):
        if len(self.captured_pairs) == 0:
            self.status_label.config(text="No images captured for calibration.")
            return

        # Stop the frame updates
        if self.update_frames_id is not None:
            self.root.after_cancel(self.update_frames_id)

        imgpoints_left = [pair[0] for pair in self.captured_pairs]
        imgpoints_right = [pair[1] for pair in self.captured_pairs]

        # Save images if needed
        if self.save_images:
            if not os.path.exists(self.images_folder):
                os.makedirs(self.images_folder)
            for idx, (img_left, img_right) in enumerate(self.captured_pairs, 1):
                cv2.imwrite(os.path.join(self.images_folder, f"left_{idx}.png"), img_left)
                cv2.imwrite(os.path.join(self.images_folder, f"right_{idx}.png"), img_right)

        # Call the calibration function
        try:
            stereo_calibrate_camera(imgpoints_left, imgpoints_right, camera_name='jetson_stereo_8MP',
                                    chessboard_box_size=47, chessboard_grid_size=(7, 7),
                                    number_of_frames=len(self.captured_pairs))
            self.status_label.config(text="Calibration completed.")
        except Exception as e:
            print(f"An error occurred during calibration: {e}")
            self.status_label.config(text="Calibration failed. Check console for details.")

        # Restart the frame updates
        self.update_frames()

    def load_images(self):
        folder = filedialog.askdirectory(title="Select Calibration Images Folder")
        if folder:
            self.images_folder = folder
            self.status_label.config(text=f"Selected folder: {folder}")

            # Load images from folder
            left_images = sorted(glob.glob(os.path.join(self.images_folder, "left_*.png")))
            right_images = sorted(glob.glob(os.path.join(self.images_folder, "right_*.png")))

            if len(left_images) != len(right_images):
                self.status_label.config(text="Number of left and right images does not match.")
                return

            if len(left_images) == 0:
                self.status_label.config(text="No images found in the folder.")
                return

            imgpoints_left = []
            imgpoints_right = []

            for left_image_path, right_image_path in zip(left_images, right_images):
                img_left = cv2.imread(left_image_path)
                img_right = cv2.imread(right_image_path)

                if img_left is None or img_right is None:
                    print(f"Failed to load images: {left_image_path}, {right_image_path}")
                    continue

                imgpoints_left.append(img_left)
                imgpoints_right.append(img_right)

            # Call the calibration function
            try:
                stereo_calibrate_camera(imgpoints_left, imgpoints_right, camera_name='jetson_stereo_8MP',
                                        chessboard_box_size=47, chessboard_grid_size=(7, 7),
                                        number_of_frames=len(imgpoints_left))
                self.status_label.config(text="Calibration completed using loaded images.")
            except Exception as e:
                print(f"An error occurred during calibration: {e}")
                self.status_label.config(text="Calibration failed. Check console for details.")

    def toggle_save_images(self):
        self.save_images = not self.save_images
        status = "On" if self.save_images else "Off"
        self.btn_save_images.config(text=f"Toggle Save Images ({status})")

    def on_closing(self):
        # Stop the cameras
        self.cam_left.stop()
        self.cam_right.stop()
        self.root.destroy()

# ----------------------- Main Function ----------------------- #

def main():
    # Initialize cameras
    try:
        cam_left = Camera(
            sensor_id=1,
            capture_width=1280,
            capture_height=720,
            display_width=960,
            display_height=540,
            framerate=30,
            flip_method=0
        )
        cam_right = Camera(
            sensor_id=0,
            capture_width=1280,
            capture_height=720,
            display_width=960,
            display_height=540,
            framerate=30,
            flip_method=0
        )
        cam_left.start()
        cam_right.start()
    except Exception as e:
        print(f"Error initializing cameras: {e}")
        return

    try:
        # Create an instance of the GUI application
        root = Tk()
        app = CalibrationGUI(root, cam_left, cam_right)
        root.mainloop()
    except Exception as e:
        print(f"An error occurred during calibration: {e}")
    finally:
        # Release resources
        cam_left.stop()
        cam_right.stop()
        print("Camera resources released.")

if __name__ == "__main__":
    main()

