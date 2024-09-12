import pyrealsense2.pyrealsense2 as rs
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

class RealSensePipeline:
    def __init__(self):
        """
        Initializes the RealSense camera pipeline.
        """
        self.pipeline = None
        self.depth_frame = None
        self.color_frame = None

    def start_pipeline(self):
        """
        Starts the RealSense pipeline for capturing color and depth streams.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

        try:
            self.pipeline.start(config)
        except RuntimeError as e:
            print(f"Failed to start pipeline: {e}")
            device = self.pipeline.get_active_profile().get_device()
            device.hardware_reset()
            self.pipeline.stop()
            exit(1)

    def stop_pipeline(self):
        """
        Stops the RealSense pipeline.
        """
        self.pipeline.stop()

    def get_frames(self):
        """
        Captures a single frame of color and depth data from the RealSense camera.
        
        :return: A tuple of (depth_frame, color_frame) as numpy arrays.
        """
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to capture frames")

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image

    def transfer_to_cuda(self, np_array):
        """
        Transfers a numpy array to CUDA device memory.
        
        :param np_array: The numpy array to transfer.
        :return: The CUDA memory object.
        """
        # Allocate memory on the CUDA device
        cuda_array = cuda.mem_alloc(np_array.nbytes)
        
        # Copy the numpy array to CUDA device memory
        cuda.memcpy_htod(cuda_array, np_array)
        
        return cuda_array

    def process_frames_with_cuda(self):
        """
        Example method to demonstrate CUDA processing.
        """
        depth_image, color_image = self.get_frames()

        # Transfer images to CUDA
        depth_cuda = self.transfer_to_cuda(depth_image)
        color_cuda = self.transfer_to_cuda(color_image)

        # Perform CUDA processing (example)
        # You need to implement actual CUDA kernel operations here
        print("Depth and color images have been transferred to CUDA.")

