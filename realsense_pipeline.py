import pyrealsense2.pyrealsense2 as rs

class RealSensePipeline:
    def __init__(self):
        """
        Initializes the RealSense camera pipeline.
        """
        self.pipeline = None

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

