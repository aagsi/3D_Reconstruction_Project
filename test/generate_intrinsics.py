import pyrealsense2 as rs
import json
import os

def get_intrinsics(json_path):
    # Configure and start the pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable the color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
    except RuntimeError as e:
        print(f"Failed to start RealSense pipeline: {e}")
        return
    
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("No color frame captured.")
        pipeline.stop()
        return
    
    # Get intrinsics
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    
    intrinsics = {
        'fx': intr.fx,
        'fy': intr.fy,
        'ppx': intr.ppx,
        'ppy': intr.ppy,
        'width': intr.width,
        'height': intr.height
    }
    
    # Save intrinsics to JSON
    with open(json_path, 'w') as f:
        json.dump(intrinsics, f, indent=4)
    
    print(f"Camera intrinsics saved to {json_path}")
    
    # Stop the pipeline
    pipeline.stop()

if __name__ == "__main__":
    # Define the path where the JSON will be saved
    json_path = 'dataset/realsense/camera_intrinsic.json'
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Generate and save intrinsics
    get_intrinsics(json_path)

