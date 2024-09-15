import numpy as np

def inspect_calibration_file(file_path):
    if not os.path.exists(file_path):
        print(f"Calibration file '{file_path}' does not exist.")
        return
    data = np.load(file_path)
    print(f"Contents of '{file_path}':")
    for key in data.files:
        print(f" - {key}")
    print("\n")

if __name__ == "__main__":
    import os
    calibration_files = [
        'jetson_stereo_8MP_calibration.npz',
        'jetson_stereo_8MPc1.npz',
        'jetson_stereo_8MPc2.npz'
    ]
    for file in calibration_files:
        inspect_calibration_file(file)

