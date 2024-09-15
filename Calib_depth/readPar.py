import numpy as np
import os

def load_stereo_calibration(calibration_file):
    """
    Load stereo calibration parameters from a .npz file.

    Parameters:
        calibration_file (str): Path to the .npz calibration file.

    Returns:
        dict: A dictionary containing all calibration parameters.
    """
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Calibration file '{calibration_file}' does not exist.")

    calib_data = np.load(calibration_file)

    # Define the required parameters
    required_params = [
        'cameraMatrix1',
        'cameraMatrix2',
        'distCoeffs1',
        'distCoeffs2',
        'E',
        'F',
        'R',
        'T',
        'R1',
        'R2',
        'P1',
        'P2',
        'Q',
        'Baseline'
    ]

    # Initialize a dictionary to store the parameters
    calibration_params = {}

    # Check for the presence of each required parameter
    missing_params = []
    for param in required_params:
        if param in calib_data:
            calibration_params[param] = calib_data[param]
        else:
            missing_params.append(param)

    if missing_params:
        raise KeyError(f"The following required parameters are missing in '{calibration_file}': {missing_params}")

    calib_data.close()
    return calibration_params

def print_calibration_parameters(calibration_params):
    """
    Print the calibration parameters in a readable format.

    Parameters:
        calibration_params (dict): Dictionary containing calibration parameters.
    """
    print("\n=== Stereo Calibration Parameters ===\n")
    print("Left Camera Matrix (cameraMatrix1):")
    print(calibration_params['cameraMatrix1'])
    print("\nRight Camera Matrix (cameraMatrix2):")
    print(calibration_params['cameraMatrix2'])
    print("\nLeft Distortion Coefficients (distCoeffs1):")
    print(calibration_params['distCoeffs1'])
    print("\nRight Distortion Coefficients (distCoeffs2):")
    print(calibration_params['distCoeffs2'])
    print("\nEssential Matrix (E):")
    print(calibration_params['E'])
    print("\nFundamental Matrix (F):")
    print(calibration_params['F'])
    print("\nRotation Matrix (R):")
    print(calibration_params['R'])
    print("\nTranslation Vector (T):")
    print(calibration_params['T'])
    print("\nRectification Matrix R1:")
    print(calibration_params['R1'])
    print("\nRectification Matrix R2:")
    print(calibration_params['R2'])
    print("\nProjection Matrix P1:")
    print(calibration_params['P1'])
    print("\nProjection Matrix P2:")
    print(calibration_params['P2'])
    print("\nDisparity-to-Depth Mapping Matrix (Q):")
    print(calibration_params['Q'])
    print("\nBaseline:")
    print(calibration_params['Baseline'])
    print("\n=======================================\n")

def main():
    # Path to the calibration file
    calibration_file = 'jetson_stereo_8MP_calibration.npz'

    try:
        # Load calibration parameters
        calibration_params = load_stereo_calibration(calibration_file)
        print(f"Successfully loaded calibration parameters from '{calibration_file}'.")
        
        # Print the loaded parameters
        print_calibration_parameters(calibration_params)

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except KeyError as key_error:
        print(key_error)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

