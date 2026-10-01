import cv2
import numpy as np
import os
import argparse
from picamera2 import Picamera2
#sudo apt install python3-picamera2


def calibrate_camera(output_file='data/camera_calibration.yml', grid_size=(7, 7), square_size=25.00, width=1920, height=1080):
    """
    Calibrates the camera using a chessboard pattern.

    :param output_file: Path to save the calibration file.
    :param grid_size: Number of inner corners per a chessboard row and column.
    :param square_size: Size of a square in mm.
    """
    if not os.path.exists('data'):
        os.makedirs('data')

    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.


    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    print("Press 'c' to capture a frame for calibration, and 'q' to quit.")

    while True:
        frame = picam2.capture_array()

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        if ret:
            cv2.drawChessboardCorners(frame, grid_size, corners, ret)

        cv2.imshow('Calibration', frame)

        key = cv2.waitKey(1) & 0xFF
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            print("Captured frame for calibration.")
        elif key == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

    if len(objpoints) > 0:
        print("Calibrating camera...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            try:
                # Open file storage for writing
                fs = cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE)
                if not fs.isOpened():
                    raise IOError(f"Failed to open file: {output_file}")

                # Write camera matrix and distortion coefficients
                fs.write("camera_matrix", mtx)
                fs.write("dist_coeffs", dist)
                fs.release()

                print(f"Calibration saved to {output_file}")
            except Exception as e:
                print(f"Failed to write calibration data: {e}")
        else:
            print("Calibration failed.")
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print(f"Mean reprojection error: {mean_error / len(objpoints)}")
    else:
        print("No frames captured for calibration.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Calibrate the camera using a chessboard pattern.")
    
    # Positional argument for calibration file
    parser.add_argument("--calibration_file",type=str, default="data/camera_calibration.yml", help="Path to the calibration file.")
    
    # Optional arguments
    parser.add_argument("--grid_size", type=int, nargs=2, default=(7, 7), help="Grid size as two integers (default: (7, 7)).")
    parser.add_argument("--square_size", type=float, default=25.0, help="Length of a square in mm (default: 25.00).")
    parser.add_argument("--height", type=int, default=800, help="Resolution height (default: 800)")
    parser.add_argument("--width", type=int, default=1280, help="Resolution width (default: 1280)")

  
    args = parser.parse_args()
    calibrate_camera(args.calibration_file, args.grid_size, args.square_size, args.width, args.height)
