#!/usr/bin/env python3
# detect_aruco.py

import sys
import os
import cv2
import numpy as np
import threading
import argparse
import time
from flask import Flask, jsonify, Response
from flask_cors import CORS
from scipy.spatial.transform import Rotation as R
from picamera2 import Picamera2
from math import cos, sin, atan2
import logging
#sudo apt install python3-picamera2

deg_to_rad = np.pi / 180.0
rad_to_deg = 180.0 / np.pi

position_data = {"x": 0.0, "y" : 0.0, "a": 0.0, "z": 0.0}
status = False # True = running, False = stopped
sucessFrames = 0
failedFrames = 0
latest_frame = None
latest_frame_lock = threading.Lock()
object_positions = {}
object_positions_lock = threading.Lock()
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/position', methods=['GET'])
def get_position():
    global status, sucessFrames, failedFrames, position_data
    if (status):
        return jsonify({"position":position_data, "sucessFrames": sucessFrames, "totalFrames": (sucessFrames+failedFrames), "failedFrames": failedFrames})
    else:
        return jsonify({"message": "Camera is not running"})

@app.route('/start', methods=['GET'])
def api_start():
    global status, sucessFrames, failedFrames, position_data
    status = True
    sucessFrames = 0
    failedFrames = 0
    position_data = {"x": 0.0, "y" : 0.0, "a": 0.0, "z": 0.0}
    with object_positions_lock:
        object_positions.clear()
    return jsonify({"message": "Starting Camera"})

@app.route('/stop', methods=['GET'])
def api_stop():
    global status
    status = False
    return jsonify({"message": "Stopped Camera"})


@app.route('/reset_tracking', methods=['GET'])
def api_reset_tracking():
    global sucessFrames, failedFrames, position_data
    sucessFrames = 0
    failedFrames = 0
    position_data = {"x": 0.0, "y" : 0.0, "a": 0.0, "z": 0.0}
    with object_positions_lock:
        object_positions.clear()
    return jsonify({"message": "Successfully resetted tracking"})

@app.route('/objects', methods=['GET'])
def get_objects():
    with object_positions_lock:
        return jsonify({"objects": object_positions})

@app.route('/preview', methods=['GET'])
def get_preview():
    global latest_frame
    with latest_frame_lock:
        if latest_frame is None:
            return jsonify({"message": "No preview available"}), 503
        frame = latest_frame.copy()

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return jsonify({"message": "Failed to encode preview"}), 500

    return Response(buffer.tobytes(), mimetype="image/jpeg")

def add_average_position(new_pos):
    global sucessFrames, position_data
    if sucessFrames == 0 or position_data["a"] is None:
        position_data = new_pos
    else:
        new_ratio = 1.0 / (sucessFrames + 1.0)
        old_ratio = 1.0 - new_ratio
        position_data["x"] = position_data["x"] * old_ratio + new_pos["x"] * new_ratio
        position_data["y"] = position_data["y"] * old_ratio + new_pos["y"] * new_ratio
        position_data["z"] = position_data["z"] * old_ratio + new_pos["z"] * new_ratio

        # Average the angle
        y1 = sin(position_data["a"] * deg_to_rad) * old_ratio
        x1 = cos(position_data["a"] * deg_to_rad) * old_ratio
        y2 = sin(new_pos["a"] * deg_to_rad) * new_ratio
        x2 = cos(new_pos["a"] * deg_to_rad) * new_ratio
        position_data["a"] = atan2(y1+y2, x1+x2) * rad_to_deg


def detect_aruco(calib_file, marker_info, headless=False, showRejected=False, width=1280, height=800):
    """
    Detects ArUco markers and calculates the camera's position relative to the center.
    :param calib_file: Path to the camera calibration file.
    :param marker_info: Dictionary with marker IDs, sizes, and positions.
    """
    if not os.path.exists(calib_file):
        print(f"Calibration file {calib_file} not found.")
        sys.exit(1)

    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
    mtx = fs.getNode('camera_matrix').mat()
    dist = fs.getNode('dist_coeffs').mat()
    fs.release()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # General detection parameters
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10
    aruco_params.adaptiveThreshConstant = 7

    # Marker perimeter and corner parameters
    aruco_params.minMarkerPerimeterRate = 0.03
    aruco_params.maxMarkerPerimeterRate = 4.0
    aruco_params.polygonalApproxAccuracyRate = 0.03
    aruco_params.minCornerDistanceRate = 0.05
    aruco_params.minDistanceToBorder = 3
    aruco_params.minMarkerDistanceRate = 0.05

    # Corner refinement parameters (enable subpixel refinement)
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    aruco_params.cornerRefinementWinSize = 5
    aruco_params.relativeCornerRefinmentWinSize = 0.01  # Relative refinement window size
    aruco_params.cornerRefinementMaxIterations = 30
    aruco_params.cornerRefinementMinAccuracy = 0.1

    # Marker border and perspective parameters
    aruco_params.markerBorderBits = 1
    aruco_params.perspectiveRemovePixelPerCell = 4
    aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
    aruco_params.maxErroneousBitsInBorderRate = 0.35
    aruco_params.minOtsuStdDev = 5.0
    aruco_params.errorCorrectionRate = 0.6

    # Inverted marker detection and ArUco3 support
    aruco_params.detectInvertedMarker = False
    aruco_params.useAruco3Detection = True



    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    print("Starting ArUco detection. Press 'q' to quit.")

    global status, sucessFrames, failedFrames, position_data
    status = not headless
    done = False
    sucessFrames = 0
    failedFrames = 0

    while True:
        while(status):
            frame = picam2.capture_array()
            with latest_frame_lock:
                global latest_frame
                latest_frame = frame.copy()

            scan_res = extract_aruco(frame, mtx, dist, aruco_dict, aruco_params, headless, showRejected)
            if (scan_res):
                sucessFrames += 1
            else:
                failedFrames += 1

            if (not headless):
                cv2.imshow('ArUco Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    status = False
                    done = True

        # Break if done
        if (done):
            break

        # Wait for status to change from rest API
        while(not status):
            time.sleep(0.01)

    picam2.stop()
    cv2.destroyAllWindows()

def extract_aruco(frame, mtx, dist, aruco_dict, aruco_params, headless, showRejected=False):
    global position_data, object_positions

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:

        found = False
        detected_objects = {}
        for i, marker_id in enumerate(ids.flatten()):
            #print(f"i = {i} marker_id = {marker_id}")
            if marker_id in marker_info or marker_id in gameobject:
                if marker_id in marker_info:
                    #print(f"in marker info size and global position {marker_info[marker_id]}")
                    size, global_position = marker_info[marker_id]
                else:
                    #print(f"in game object size set to 100 and entry = {gameobject.get(marker_id)}")
                    entry = gameobject.get(marker_id)
                    size = 100.0
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        size = float(entry[0])
                    elif isinstance(entry, dict) and "size" in entry:
                        size = float(entry["size"])
                    #print(f"new size set to {size}")

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], size, mtx, dist)
                rvec, tvec = rvecs[0], tvecs[0]  # shape (1,3) → (3,)
                #print(f"rotation vec = {rvec} and translation vect = {tvec}")
                if not headless:
                    cv2.aruco.drawDetectedMarkers(frame, corners)
                    cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 50)
                    if showRejected and len(rejected) > 0:
                        cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(100, 0, 255))

                # Rotation matrix (marker→camera)
                rotation_matrix, _ = cv2.Rodrigues(rvec)

                # Camera position in marker coordinates
                camera_position = -rotation_matrix.T @ tvec.reshape(3, 1)
                camera_position = camera_position.flatten()
                #print(f"camera position in marker coord {camera_position}")
                
                if marker_id in marker_info:
                    # Combine with marker’s global position
                    # assuming global_position = (X_marker, Y_marker, Z_marker)
                    new_pos_data = {"x": 0, "y" : 0, "a": 0, "z": 0}
                    new_pos_data["x"] = float(global_position[0] - camera_position[1])
                    new_pos_data["y"] = float(global_position[1] + camera_position[0])
                    new_pos_data["z"] = float(camera_position[2])

                    # Step 2: Convert rotation matrix to Euler angles
                    # Specify the order of axes (e.g., "xyz", "zyx", etc.)
                    rotation = R.from_matrix(rotation_matrix)
                    euler_angles = rotation.as_euler('zyx', degrees=True)  # 'xyz' or your desired convention
                    new_pos_data["a"] = -euler_angles[0] + 180.0
                    if(new_pos_data["a"] > 180.0):
                        new_pos_data["a"] -= 360
                    elif(new_pos_data["a"] < -180.0):
                        new_pos_data["a"] += 360.0

                    #print(f"Camera: {new_pos_data}")
                    add_average_position(new_pos_data)
                    found = True

                if marker_id in gameobject:

                    # --- FILTRE DES FACES LATÉRALES ---
                    # On vérifie l'alignement de l'axe Z du marqueur avec l'axe Z de la caméra.
                    # 0.707 correspond au cosinus d'un angle de 45°. 
                    # Si la valeur est inférieure, le tag est trop incliné (donc sur une face latérale).
                    seuil_inclinaison = 0.6 
                    if abs(rotation_matrix[2, 2]) < seuil_inclinaison:
                        continue # On ignore ce tag et on passe au suivant dans la boucle for
                    
                    entry = gameobject.get(marker_id) #déjà fait ligne 211
                    #print(f" entry toujours {entry}")
                    # Récupération des infos de l'objet
                    if isinstance(entry, dict):
                        label = entry.get("label") or str(marker_id)
                        real_size = float(entry.get("size", 30.0))
                    elif entry is not None:
                        #print(entry)# debug
                        real_size = float(entry[0])
                        label = entry[1]
                    else:
                        print("Entry is None: Error")

                    # ORIENTATION
                    rotation = R.from_matrix(rotation_matrix)
                    euler_angles = rotation.as_euler('zyx', degrees=True)
                    yaw = -euler_angles[0] + 180.0
                    if yaw > 180.0: yaw -= 360
                    elif yaw < -180.0: yaw += 360.0

                    obj_data = {
                        "label": label,
                        "x": float(float(0 - camera_position[1])),
                        "y": float(camera_position[0]),
                        "z": float(camera_position[2]),
                        "a": float(yaw),
                        "last_seen": time.time()
                    }
                    #print(f"Un objet {obj_data}")
                    key = str(marker_id)
                    if key not in detected_objects:
                        detected_objects[key] = []
                    detected_objects[key].append(obj_data)

        with object_positions_lock:
            object_positions.clear()
            object_positions.update(detected_objects)

        return found

    with object_positions_lock:
        object_positions.clear()
    return False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Detect ArUco markers and run a REST API.")
    
    # Positional argument for calibration file
    parser.add_argument("calibration_file", type=str, help="Path to the calibration file.")
    
    # Optional arguments
    parser.add_argument("--api-port", type=int, default=5000, help="Port for the REST API (default: 5000).")
    parser.add_argument("--headless", type=bool, default=False, help="If the program should run headless (default: False).")
    parser.add_argument("--showRejected", type=bool, default=False, help="Show rejected markers (default: False).")
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280).")
    parser.add_argument("--height", type=int, default=800, help="Camera height (default: 800).")

    args = parser.parse_args()

    marker_info = {
        33: (100.0, (0.0, 0.0)),   # testing
        20: (100.0, (-400.0, -900.0)),   # Top-right
        21: (100.0, (-400.0, 900.0)),  # Top-left
        22: (100.0, (400.0, -900.0)), # Bottom-left
        23: (100.0, (400.0, 900.0)),   # Bottom-right
        0: (100.0, (0.0, 0.0))  # test
    }
    gameobject = {
        36: (30.0, "Blue"),
        47: (30.0, "Yellow"),
    }

    # Start REST API in a separate thread
    api_thread = threading.Thread(target=app.run, kwargs={
        "port": args.api_port,
        "debug": False,
        "host": "0.0.0.0"
    })
    api_thread.daemon = True
    api_thread.start()

    detect_aruco(args.calibration_file, marker_info, args.headless, args.showRejected, args.width, args.height)
