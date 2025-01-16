# Implementation from: docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

from visualization import plot_chessboard_corners, plot_reprojection_error
from utils import extract_frames
import numpy as np
import pickle
import cv2
import os

# Chessboard parameters
CHESSBOARD_SIZE = (9, 7)
SQUARE_SIZE = 20.0

def calibrate_camera(frames):
    """
    Calibrate camera using chessboard pattern.
    Returns camera matrix, distortion coefficients, rotation and translation vectors.
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)
    objp = objp * SQUARE_SIZE  # Scale to actual size

    objpoints = []    # 3d points in real world space
    imgpoints = []    # 2d points in image plane
    used_frames = []  # frames with detected chessboard 

    print("\nDetecting chessboard corners...")
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            used_frames.append(frame)
            
            plot_chessboard_corners(frame, corners2, ret, i, CHESSBOARD_SIZE)
            print(f"Successfully detected corners in frame {i+1}")

    if len(objpoints) == 0:
        raise Exception("No chessboard patterns were found in the frames!")

    print(f"\nFound chessboard pattern in {len(objpoints)} frames")
    print("Performing camera calibration...")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, 
        imgpoints, 
        gray.shape[::-1], 
        None, 
        None
    )

    mean_error = plot_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist, gray.shape[::-1], used_frames)
    print(f"\nMean reprojection error: {mean_error:.4f} pixels")

    return mtx, dist, rvecs, tvecs


def save_calibration(mtx, dist, output_dir='output'):
    """Save calibration results to a file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    calibration_data = {
        'camera_matrix': mtx,
        'dist_coeffs': dist
    }
    
    output_path = os.path.join(output_dir, 'calibration.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(calibration_data, f)
    print(f"\nCalibration data saved to {output_path}")
    

if __name__ == "__main__":

    max_frames = 20
    frame_interval = 30
    video_path = 'data/calibrate.MOV'
    
    print("Extracting frames from video...")

    frames = extract_frames(video_path, frame_interval, max_frames)

    if len(frames) == 0:
        raise Exception("No frames were extracted from the video!")

    print(f"Extracted {len(frames)} frames. Starting calibration...")

    try:
        mtx, dist, rvecs, tvecs = calibrate_camera(frames)
        
        print("\nCalibration successful!")
        print("\nCamera matrix:")
        print(mtx)
        print("\nDistortion coefficients:")
        print(dist.ravel())
        
        save_calibration(mtx, dist)
        
        print("\nVisualization results have been saved in output/calibration_steps/")
        
    except Exception as e:
        print(f"Calibration failed: {str(e)}")