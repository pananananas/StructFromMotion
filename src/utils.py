from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
import sys
import cv2
import os

def extract_frames(video_path, frame_interval=30, max_frames=100, skip_frames=0, scale_factor=1):
    """
    Extract frames from a video file.

    Parameters:
        video_path (str): Path to the video file.
        frame_interval (int): Extract one frame every 'frame_interval' frames.
        max_frames (int): Maximum number of frames to extract.
        skip_frames (int): Number of frames to skip at the start.
        scale_factor (int): Factor to scale down resolution by (1, 2, or 4).

    Returns:
        List of extracted frames as BGR images.
    """
    if not os.path.isfile(video_path):
        print(f"Video file not found: {video_path}")
        sys.exit(1)
        
    if scale_factor not in [1, 2, 4]:
        print(f"Invalid scale_factor {scale_factor}. Must be 1, 2, or 4.")
        sys.exit(1)
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    extracted = 0
    
    # Skip initial frames
    for _ in range(skip_frames):
        cap.read()
        count += 1
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - skip_frames
    expected_frames = min(max_frames, total_frames // frame_interval)
    
    with tqdm(total=expected_frames, desc="Extracting frames") as pbar:
        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                if scale_factor > 1:
                    height, width = frame.shape[:2]
                    new_height = height // scale_factor
                    new_width = width // scale_factor
                    frame = cv2.resize(frame, (new_width, new_height))
                frames.append(frame)
                extracted += 1
                pbar.update(1)
            count += 1
    
    cap.release()
    print(f"Total extracted frames: {len(frames)}")
    return frames


def extract_frames_from_dir(directory_path, max_frames=100):
    """
    Extract frames from a directory of images.
    
    Args:
        directory_path: Path to directory containing images
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of frames as BGR images
    """
    frames = []
    for file in os.listdir(directory_path):
        if len(frames) >= max_frames:
            break
        if file.endswith(".JPG") or file.endswith(".png"):
            img = cv2.imread(os.path.join(directory_path, file))
            frames.append(img)
    return frames


def visualize_matches(img1, kp1, img2, kp2, matches, max_matches=50):
    """
    Visualize matches between two images.

    Parameters:
        img1: First image (BGR).
        kp1: Keypoints from the first image.
        img2: Second image (BGR).
        kp2: Keypoints from the second image.
        matches: List of matches.
        max_matches: Maximum number of matches to display.
    """
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_matches], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Top {max_matches} Matches")
    plt.axis('off')
    plt.show()


def filter_points(points_3d, percentile=95):
    """
    Filter out outlier points based on distance from median.
    """
    # Calculate distances from median point
    median = np.median(points_3d, axis=0)
    distances = np.linalg.norm(points_3d - median, axis=1)
    
    # Filter out points beyond the specified percentile
    threshold = np.percentile(distances, percentile)
    mask = distances < threshold
    
    return points_3d[mask]


def convert_npy_to_ply(npy_file, ply_file):
    # Load the .npy file
    points = np.load(npy_file)  # Expecting shape (N, 3) or (N, 6) for RGB

    if points.shape[1] == 3:
        # Only XYZ coordinates
        vertices = np.array([tuple(pt) for pt in points],
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    elif points.shape[1] >= 6:
        # XYZ plus RGB
        vertices = np.array([tuple(pt[:6]) for pt in points],
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    else:
        raise ValueError("Unsupported .npy file format. Expected shape (N, 3) or (N, 6).")

    # Create PlyElement
    ply_element = PlyElement.describe(vertices, 'vertex')

    # Write to .ply file
    PlyData([ply_element], text=True).write(ply_file)
    print(f"Successfully converted {npy_file} to {ply_file}")


def load_camera_calibration(calibration_path='output/calibration.pkl'):
    """Load camera calibration parameters from file."""
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(
            f"Camera calibration file not found at {calibration_path}. "
            "Please run camera_calibration.py first."
        )
    
    with open(calibration_path, 'rb') as f:
        calibration_data = pickle.load(f)
    
    return calibration_data['camera_matrix'], calibration_data['dist_coeffs']

def use_camera_calibration():
# Load camera calibration parameters
    try:
        K, dist = load_camera_calibration()
        print("\nLoaded camera calibration parameters:")
        print("Camera Matrix (K):")
        print(K)
        print("\nDistortion Coefficients:")
        print(dist.ravel())

    except FileNotFoundError as e:
        print(f"\nWarning: {str(e)}")
        print("Using default camera parameters...")
        # Fallback to default parameters if calibration file not found
        f_x = 3225.6
        f_y = 3225.6
        c_x = 2016  # Will be updated with actual image size
        c_y = 1512  # Will be updated with actual image size
        K = np.array([[f_x, 0, c_x],
                    [0, f_y, c_y],
                    [0,  0,    1]])
        dist = np.zeros(5)  # No distortion correction
    return K, dist