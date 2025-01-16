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


def show_frames(frames):
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    fig, axs = plt.subplots(1, len(frames), figsize=(15, 5))
    for i, frame in enumerate(frames):
        axs[i].imshow(frame)
        axs[i].axis('off')
    
    plt.show()

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
        # Fallback to iPhone 12 Pro parameters
        f_x = 3225.6
        f_y = 3225.6
        c_x = 1200
        c_y = 1800
        K = np.array([[f_x, 0, c_x],
                    [0, f_y, c_y],
                    [0,  0,    1]])
        dist = np.zeros(5)  # No distortion
        print("Camera Matrix (K):")
        print(K)
        print("\nNo distortion:")
        print(dist.ravel())
    return K, dist

def print_adjacency_matrix(descriptors_list, ratio=0.8, min_matches=20, window_size=5):
    """
    Print an adjacency matrix showing the number of matches between sequential frames within a window.
    
    Parameters:
        descriptors_list: List of descriptors for each image
        ratio: Ratio test threshold for feature matching (default: 0.8)
        min_matches: Minimum number of matches to consider images connected (default: 20)
        window_size: Size of sliding window for matching (default: 3)
    
    Returns:
        numpy array: Adjacency matrix where each cell contains the number of matches
    """
    from features import match_features
    
    n_images = len(descriptors_list)
    adj_matrix = np.zeros((n_images, n_images), dtype=int)
    
    print("\nImage Adjacency Matrix (number of matches within window):")
    print("-" * (n_images * 6 + 1))
    
    # Build adjacency matrix using sliding window
    for i in range(n_images - 1):
        # Only match with next window_size frames
        for j in range(i + 1, min(i + window_size + 1, n_images)):
            matches = match_features(
                descriptors_list[i],
                descriptors_list[j],
                ratio=ratio,
                cross_check=True
            )
            num_matches = len(matches)
            adj_matrix[i, j] = num_matches
            adj_matrix[j, i] = num_matches  # Keep matrix symmetric for visualization
    
    # Print header
    print("    ", end="")
    for i in range(n_images):
        print(f"{i:4d} ", end="")
    print("\n" + "-" * (n_images * 6 + 1))
    
    # Print matrix with row labels
    for i in range(n_images):
        print(f"{i:2d} |", end=" ")
        for j in range(n_images):
            if i == j:
                print("  -  ", end="")
            else:
                matches = adj_matrix[i, j]
                if matches == 0:
                    print("  .  ", end="")  # Dots for pairs not in window
                elif matches >= min_matches:
                    print(f"\033[92m{matches:4d}\033[0m ", end="")  # Green for good matches
                else:
                    print(f"{matches:4d} ", end="")
        print()
    
    print("-" * (n_images * 6 + 1))
    return adj_matrix


def compute_baseline_angle(R1, t1, R2, t2) -> float:
    """Compute baseline angle between two cameras."""
    # Ensure vectors are properly shaped
    c1 = -R1.T @ t1.reshape(3, 1)  # Camera center 1
    c2 = -R2.T @ t2.reshape(3, 1)  # Camera center 2
    v1 = R1.T @ np.array([0, 0, 1]).reshape(3, 1)  # View direction 1
    v2 = R2.T @ np.array([0, 0, 1]).reshape(3, 1)  # View direction 2
    
    # Compute baseline vector
    baseline = c2 - c1
    baseline = baseline / np.linalg.norm(baseline)
    
    # Compute angles
    angle1 = np.arccos(np.clip(np.dot(baseline.flatten(), v1.flatten()), -1.0, 1.0))
    angle2 = np.arccos(np.clip(np.dot(-baseline.flatten(), v2.flatten()), -1.0, 1.0))
    
    return min(angle1, angle2) * 180 / np.pi


def visualize_all_matches(frames, keypoints_list, descriptors_list, window_size=5, ratio=0.8, max_matches=2000, output_dir='output'):
    """
    Visualize matches between sequential frames within a window and save to file.
    
    Parameters:
        frames: List of input frames
        keypoints_list: List of keypoints for each frame
        descriptors_list: List of descriptors for each frame
        window_size: Size of sliding window for matching
        ratio: Ratio test threshold for feature matching
        max_matches: Maximum number of matches to display
        output_dir: Directory to save visualization
    """
    from features import match_features
    import matplotlib.pyplot as plt
    
    n_images = len(frames)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate grid size
    total_pairs = sum(1 for i in range(n_images - 1) 
                     for j in range(i + 1, min(i + window_size + 1, n_images)))
    grid_size = int(np.ceil(np.sqrt(total_pairs)))
    
    # Create figure
    fig = plt.figure(figsize=(grid_size * 6, grid_size * 4))
    plt.suptitle("Feature Matches Between Sequential Frames", fontsize=16, y=0.95)
    
    plot_idx = 1
    for i in range(n_images - 1):
        for j in range(i + 1, min(i + window_size + 1, n_images)):
            # Match features
            matches = match_features(
                descriptors_list[i],
                descriptors_list[j],
                ratio=ratio,
                cross_check=True
            )
            
            # Create subplot
            ax = fig.add_subplot(grid_size, grid_size, plot_idx)
            
            # Draw matches
            matched_img = cv2.drawMatches(
                frames[i], keypoints_list[i],
                frames[j], keypoints_list[j],
                matches[:max_matches], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            ax.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
            ax.set_title(f'Frames {i}-{j}: {len(matches)} matches')
            ax.axis('off')
            
            plot_idx += 1
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'feature_matches.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved feature matches visualization to {save_path}")
    plt.close()


def visualize_all_keypoints(frames, keypoints_list, output_dir='output'):
    """
    Visualize keypoints for all frames and save to file.
    
    Parameters:
        frames: List of input frames
        keypoints_list: List of keypoints for each frame
        output_dir: Directory to save visualization
    """
    import matplotlib.pyplot as plt
    
    n_images = len(frames)
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(n_images)))
    
    # Create figure
    fig = plt.figure(figsize=(grid_size * 6, grid_size * 4))
    plt.suptitle("Detected Keypoints in Each Frame", fontsize=16, y=0.95)
    
    for i in range(n_images):
        # Create subplot
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        
        # Draw keypoints
        img_with_kp = cv2.drawKeypoints(
            frames[i],
            keypoints_list[i],
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        ax.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Frame {i}: {len(keypoints_list[i])} keypoints')
        ax.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'keypoints.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved keypoints visualization to {save_path}")
    plt.close()
