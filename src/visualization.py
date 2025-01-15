from utils import filter_points
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import cv2
import os


def create_camera_geometry(R, t, size=1.0, color=[1, 0, 0]):
    """Create a camera geometry for visualization."""
    # Create camera model pointing in positive z direction (looking forward)
    points = np.array([
        [0, 0, 0],          # Camera center
        [-1, -1, 2],        # Front face corners - made deeper and larger
        [1, -1, 2],
        [1, 1, 2],
        [-1, 1, 2]
    ]) * size
    
    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from center to front face
        [1, 2], [2, 3], [3, 4], [4, 1]    # Front face
    ])
    
    # Transform points by camera pose
    # Note: We need to transform from camera to world coordinates
    points = (R.T @ (points.T - t)).T
    
    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set


def create_camera_trajectory(camera_poses):
    """Create a line set showing the camera trajectory."""
    points = []
    lines = []
    colors = []
    
    # Extract camera centers (not translation vectors)
    for i, (R, t) in camera_poses.items():
        # Convert from translation to camera center: C = -R^T * t
        center = (-R.T @ t).ravel()
        points.append(center)
        if len(points) > 1:
            lines.append([len(points)-2, len(points)-1])
            colors.append([0, 1, 0])  # Green trajectory
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set


def visualize_3d_reconstruction(points_3d, R, t, K, additional_cameras=None):
    """
    Visualize 3D points and all camera poses using Open3D.
    
    Args:
        points_3d: Nx3 array of 3D points
        R: Rotation matrix of the final camera
        t: Translation vector of the final camera
        K: Camera intrinsic matrix
        additional_cameras: Dict of {frame_idx: (R, t)} for all cameras
    """
    # Filter outlier points
    filtered_points = filter_points(points_3d, percentile=98)
    
    # Create Open3D point cloud
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    
    # Add colors based on height (z-coordinate)
    colors = np.zeros((len(filtered_points), 3))
    z_vals = filtered_points[:, 2]
    z_min, z_max = np.min(z_vals), np.max(z_vals)
    normalized_z = (z_vals - z_min) / (z_max - z_min)
    
    # Create a color gradient (blue to red)
    colors[:, 0] = normalized_z  # Red channel
    colors[:, 2] = 1 - normalized_z  # Blue channel
    o3d_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=2560, height=1440)
    
    # Add point cloud
    vis.add_geometry(o3d_cloud)
    
    # Add camera geometries with adjusted size
    # First camera at origin
    camera1 = create_camera_geometry(np.eye(3), np.zeros((3, 1)), size=0.5, color=[0, 1, 0])
    vis.add_geometry(camera1)
    
    # Second camera
    camera2 = create_camera_geometry(R, t, size=0.5, color=[1, 0, 0])
    vis.add_geometry(camera2)
    
    # Add all additional cameras if provided
    if additional_cameras:
        for frame_idx, (R_cam, t_cam) in additional_cameras.items():
            if frame_idx == 0:  # Skip first camera (already added)
                continue
            camera = create_camera_geometry(
                R_cam, 
                t_cam, 
                size=0.5,  # Increased size
                color=[0.5, 0.5, 0.5]
            )
            vis.add_geometry(camera)
    
    # Add camera trajectory if we have multiple cameras
    if additional_cameras and len(additional_cameras) > 2:
        trajectory = create_camera_trajectory(additional_cameras)
        vis.add_geometry(trajectory)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.line_width = 2.0
    
    # Set camera viewpoint for better initial view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)
    ctr.set_front([0, -1, -1])  # Look from above and front
    ctr.set_lookat([0, 0, 0])   # Look at center
    ctr.set_up([0, -1, 0])      # Set up direction
    
    # Run visualization
    vis.run()
    vis.destroy_window()


def plot_chessboard_corners(frame, corners, ret, index, CHESSBOARD_SIZE):
    """Plot detected chessboard corners for a frame."""
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Original Frame')
    plt.axis('off')
    
    # Draw corners on a copy of the frame
    frame_corners = frame.copy()
    cv2.drawChessboardCorners(frame_corners, CHESSBOARD_SIZE, corners, ret)
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(frame_corners, cv2.COLOR_BGR2RGB))
    plt.title('Detected Corners')
    plt.axis('off')
    
    plt.suptitle(f'Frame {index + 1}')
    plt.tight_layout()
    
    # Save the plot
    output_dir = 'output/calibration_steps'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'frame_{index+1}_corners.png'))
    plt.close()


def plot_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist, image_size, frames):
    """Plot reprojection error for each point in each image and show used frames."""
    mean_error = 0
    errors = []
    
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        errors.append(error)
        mean_error += error

    mean_error = mean_error/len(objpoints)
    
    # Calculate grid dimensions for frames
    n_frames = len(frames)
    n_cols = min(5, n_frames)  # Maximum 5 columns
    n_rows = (n_frames + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with appropriate size
    plt.figure(figsize=(15, 5 + 3*n_rows))
    
    # Plot reprojection error on top
    plt.subplot2grid((n_rows + 1, 1), (0, 0))
    plt.plot(errors, 'bo-')
    plt.axhline(y=mean_error, color='r', linestyle='--', label=f'Mean Error: {mean_error:.4f}')
    plt.xlabel('Image Index')
    plt.ylabel('Reprojection Error (pixels)')
    plt.title('Reprojection Error per Image')
    plt.grid(True)
    plt.legend()
    
    # Plot frames in a grid below
    for i, frame in enumerate(frames):
        row = i // n_cols + 1  # +1 because first row is error plot
        col = i % n_cols
        plt.subplot2grid((n_rows + 1, n_cols), (row, col))
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f'Frame {i+1}\nError: {errors[i]:.4f}')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = 'output/calibration_steps'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'calibration_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return mean_error