from utils import filter_points
import open3d as o3d
import numpy as np

def visualize_3d_reconstruction(points_3d, R, t, K):
    """
    Visualize 3D points and camera poses using Open3D with improved visualization.
    """
    # Filter outlier points
    filtered_points = filter_points(points_3d, percentile=98)
    # filtered_points = points_3d
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
    
    # Statistical outlier removal
    # o3d_cloud, _ = o3d_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=2560, height=1440)
    
    # Add geometries
    vis.add_geometry(o3d_cloud)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.point_size = 3.0  # Increased point size
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
    
    # Set camera viewpoint
    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)
    ctr.set_front([-0.5, -0.5, -0.5])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    
    # Run visualization
    vis.run()
    vis.destroy_window()



def visualize_full_reconstruction(all_points_3d, all_cameras):
    """
    Visualize the complete 3D reconstruction with camera positions.
    """
    # Filter outliers
    filtered_points = filter_points(all_points_3d, percentile=98)
    
    # Create Open3D point cloud
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    
    # Add colors based on height
    colors = np.zeros((len(filtered_points), 3))
    z_vals = filtered_points[:, 2]
    z_min, z_max = np.min(z_vals), np.max(z_vals)
    normalized_z = (z_vals - z_min) / (z_max - z_min)
    colors[:, 0] = normalized_z
    colors[:, 2] = 1 - normalized_z
    o3d_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=2560, height=1440)
    vis.add_geometry(o3d_cloud)
    
    # Add camera frustums
    for R, t in all_cameras:
        # Create camera frustum geometry
        cam_size = 0.1
        camera_points = np.array([
            [0, 0, 0],
            [-cam_size, -cam_size, cam_size],
            [cam_size, -cam_size, cam_size],
            [cam_size, cam_size, cam_size],
            [-cam_size, cam_size, cam_size]
        ])
        
        # Transform camera points to global coordinate system
        camera_points = (R @ camera_points.T + t).T
        
        # Create lines for camera frustum
        lines = [[0, 1], [0, 2], [0, 3], [0, 4],
                [1, 2], [2, 3], [3, 4], [4, 1]]
        
        # Create LineSet for camera frustum
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(camera_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])
        vis.add_geometry(line_set)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    
    # Set camera viewpoint
    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)
    ctr.set_front([-0.5, -0.5, -0.5])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    
    # Run visualization
    vis.run()
    vis.destroy_window()