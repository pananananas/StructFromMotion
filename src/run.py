from utils import use_camera_calibration, extract_frames, extract_frames_from_dir, visualize_matches, convert_npy_to_ply
from incremental_reconstruction import incremental_reconstruction
from triangulation import estimate_pose_and_triangulate
from visualization import visualize_3d_reconstruction
from features import detect_features, match_features
from icecream import ic
import numpy as np
import os
import sys

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Config
output_dir = 'output'
save_path = f'{output_dir}/points_3d.npy'
save_ply_path = f'{output_dir}/points_3d.ply'

max_frames = 10
frame_interval = 45
video_path = 'data/rollei3.mov'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Extracting frames
frames = extract_frames(video_path, frame_interval, max_frames)

# frames = extract_frames_from_dir('data/gerrard-hall/images', max_frames)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Feature Detection
keypoints_list, descriptors_list = detect_features(frames)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Camera Calibration
K, dist = use_camera_calibration()

# Initialize with first pair
print("\nInitializing with first pair...")
matches_01 = match_features(descriptors_list[0], descriptors_list[1], ratio=0.8, cross_check=True)
print(f"Found {len(matches_01)} verified matches")

R, t, points_3d, mask = estimate_pose_and_triangulate(
    keypoints_list[0],
    keypoints_list[1],
    matches_01,
    K,
    debug=True
)

if R is None:
    print("Failed to initialize reconstruction")
    sys.exit(1)

print(f"Initial reconstruction: {len(points_3d)} points")

# Visualize initial reconstruction
visualize_3d_reconstruction(points_3d, R, t, K)

# Proceed with incremental reconstruction
print("\nPerforming incremental reconstruction...")
reconstruction_state = incremental_reconstruction(
    frames=frames,
    keypoints_list=keypoints_list,
    descriptors_list=descriptors_list,
    K=K,
    window_size=5
)

# Visualize final reconstruction
if len(reconstruction_state.points_3d) > 0:
    print(f"\nReconstruction complete:")
    print(f"- Number of 3D points: {len(reconstruction_state.points_3d)}")
    print(f"- Number of cameras: {len(reconstruction_state.camera_poses)}")
    
    # Save results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(save_path, reconstruction_state.points_3d)
    convert_npy_to_ply(save_path, save_ply_path)
    
    # Final visualization
    last_camera_idx = max(reconstruction_state.camera_poses.keys())
    R_final = reconstruction_state.camera_poses[last_camera_idx][0]
    t_final = reconstruction_state.camera_poses[last_camera_idx][1]
    
    visualize_3d_reconstruction(
        reconstruction_state.points_3d,
        R_final,
        t_final,
        K,
        additional_cameras=reconstruction_state.camera_poses
    )
else:
    print("Reconstruction failed - insufficient points reconstructed")
