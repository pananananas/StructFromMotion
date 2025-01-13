from utils import use_camera_calibration, extract_frames, extract_frames_from_dir, visualize_matches, convert_npy_to_ply
from triangulation import estimate_pose_and_triangulate
from visualization import visualize_3d_reconstruction
from features import detect_features, match_features
from icecream import ic
import numpy as np
import os
from incremental_reconstruction import incremental_reconstruction

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Config
output_dir = 'output'
save_path = f'{output_dir}/points_3d.npy'
save_ply_path = f'{output_dir}/points_3d.ply'

max_frames = 20
frame_interval = 45
video_path = 'data/video2.MOV'

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

img1 = frames[0]
img2 = frames[1]
kp1  = keypoints_list[0]
kp2  = keypoints_list[1]
des1 = descriptors_list[0]
des2 = descriptors_list[1] 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Feature Matching
good_matches = match_features(des1, des2, ratio=0.8)
print(f"Number of good matches between Frame 1 and Frame 2: {len(good_matches)}")

visualize_matches(img1, kp1, img2, kp2, good_matches, max_matches=10000)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 2 Images Reconstruction

R, t, points_3d, mask = estimate_pose_and_triangulate(kp1, kp2, good_matches, K)

if R is None or t is None or points_3d is None:
    print("Failed")
print("For 2 images")
ic(len(points_3d))

visualize_3d_reconstruction(points_3d, R, t, K) 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Incremental Reconstruction

print("\nPerforming incremental reconstruction...")
reconstruction_state = incremental_reconstruction(
    frames=frames,
    keypoints_list=keypoints_list,
    descriptors_list=descriptors_list,
    K=K,
    window_size=5
)

# Visualize the complete reconstruction
if len(reconstruction_state.points_3d) > 0:
    print(f"\nReconstruction complete:")
    print(f"- Number of 3D points: {len(reconstruction_state.points_3d)}")
    print(f"- Number of cameras: {len(reconstruction_state.camera_poses)}")
    
    # Save the 3D points
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(save_path, reconstruction_state.points_3d)
    
    # Convert to PLY for external viewing
    convert_npy_to_ply(save_path, save_ply_path)
    
    # Visualize the reconstruction with all cameras
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
