from utils import use_camera_calibration, extract_frames, extract_frames_from_dir, visualize_matches, convert_npy_to_ply
from triangulation import estimate_pose_and_triangulate
from visualization import visualize_3d_reconstruction
from features import detect_features, match_features
from icecream import ic
import os

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Config
output_dir = 'output'
save_path = f'{output_dir}/points_3d.npy'
save_ply_path = f'{output_dir}/points_3d.ply'

max_frames = 2
frame_interval = 60
video_path = 'data/rollei3.mov'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Extracting frames
frames = extract_frames(video_path, frame_interval, max_frames)

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

# visualize_matches(img1, kp1, img2, kp2, good_matches, max_matches=10000)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 2 Images Reconstruction

R, t, points_3d, mask = estimate_pose_and_triangulate(kp1, kp2, good_matches, K)

if R is None or t is None or points_3d is None:
    print("Failed")

ic(len(points_3d))

visualize_3d_reconstruction(points_3d, R, t, K) 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Full Reconstruction