from utils import extract_frames, extract_frames_from_dir, visualize_matches, convert_npy_to_ply
from triangulation import estimate_pose_and_triangulate, estimate_poses_and_triangulate_sequence
from visualization import visualize_full_reconstruction, visualize_3d_reconstruction
from features import detect_features, match_features
import numpy as np

output_dir = 'output'

max_frames = 20
frame_interval = 60
video_path = 'data/rollei3.mov'
frames = extract_frames(video_path, frame_interval, max_frames)

# getting images from directory
# directory_path = '../data/south-building/images'
# frames = extract_frames_from_dir(directory_path, max_frames)

keypoints_list, descriptors_list = detect_features(frames)

# Select the first two frames
img1 = frames[0]
img2 = frames[1]
kp1  = keypoints_list[0]
kp2  = keypoints_list[1]
des1 = descriptors_list[0]
des2 = descriptors_list[1] 

good_matches = match_features(des1, des2, ratio=0.8)
print(f"Number of good matches between Frame 1 and Frame 2: {len(good_matches)}")

# visualize_matches(img1, kp1, img2, kp2, good_matches, max_matches=10000)

f_x = 3225.6
f_y = 3225.6
c_x = img1.shape[1] / 2  # 4032 / 2 = 2016
c_y = img1.shape[0] / 2  # 3024 / 2 = 1512

K = np.array([[f_x, 0, c_x],
              [0, f_y, c_y],
              [0,  0,    1]])
print("Camera Intrinsic Matrix K:")
print(K)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 2 Images Reconstruction

R, t, points_3d, mask = estimate_pose_and_triangulate(kp1, kp2, good_matches, K)

if R is None or t is None or points_3d is None:
    print("Failed")

print(f'number of points: {len(points_3d)}')

visualize_3d_reconstruction(points_3d, R, t, K) 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Full Reconstruction

all_points_3d, camera_poses, point_tracks = estimate_poses_and_triangulate_sequence(frames, keypoints_list, descriptors_list, K, debug=True)

if all_points_3d is not None:
    print(f"Reconstructed {len(all_points_3d)} 3D points")
    print(f"Estimated {len(camera_poses)} camera poses")


save_path = f'{output_dir}/points_3d.npy'
np.save(save_path, all_points_3d)
convert_npy_to_ply(save_path, f'{output_dir}/points_3d.ply')

visualize_full_reconstruction(all_points_3d, camera_poses)