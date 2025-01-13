import numpy as np
import cv2
from typing import List, Tuple, Dict
from dataclasses import dataclass
from features import match_features
from triangulation import estimate_pose_and_triangulate
import logging

@dataclass
class ReconstructionState:
    """Stores the state of the reconstruction process."""
    camera_poses: Dict[int, Tuple[np.ndarray, np.ndarray]]  # frame_idx -> (R, t)
    points_3d: np.ndarray
    point_tracks: Dict[int, List[Tuple[int, int]]]  # point_idx -> [(frame_idx, keypoint_idx)]
    scale_factors: List[float]

def compute_relative_scale(points_3d_prev: np.ndarray, points_3d_new: np.ndarray) -> float:
    """
    Compute relative scale between two sets of 3D points to maintain consistent scale.
    """
    if len(points_3d_prev) < 4 or len(points_3d_new) < 4:
        return 1.0

    # Compute the median distances between consecutive points in both sets
    dist_prev = np.median([np.linalg.norm(points_3d_prev[i] - points_3d_prev[i-1]) 
                          for i in range(1, len(points_3d_prev))])
    dist_new = np.median([np.linalg.norm(points_3d_new[i] - points_3d_new[i-1]) 
                         for i in range(1, len(points_3d_new))])
    
    if dist_new < 1e-10:
        return 1.0
        
    scale = dist_prev / dist_new
    return min(max(scale, 0.1), 10.0)  # Limit scale factor to reasonable range

def find_common_matches(matches_dict: Dict[Tuple[int, int], List]) -> List[Tuple[int, int, int]]:
    """
    Find keypoints that match across multiple frames in the sliding window.
    Returns: List of (frame1_idx, frame2_idx, num_matches)
    """
    common_matches = []
    frame_pairs = list(matches_dict.keys())
    
    for i, (frame1, frame2) in enumerate(frame_pairs):
        num_matches = len(matches_dict[(frame1, frame2)])
        if num_matches >= 20:  # Minimum number of matches threshold
            common_matches.append((frame1, frame2, num_matches))
    
    return sorted(common_matches, key=lambda x: x[2], reverse=True)

def incremental_reconstruction(frames: List[np.ndarray], 
                             keypoints_list: List, 
                             descriptors_list: List,
                             K: np.ndarray,
                             window_size: int = 5) -> ReconstructionState:
    """
    Perform incremental reconstruction using a sliding window approach.
    """
    state = ReconstructionState(
        camera_poses={0: (np.eye(3), np.zeros((3, 1)))},  # First camera at origin
        points_3d=np.array([]),
        point_tracks={},
        scale_factors=[]
    )
    
    # Initialize with first pair of frames
    matches = match_features(descriptors_list[0], descriptors_list[1])
    R, t, points_3d, mask = estimate_pose_and_triangulate(
        keypoints_list[0],
        keypoints_list[1],
        matches,
        K,
        debug=False
    )
    
    if R is not None and points_3d is not None:
        state.camera_poses[1] = (R, t)
        state.points_3d = points_3d
        # Initialize point tracks
        for i, match in enumerate(matches):
            if mask[i]:
                state.point_tracks[i] = [(0, match.queryIdx), (1, match.trainIdx)]
    else:
        print("Failed to initialize with first pair")
        return state
    
    # Keep track of which points are visible in which frames
    point_visibility = {i: {0, 1} for i in range(len(state.points_3d))}
    
    # Process remaining frames
    for frame_idx in range(2, len(frames)):
        print(f"Processing frame {frame_idx}")
        window_start = max(0, frame_idx - window_size)
        window_end = frame_idx
        
        best_num_matches = 0
        best_matches = None
        best_ref_idx = None
        
        # Find best matching reference frame in window
        for ref_idx in range(window_start, window_end):
            matches = match_features(descriptors_list[ref_idx], descriptors_list[frame_idx])
            if len(matches) > best_num_matches:
                best_num_matches = len(matches)
                best_matches = matches
                best_ref_idx = ref_idx
        
        if best_matches is None or len(best_matches) < 20:
            print(f"Insufficient matches for frame {frame_idx}")
            continue
            
        # Get reference camera pose
        ref_R, ref_t = state.camera_poses[best_ref_idx]
        
        # Estimate new camera pose relative to reference frame
        R_new, t_new, points_3d_new, mask = estimate_pose_and_triangulate(
            keypoints_list[best_ref_idx],
            keypoints_list[frame_idx],
            best_matches,
            K,
            debug=False
        )
        
        if R_new is None or points_3d_new is None or len(points_3d_new) < 10:
            print(f"Failed to estimate pose for frame {frame_idx}")
            continue
            
        # Transform to global coordinate system
        R_global = ref_R @ R_new
        t_global = ref_R @ t_new + ref_t
        
        # Compute and apply scale factor
        scale = compute_relative_scale(state.points_3d, points_3d_new)
        points_3d_new *= scale
        t_global *= scale
        state.scale_factors.append(scale)
        
        # Update reconstruction state
        state.camera_poses[frame_idx] = (R_global, t_global)
        
        # Update point visibility for new points
        start_idx = len(state.points_3d)
        for i in range(len(points_3d_new)):
            point_visibility[start_idx + i] = {best_ref_idx, frame_idx}
        
        # Merge new points with existing reconstruction
        state.points_3d = np.vstack([state.points_3d, points_3d_new])
        
        # Update point tracks
        for i, match in enumerate(best_matches):
            if mask[i]:
                point_idx = start_idx + i
                state.point_tracks[point_idx] = [
                    (best_ref_idx, match.queryIdx),
                    (frame_idx, match.trainIdx)
                ]
        
        print(f"Added frame {frame_idx} with {len(points_3d_new)} new points")
    
    # Final filtering of outlier points with improved visibility check
    if len(state.points_3d) > 0:
        filtered_points = []
        filtered_visibility = {}
        valid_point_indices = []
        
        for point_idx, point_3d in enumerate(state.points_3d):
            visible_frames = point_visibility[point_idx]
            visible_count = 0
            
            # Check if point is visible in at least 2 cameras
            for frame_idx in visible_frames:
                R, t = state.camera_poses[frame_idx]
                point_cam = R @ point_3d + t.ravel()
                if point_cam[2] > 0:  # Point is in front of camera
                    visible_count += 1
            
            if visible_count >= 2:  # Keep point if visible in at least 2 views
                filtered_points.append(point_3d)
                filtered_visibility[len(filtered_points)-1] = visible_frames
                valid_point_indices.append(point_idx)
        
        # Update state with filtered points
        state.points_3d = np.array(filtered_points)
        
        # Update point tracks to only include valid points
        new_tracks = {}
        for i, idx in enumerate(valid_point_indices):
            if idx in state.point_tracks:
                new_tracks[i] = state.point_tracks[idx]
        state.point_tracks = new_tracks
        
    return state 