from utils import compute_baseline_angle
from triangulation import estimate_pose_and_triangulate
from typing import List, Tuple, Dict
from features import match_features
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class ReconstructionState:
    camera_poses: Dict[int, Tuple[np.ndarray, np.ndarray]]  # frame_idx -> (R, t)
    points_3d: np.ndarray
    point_tracks: Dict[int, List[Tuple[int, int]]]  # track_id -> [(frame_idx, keypoint_idx)]
    track_to_point: Dict[int, int]                  # track_id -> point_3d_idx


def build_feature_tracks(keypoints_list, descriptors_list, window_size=3):
    """Build feature tracks across multiple frames."""
    tracks = {}
    next_track_id = 0
    keypoint_to_track = [{} for _ in range(len(keypoints_list))]
    
    # Process frames in sequence
    for frame_idx in range(len(keypoints_list) - 1):
        # Match with next few frames in window
        for offset in range(1, min(window_size, len(keypoints_list) - frame_idx)):
            next_idx = frame_idx + offset
            matches = match_features(
                descriptors_list[frame_idx], 
                descriptors_list[next_idx],
                ratio=0.8
            )
            print(f"Matches between frames {frame_idx}-{next_idx}: {len(matches)}")
            
            for match in matches:
                query_idx = match.queryIdx
                train_idx = match.trainIdx
                
                # Check if current keypoint is already part of a track
                if query_idx in keypoint_to_track[frame_idx]:
                    track_id = keypoint_to_track[frame_idx][query_idx]
                    # Add new observation to existing track
                    if next_idx not in [f for f, _ in tracks[track_id]]:
                        tracks[track_id].append((next_idx, train_idx))
                        keypoint_to_track[next_idx][train_idx] = track_id
                else:
                    # Start new track
                    track_id = next_track_id
                    next_track_id += 1
                    tracks[track_id] = [(frame_idx, query_idx), (next_idx, train_idx)]
                    keypoint_to_track[frame_idx][query_idx] = track_id
                    keypoint_to_track[next_idx][train_idx] = track_id
    
    filtered_tracks = {k: v for k, v in tracks.items() if len(v) >= 2}
    return filtered_tracks


def initialize_reconstruction(keypoints_list, descriptors_list, K) -> ReconstructionState:
    print("Initializing reconstruction from first pair...")
    
    matches = match_features(descriptors_list[0], descriptors_list[1], ratio=0.8)
    print(f"Found {len(matches)} matches for initialization")
    
    if len(matches) < 100:
        raise RuntimeError(f"Insufficient matches for initialization: {len(matches)}")
    
    R, t, points_3d, mask = estimate_pose_and_triangulate(
        keypoints_list[0],
        keypoints_list[1],
        matches,
        K,
        debug=True
    )
    
    if R is None or points_3d is None:
        raise RuntimeError("Failed to initialize reconstruction")
    
    print(f"Initial reconstruction: {len(points_3d)} points")
    
    state = ReconstructionState(
        camera_poses={
            0: (np.eye(3), np.zeros((3, 1))),
            1: (R, t)
        },
        points_3d=points_3d,
        point_tracks={},
        track_to_point={}
    )
    
    # Initialize point tracks and mapping
    point_idx = 0
    for i, match in enumerate(matches):
        if mask[i]:
            state.point_tracks[i] = [(0, match.queryIdx), (1, match.trainIdx)]
            state.track_to_point[i] = point_idx
            point_idx += 1
    
    return state


def optimize_poses_and_points(state, keypoints_list, K):
    """
    Simple bundle adjustment using motion-only optimization.
    """
    # Only process frames that exist in camera_poses
    processed_frames = sorted(state.camera_poses.keys())  # Process all frames
    
    for frame_idx in processed_frames:
        R, t = state.camera_poses[frame_idx]
        points_3d = []
        points_2d = []
        
        # Collect 2D-3D correspondences
        for track_id, track in state.point_tracks.items():
            if track_id in state.track_to_point:
                point_idx = state.track_to_point[track_id]
                point_3d = state.points_3d[point_idx]
                
                # Find 2D point in current frame
                for f, kp_idx in track:
                    if f == frame_idx:
                        point_2d = keypoints_list[frame_idx][kp_idx].pt
                        points_3d.append(point_3d)
                        points_2d.append(point_2d)
                        break
        
        if len(points_3d) < 10:
            continue
            
        # Optimize pose
        points_3d = np.array(points_3d)
        points_2d = np.array(points_2d)
        
        success, R_new, t_new, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            K,
            None,
            cv2.Rodrigues(R)[0],
            t,
            useExtrinsicGuess=True,
            iterationsCount=100,
            reprojectionError=4.0,
            confidence=0.999
        )
        
        if success:
            R_new, _ = cv2.Rodrigues(R_new)
            state.camera_poses[frame_idx] = (R_new, t_new)


def incremental_reconstruction(frames, keypoints_list, descriptors_list, K, window_size=5):
    """Perform incremental reconstruction with improved camera pose estimation."""
    
    MIN_BASELINE_ANGLE = 1.0            # Angle thresholds [degrees]
    MAX_BASELINE_ANGLE = 90.0
    REPROJECTION_ERROR_THRESHOLD = 8.0  # Reprojection error threshold [pixels]
    
    # Initialize reconstruction with first pair
    state = initialize_reconstruction(keypoints_list, descriptors_list, K)
    print(f"Initial state: {len(state.points_3d)} points")
    
    # Build complete feature tracks
    all_tracks = build_feature_tracks(keypoints_list, descriptors_list, window_size)
    print(f"Built {len(all_tracks)} feature tracks")
    
    # Process frames sequentially
    processed_frames = {0, 1} 
    
    for new_idx in range(2, len(frames)):
        print(f"\nProcessing frame {new_idx}")
        
        # Try to find best reference frame from already processed frames
        best_ref_idx = None
        best_num_points = 0
        best_points_3d = None
        best_points_2d = None
        
        # Try multiple reference frames
        for ref_idx in sorted(processed_frames, reverse=True)[:3]:
            points_3d = []
            points_2d = []
            
            # Find 3D-2D correspondences through tracks
            for track_id, track in state.point_tracks.items():
                # Find if this track extends to new frame
                new_frame_kp = None
                ref_frame_kp = None
                
                for f, kp in track:
                    if f == ref_idx:
                        ref_frame_kp = kp
                    
                for f, kp in all_tracks.get(track_id, []):
                    if f == new_idx:
                        new_frame_kp = kp
                        
                if ref_frame_kp is not None and new_frame_kp is not None:
                    point_3d = state.points_3d[state.track_to_point[track_id]]
                    point_2d = keypoints_list[new_idx][new_frame_kp].pt
                    points_3d.append(point_3d)
                    points_2d.append(point_2d)
            
            if len(points_3d) > best_num_points:
                best_num_points = len(points_3d)
                best_ref_idx = ref_idx
                best_points_3d = np.array(points_3d, dtype=np.float32)
                best_points_2d = np.array(points_2d, dtype=np.float32)
        
        print(f"Best reference frame: {best_ref_idx} with {best_num_points} correspondences")
        
        if best_num_points < 10:
            print(f"Insufficient correspondences with any reference frame")
            continue
            
        # Debug: Print point shapes and values
        print(f"3D points shape: {best_points_3d.shape}")
        print(f"2D points shape: {best_points_2d.shape}")
        print(f"3D points range: [{best_points_3d.min():.2f}, {best_points_3d.max():.2f}]")
        print(f"2D points range: [{best_points_2d.min():.2f}, {best_points_2d.max():.2f}]")
        
        # Ensure points are properly shaped for PnP
        best_points_3d = best_points_3d.reshape(-1, 3)
        best_points_2d = best_points_2d.reshape(-1, 2)
        
        # Try PnP methods
        for pnp_method in [cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_P3P, cv2.SOLVEPNP_ITERATIVE]:
            try:
                success, R_vec, t, inliers = cv2.solvePnPRansac(
                    objectPoints=best_points_3d,
                    imagePoints=best_points_2d,
                    cameraMatrix=K,
                    distCoeffs=None,
                    flags=pnp_method,
                    iterationsCount=200,
                    reprojectionError=REPROJECTION_ERROR_THRESHOLD,
                    confidence=0.99,
                    useExtrinsicGuess=False
                )
                
                if success:
                    print(f"PnP succeeded with method {pnp_method}")
                    break
            except cv2.error as e:
                print(f"PnP failed with method {pnp_method}: {e}")
                continue
        
        if not success:
            print("All PnP methods failed")
            continue
            
        R, _ = cv2.Rodrigues(R_vec)
        
        # Verify pose by checking reprojection error
        projected_points = cv2.projectPoints(
            best_points_3d[inliers].reshape(-1, 3),
            R_vec,
            t,
            K,
            None
        )[0].reshape(-1, 2)
        
        actual_points = best_points_2d[inliers].reshape(-1, 2)
        reproj_errors = np.linalg.norm(projected_points - actual_points, axis=1)
        mean_error = np.mean(reproj_errors)
        
        print(f"Mean reprojection error: {mean_error:.2f} pixels")
        print(f"Number of inliers: {len(inliers)}")
        
        if mean_error > REPROJECTION_ERROR_THRESHOLD:
            print(f"Reprojection error too high")
            continue
        
        # Check baseline angle
        ref_R = state.camera_poses[best_ref_idx][0]
        ref_t = state.camera_poses[best_ref_idx][1]
        baseline_angle = compute_baseline_angle(ref_R, ref_t, R, t)
        
        print(f"Baseline angle: {baseline_angle:.2f} degrees")
        
        if baseline_angle < MIN_BASELINE_ANGLE or baseline_angle > MAX_BASELINE_ANGLE:
            print(f"Bad baseline angle")
            continue
        
        # Add new camera
        state.camera_poses[new_idx] = (R, t)
        processed_frames.add(new_idx)
        
        # Triangulate new points using all possible pairs
        new_points_count = 0
        for ref_idx in processed_frames:
            if ref_idx == new_idx:
                continue
                
            ref_R, ref_t = state.camera_poses[ref_idx]
            P1 = K @ np.hstack((ref_R, ref_t))
            P2 = K @ np.hstack((R, t))
            
            # Find matches between ref_idx and new_idx through tracks
            for track_id, track in all_tracks.items():
                if track_id in state.track_to_point:
                    continue  # Skip already triangulated points
                    
                ref_kp = None
                new_kp = None
                
                for f, kp in track:
                    if f == ref_idx:
                        ref_kp = kp
                    elif f == new_idx:
                        new_kp = kp
                
                if ref_kp is not None and new_kp is not None:
                    pts1 = np.array([keypoints_list[ref_idx][ref_kp].pt])
                    pts2 = np.array([keypoints_list[new_idx][new_kp].pt])
                    
                    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
                    point_3d = (points_4d / points_4d[3]).T[0, :3]
                    
                    # Check if point is in front of both cameras
                    point_cam1 = ref_R @ point_3d + ref_t.flatten()
                    point_cam2 = R @ point_3d + t.flatten()
                    
                    if point_cam1[2] > 0 and point_cam2[2] > 0:
                        state.points_3d = np.vstack([state.points_3d, point_3d])
                        state.point_tracks[track_id] = track
                        state.track_to_point[track_id] = len(state.points_3d) - 1
                        new_points_count += 1
        
        print(f"Added camera {new_idx}")
        print(f"Added {new_points_count} new points")
        print(f"Total points: {len(state.points_3d)}")
        
        # Optimize the latest camera pose
        optimize_poses_and_points(state, keypoints_list, K)
    return state 