from triangulation import estimate_pose_and_triangulate
from typing import List, Tuple, Dict
from features import match_features
from dataclasses import dataclass
import numpy as np
import logging
import cv2

@dataclass
class ReconstructionState:
    """Stores the state of the reconstruction process."""
    camera_poses: Dict[int, Tuple[np.ndarray, np.ndarray]]  # frame_idx -> (R, t)
    points_3d: np.ndarray
    point_tracks: Dict[int, List[Tuple[int, int]]]  # track_id -> [(frame_idx, keypoint_idx)]
    track_to_point: Dict[int, int]  # track_id -> point_3d_idx
    scale_factors: List[float]

def compute_relative_scale(points_3d_prev: np.ndarray, points_3d_new: np.ndarray) -> float:
    """
    Compute relative scale between two sets of 3D points to maintain consistent scale.
    Uses median distance ratios for robustness.
    """
    if len(points_3d_prev) < 4 or len(points_3d_new) < 4:
        print("Warning: Too few points for reliable scale computation")
        return 1.0

    # Compute pairwise distances for both point sets
    def compute_distances(points):
        dists = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                if dist > 1e-10:  # Avoid zero distances
                    dists.append(dist)
        return np.array(dists)

    dist_prev = compute_distances(points_3d_prev)
    dist_new = compute_distances(points_3d_new)
    
    if len(dist_prev) == 0 or len(dist_new) == 0:
        print("Warning: Could not compute distances for scale estimation")
        return 1.0
    
    # Use median distance ratio for scale
    scale = np.median(dist_prev) / np.median(dist_new)
    
    # Limit scale factor to reasonable range
    scale = np.clip(scale, 0.1, 10.0)
    
    return scale

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

def build_feature_tracks(keypoints_list, descriptors_list, window_size=3):
    """Build consistent feature tracks across multiple frames."""
    tracks = {}
    next_track_id = 0
    keypoint_to_track = [{} for _ in range(len(keypoints_list))]
    
    # Process frames in sequence
    for frame_idx in range(len(keypoints_list) - 1):
        # Match with next few frames in window
        for offset in range(1, min(window_size, len(keypoints_list) - frame_idx)):
            next_idx = frame_idx + offset
            # Use same ratio test as initialization
            matches = match_features(
                descriptors_list[frame_idx], 
                descriptors_list[next_idx],
                ratio=0.8  # Changed from 0.7 to 0.8
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
    
    # Filter tracks - keep only those visible in at least 2 frames (changed from 3)
    filtered_tracks = {k: v for k, v in tracks.items() if len(v) >= 2}
    return filtered_tracks

def triangulate_track(track, keypoints_list, camera_poses, K):
    """
    Triangulate a 3D point from a feature track using all available views.
    """
    # Filter track to only use frames that have camera poses
    valid_track = [(f, kp) for f, kp in track if f in camera_poses]
    
    if len(valid_track) < 2:
        return None  # Need at least 2 views for triangulation
    
    points_2d = []
    projection_matrices = []
    
    for frame_idx, kp_idx in valid_track:
        R, t = camera_poses[frame_idx]
        P = K @ np.hstack((R, t))
        point_2d = keypoints_list[frame_idx][kp_idx].pt
        
        points_2d.append(point_2d)
        projection_matrices.append(P)
    
    points_2d = np.array(points_2d)
    projection_matrices = np.array(projection_matrices)
    
    # Triangulate using first two views
    points_4d = cv2.triangulatePoints(
        projection_matrices[0],
        projection_matrices[1],
        points_2d[0].reshape(-1, 1, 2),
        points_2d[1].reshape(-1, 1, 2)
    )
    
    point_3d = (points_4d / points_4d[3])[:-1].reshape(3)
    
    # Verify reprojection error in all valid views
    max_error = 0
    for i, (frame_idx, kp_idx) in enumerate(valid_track):
        P = projection_matrices[i]
        point_projected = P @ np.append(point_3d, 1)
        point_projected = point_projected[:2] / point_projected[2]
        error = np.linalg.norm(point_projected - points_2d[i])
        max_error = max(max_error, error)
    
    return point_3d if max_error < 5.0 else None

def initialize_reconstruction(keypoints_list, descriptors_list, K) -> ReconstructionState:
    """Initialize reconstruction from first pair of frames."""
    print("Initializing reconstruction from first pair...")
    
    # Use the same ratio test as in the original reconstruction
    matches = match_features(descriptors_list[0], descriptors_list[1], ratio=0.8)  # Changed from 0.7 to 0.8
    print(f"Found {len(matches)} matches for initialization")
    
    if len(matches) < 100:
        raise RuntimeError(f"Insufficient matches for initialization: {len(matches)}")
    
    # Estimate pose and triangulate initial points
    R, t, points_3d, mask = estimate_pose_and_triangulate(
        keypoints_list[0],
        keypoints_list[1],
        matches,
        K,
        debug=True  # Enable debug output to see what's happening
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
        track_to_point={},
        scale_factors=[]
    )
    
    # Initialize point tracks and mapping
    point_idx = 0
    for i, match in enumerate(matches):
        if mask[i]:
            state.point_tracks[i] = [(0, match.queryIdx), (1, match.trainIdx)]
            state.track_to_point[i] = point_idx
            point_idx += 1
    
    return state

def estimate_camera_pose(points_3d, points_2d, K):
    """
    Estimate camera pose using PnP with fallback options.
    """
    if len(points_3d) < 10:
        return None, None, None
    
    # First try simple PnP
    success, R_vec, t = cv2.solvePnP(
        points_3d,
        points_2d,
        K,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        print("Initial PnP failed, trying RANSAC...")
        return None, None, None
    
    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(R_vec)
    
    # Try to refine with RANSAC
    try:
        success, R_vec_ransac, t_ransac, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            K,
            None,
            cv2.Rodrigues(R)[0],
            t,
            useExtrinsicGuess=True,
            iterationsCount=100,
            reprojectionError=8.0,
            confidence=0.99
        )
        
        if success and inliers is not None and len(inliers) >= 10:
            R_ransac, _ = cv2.Rodrigues(R_vec_ransac)
            return R_ransac, t_ransac, inliers.ravel().astype(bool)
            
    except cv2.error as e:
        print(f"RANSAC refinement failed: {e}")
    
    # If RANSAC fails, return initial estimate
    return R, t, np.ones(len(points_3d), dtype=bool)

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

def triangulate_points(points1, points2, P1, P2):
    """Triangulate points from two views."""
    points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points_3d = (points_4d / points_4d[3]).T[:, :3]
    
    # Check points are in front of both cameras
    depths1 = points_3d[:, 2]
    depths2 = (P2[:3, :3] @ points_3d.T + P2[:3, 3:4]).T[:, 2]
    mask = (depths1 > 0) & (depths2 > 0)
    
    return points_3d, mask

def align_reconstructions(points1, points2, R1, t1, R2, t2):
    """
    Align second reconstruction to the first one using common points.
    Returns transformation (R, t, scale) that transforms points2 to points1's coordinate system.
    """
    # Convert points to homogeneous coordinates
    points1_h = np.hstack([points1, np.ones((len(points1), 1))])
    points2_h = np.hstack([points2, np.ones((len(points2), 1))])
    
    # Create transformation matrices
    T1 = np.vstack([np.hstack([R1, t1]), [0, 0, 0, 1]])
    T2 = np.vstack([np.hstack([R2, t2]), [0, 0, 0, 1]])
    
    # Compute relative transformation
    T_rel = np.linalg.inv(T1) @ T2
    
    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3:]
    
    # Compute scale by comparing distances
    dist1 = np.mean([np.linalg.norm(p1 - p2) for i, p1 in enumerate(points1) 
                     for p2 in points1[i+1:]])
    dist2 = np.mean([np.linalg.norm(p1 - p2) for i, p1 in enumerate(points2) 
                     for p2 in points2[i+1:]])
    scale = dist1 / dist2 if dist2 > 0 else 1.0
    
    return R_rel, t_rel, scale

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

def incremental_reconstruction(frames, keypoints_list, descriptors_list, K, window_size=5):
    """Perform incremental reconstruction with improved camera pose estimation."""
    # Parameters - made more lenient
    MIN_BASELINE_ANGLE = 1.0  # degrees
    MAX_BASELINE_ANGLE = 90.0  # degrees
    REPROJECTION_ERROR_THRESHOLD = 8.0  # pixels
    
    # Initialize reconstruction with first pair
    state = initialize_reconstruction(keypoints_list, descriptors_list, K)
    print(f"Initial state: {len(state.points_3d)} points")
    
    # Build complete feature tracks
    all_tracks = build_feature_tracks(keypoints_list, descriptors_list, window_size)
    print(f"Built {len(all_tracks)} feature tracks")
    
    # Process frames sequentially
    processed_frames = {0, 1}  # First two frames are already processed
    
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
        
        # Try different PnP methods
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