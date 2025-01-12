from features import match_features
import numpy as np
import cv2


def estimate_pose_and_triangulate(kp1, kp2, matches, K, debug=False):
    """
    Estimate camera pose and triangulate 3D points.
    """
    if len(matches) < 8:
        print("Not enough matches to compute Essential matrix.")
        return None, None, None, None

    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    if debug:   
        print(f"Number of matches before RANSAC: {len(matches)}")

    # Compute Essential matrix with more relaxed threshold
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, 
                                  prob=0.999, threshold=3.0)
    if debug:
        print(f"Number of matches after RANSAC: {len(matches)}")
    if E is None:
        if debug:
            print("Essential matrix estimation failed.")
        return None, None, None, None

    # Recover pose
    points_in_front, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    
    if debug:
        print(f"Number of points in front of camera: {points_in_front}")
        print("Recovered Pose:")
        print("Rotation Matrix R:")
        print(R)
        print("Translation Vector t:")
        print(t)

    # Triangulate points
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t))

    # Convert points to homogeneous coordinates
    pts1_h = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]
    pts2_h = cv2.convertPointsToHomogeneous(pts2)[:, 0, :]

    # Triangulate all points
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = (points_4d / points_4d[3]).T[:, :3]

    # Filter points based on positive depth and reprojection error
    depths1 = points_3d[:, 2]
    depths2 = (R @ points_3d.T + t).T[:, 2]
    
    # Keep points with positive depth in both views
    mask_depths = (depths1 > 0) & (depths2 > 0)
    points_3d = points_3d[mask_depths]
    
    # Center the points around origin
    centroid = np.mean(points_3d, axis=0)
    points_3d = points_3d - centroid
    
    # Scale the points to a reasonable size
    scale = 10.0 / np.max(np.abs(points_3d))
    points_3d = points_3d * scale
    t = t * scale  # Scale translation vector accordingly
    
    if debug:
        if len(points_3d) < 10:
            print(f"Too few valid 3D points: {len(points_3d)}")
            return None, None, None, None

        print(f"Number of valid 3D points: {len(points_3d)}")
    return R, t, points_3d, mask_pose


def estimate_poses_and_triangulate_sequence(frames, keypoints_list, descriptors_list, K, debug=False):
    """
    Estimate camera poses and triangulate 3D points for a sequence of frames.
    
    Parameters:
        frames: List of input frames
        keypoints_list: List of keypoints for each frame
        descriptors_list: List of descriptors for each frame
        K: Camera intrinsic matrix
        debug: Boolean for debug printing
    
    Returns:
        all_points_3d: List of 3D points
        camera_poses: List of (R, t) pairs for each camera pose
        point_tracks: Dictionary mapping point indices to their 2D observations
    """
    if len(frames) < 2:
        print("Need at least 2 frames")
        return None, None, None

    # Initialize structures
    camera_poses = []  # List to store all camera poses (R, t)
    all_points_3d = []  # List to store all 3D points
    point_tracks = {}   # Dictionary to store point tracks
    
    # Initialize first camera at origin
    camera_poses.append((np.eye(3), np.zeros((3, 1))))
    
    # Process second frame to establish initial reconstruction
    matches_01 = match_features(descriptors_list[0], descriptors_list[1])
    R, t, points_3d, mask = estimate_pose_and_triangulate(
        keypoints_list[0], keypoints_list[1], matches_01, K, debug)
    
    if R is None:
        print("Failed to initialize with first two frames")
        return None, None, None
        
    camera_poses.append((R, t))
    all_points_3d = points_3d.tolist()
    
    # Initialize point tracks from first two frames
    for idx, match in enumerate(matches_01):
        if mask[idx]:
            point_tracks[len(point_tracks)] = {
                0: keypoints_list[0][match.queryIdx].pt,
                1: keypoints_list[1][match.trainIdx].pt
            }
    
    # Process remaining frames
    for i in range(2, len(frames)):
        if debug:
            print(f"\nProcessing frame {i}")
        
        # Match with multiple previous frames to establish more correspondences
        points_3d = []
        points_2d = []
        used_keypoints = set()  # Track used keypoints to avoid duplicates
        
        # Look at last 5 frames for matches
        for prev_idx in range(max(0, i-5), i):
            matches = match_features(descriptors_list[prev_idx], descriptors_list[i])
            if debug:
                print(f"Matches with frame {prev_idx}: {len(matches)}")
            
            # Find 2D-3D correspondences through point tracks
            for track_id, track in point_tracks.items():
                if prev_idx in track:  # If point was seen in previous frame
                    for match in matches:
                        curr_kp_idx = match.trainIdx
                        if (curr_kp_idx not in used_keypoints and 
                            match.queryIdx == track[prev_idx]):
                            points_3d.append(all_points_3d[track_id])
                            points_2d.append(keypoints_list[i][curr_kp_idx].pt)
                            used_keypoints.add(curr_kp_idx)
                            track[i] = curr_kp_idx  # Add to track
                            break
        
        if len(points_3d) < 8:
            if debug:
                print(f"Not enough correspondences for frame {i}, found {len(points_3d)}")
            continue
            
        # Convert to numpy arrays
        points_3d = np.array(points_3d)
        points_2d = np.array(points_2d)
        
        # Estimate pose using PnP with more robust parameters
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, K, None,
            iterationsCount=500,  # More iterations for better results
            reprojectionError=5.0,  # Slightly more permissive threshold
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
            
        if not success:
            print(f"PnP failed for frame {i}")
            continue
            
        # Convert rotation vector to matrix
        R_new, _ = cv2.Rodrigues(rvec)
        t_new = tvec
        
        # Verify pose by checking reprojection error
        proj_points, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
        proj_points = proj_points.reshape(-1, 2)
        reproj_errors = np.linalg.norm(proj_points - points_2d, axis=1)
        if np.median(reproj_errors) > 10.0:  # Threshold in pixels
            print(f"High reprojection error for frame {i}")
            continue
        
        camera_poses.append((R_new, t_new))
        
        # Triangulate new points with the best previous frame
        best_prev_idx = i - 1  # Default to previous frame
        max_matches = 0
        best_matches = None
        
        # Find the previous frame with most matches
        for prev_idx in range(max(0, i-5), i):
            matches = match_features(descriptors_list[prev_idx], descriptors_list[i])
            if len(matches) > max_matches:
                max_matches = len(matches)
                best_prev_idx = prev_idx
                best_matches = matches
        
        # Filter out already tracked points
        matches_new = [m for m in best_matches if not any(
            i in track and track[i] == m.trainIdx for track in point_tracks.values())]
        
        if matches_new:
            # Get camera matrices
            P1 = K @ np.hstack((camera_poses[best_prev_idx][0], camera_poses[best_prev_idx][1]))
            P2 = K @ np.hstack((R_new, t_new))
            
            pts1 = np.float32([keypoints_list[best_prev_idx][m.queryIdx].pt for m in matches_new])
            pts2 = np.float32([keypoints_list[i][m.trainIdx].pt for m in matches_new])
            
            # Triangulate new points
            points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            new_points_3d = (points_4d / points_4d[3]).T[:, :3]
            
            # Filter new points based on reprojection error and cheirality
            valid_points = []
            valid_matches = []
            
            for j, (point, match) in enumerate(zip(new_points_3d, matches_new)):
                # Check point is in front of both cameras
                point_cam1 = camera_poses[best_prev_idx][0] @ point + camera_poses[best_prev_idx][1].ravel()
                point_cam2 = R_new @ point + t_new.ravel()
                
                if point_cam1[2] > 0 and point_cam2[2] > 0:
                    # Check reprojection error
                    proj1, _ = cv2.projectPoints(
                        point.reshape(1, 3), 
                        cv2.Rodrigues(camera_poses[best_prev_idx][0])[0],
                        camera_poses[best_prev_idx][1], K, None)
                    proj2, _ = cv2.projectPoints(point.reshape(1, 3), rvec, tvec, K, None)
                    
                    error1 = np.linalg.norm(proj1.ravel() - pts1[j])
                    error2 = np.linalg.norm(proj2.ravel() - pts2[j])
                    
                    if error1 < 5.0 and error2 < 5.0:  # 5 pixel threshold
                        valid_points.append(point)
                        valid_matches.append(match)
            
            # Add new points and tracks
            for j, (point, match) in enumerate(zip(valid_points, valid_matches)):
                track_id = len(point_tracks)
                point_tracks[track_id] = {
                    best_prev_idx: keypoints_list[best_prev_idx][match.queryIdx].pt,
                    i: keypoints_list[i][match.trainIdx].pt
                }
                all_points_3d.append(point.tolist())
        
        if debug:
            print(f"Frame {i}: {len(points_2d)} 2D-3D correspondences, "
                  f"{len(valid_points) if matches_new else 0} new points")
    
    return np.array(all_points_3d), camera_poses, point_tracks