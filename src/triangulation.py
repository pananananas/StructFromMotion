from features import match_features
import numpy as np
import cv2

def estimate_pose_and_triangulate(kp1, kp2, matches, K, debug=False):
    """
    Estimate camera pose and triangulate 3D points with improved error handling.
    """
    if len(matches) < 8:
        if debug:
            print("Not enough matches to compute Essential matrix.")
        return None, None, None, None

    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute Essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, 
                                  prob=0.999, threshold=3.0)
    
    if E is None:
        if debug:
            print("Essential matrix estimation failed.")
        return None, None, None, None

    # Recover pose
    points_in_front, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    
    if points_in_front < 10:  # Add minimum points threshold
        if debug:
            print(f"Too few points in front of camera: {points_in_front}")
        return None, None, None, None

    # Triangulate points
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t))

    pts1_h = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]
    pts2_h = cv2.convertPointsToHomogeneous(pts2)[:, 0, :]

    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = (points_4d / points_4d[3]).T[:, :3]

    # Filter points based on positive depth
    depths1 = points_3d[:, 2]
    depths2 = (R @ points_3d.T + t).T[:, 2]
    mask_depths = (depths1 > 0) & (depths2 > 0)
    points_3d = points_3d[mask_depths]

    if len(points_3d) < 10:  # Add minimum points threshold
        if debug:
            print(f"Too few valid 3D points after filtering: {len(points_3d)}")
        return None, None, None, None

    # Center and scale points only if we have valid points
    if len(points_3d) > 0:
        centroid = np.mean(points_3d, axis=0)
        points_3d = points_3d - centroid
        
        # Avoid division by zero in scaling
        max_abs_val = np.max(np.abs(points_3d))
        if max_abs_val > 1e-10:  # Add small threshold
            scale = 10.0 / max_abs_val
            points_3d = points_3d * scale
            t = t * scale
        else:
            if debug:
                print("Points too close to origin, skipping scaling")
    
    return R, t, points_3d, mask_pose

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
                                  prob=0.999, threshold=3.0)  # Increased threshold
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


def reconstruct_sequence(frames, keypoints_list, descriptors_list, K, max_frame_gap=3):
    """
    Reconstruct 3D scene with adaptive frame selection
    """
    all_points_3d = []
    all_cameras = []
    
    # Initialize first camera at origin
    all_cameras.append((np.eye(3), np.zeros((3, 1))))
    
    i = 0
    while i < len(frames) - 1:
        success = False
        
        # Try different frame gaps if initial matching fails
        for frame_gap in range(1, max_frame_gap + 1):
            if i + frame_gap >= len(frames):
                break
                
            matches = match_features(descriptors_list[i], descriptors_list[i + frame_gap])
            
            if len(matches) >= 50:  # Require more matches for reliability
                print(f"Found {len(matches)} matches between frames {i} and {i + frame_gap}")
                
                R, t, points_3d, mask = estimate_pose_and_triangulate(
                    keypoints_list[i], 
                    keypoints_list[i + frame_gap], 
                    matches, 
                    K,
                    debug=True
                )
                
                if R is not None and t is not None and points_3d is not None and len(points_3d) > 20:
                    # Transform points to global coordinate system
                    if len(all_cameras) > 1:
                        R_prev, t_prev = all_cameras[-1]
                        R_new = R_prev @ R
                        t_new = t_prev + R_prev @ t
                        all_cameras.append((R_new, t_new))
                        points_3d = (R_prev @ points_3d.T + t_prev).T
                    else:
                        all_cameras.append((R, t))
                    
                    all_points_3d.extend(points_3d)
                    success = True
                    i += frame_gap
                    break
        
        if not success:
            print(f"Failed to find good matches after frame {i}, skipping...")
            i += 1
    
    if len(all_points_3d) > 0:
        all_points_3d = np.array(all_points_3d)
        print(f"Final reconstruction has {len(all_points_3d)} points and {len(all_cameras)} cameras")
    else:
        print("Warning: No valid points reconstructed")
        return np.array([]), []
    
    return all_points_3d, all_cameras