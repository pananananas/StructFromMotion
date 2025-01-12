from features import match_features
import numpy as np
import cv2


def estimate_pose_and_triangulate(kp1, kp2, matches, K, debug=False):
    """
    Estimate camera pose and triangulate 3D points.
    Returns:
        R: 3x3 rotation matrix of second camera (first camera is at identity)
        t: 3x1 translation vector of second camera
        points_3d: Nx3 array of triangulated 3D points
        mask_pose: mask of inlier points from pose estimation
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

    # Recover pose (R,t) from Essential matrix
    points_in_front, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    
    if debug:
        print(f"Number of points in front of camera: {points_in_front}")
        print("Recovered Pose:")
        print("Rotation Matrix R:")
        print(R)
        print("Translation Vector t:")
        print(t)

    # First camera projection matrix [I|0]
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