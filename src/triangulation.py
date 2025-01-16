import numpy as np
import cv2


def estimate_pose_and_triangulate(kp1, kp2, matches, K, debug=False):
    
    if len(matches) < 8:
        if debug:
            print("Not enough matches for pose estimation")
        return None, None, None, None

    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Normalize points
    pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None)
    pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None)
    
    # Estimate Essential matrix with RANSAC
    E, mask = cv2.findEssentialMat(
        pts1_norm, 
        pts2_norm, 
        focal=1.0, 
        pp=(0., 0.),
        method=cv2.RANSAC,
        prob=0.999, # parameter for RANSAC
        threshold=3.0/K[0,0]  # Convert threshold to normalized coordinates
    )

    if E is None:
        if debug:
            print("Essential matrix estimation failed")
        return None, None, None, None

    # Recover pose from Essential matrix
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_norm, pts2_norm, mask=mask)
    
    if debug:
        print(f"Number of inliers: {np.sum(mask_pose)}")
        print("Recovered pose:")
        print("R:", R)
        print("t:", t)

    # Verify pose by checking points in front of both cameras
    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = np.hstack((R, t))
    
    points_4d = cv2.triangulatePoints(
        P1, 
        P2, 
        pts1_norm.reshape(-1,2).T, 
        pts2_norm.reshape(-1,2).T
    )
    # Returns 4D points (affine(W*X, W*Y, W*Z, W)), divide by 4th coordinate (scalar W) to get 3D points
    points_3d = (points_4d / points_4d[3]).T[:, :3]
    
    # Check points are in front of both cameras
    depths1 = points_3d[:, 2]
    depths2 = (R @ points_3d.T + t).T[:, 2]
    
    mask_depths = (depths1 > 0) & (depths2 > 0)
    points_3d = points_3d[mask_depths]
    
    if len(points_3d) < len(matches) * 0.3:  # At least 30% points should be valid
        if debug:
            print("Too few points in front of cameras")
        return None, None, None, None
    
    # Scale reconstruction
    scale = 1.0 / np.median(np.abs(points_3d))
    points_3d *= scale
    t *= scale
    
    return R, t, points_3d, mask_depths