from tqdm import tqdm
import numpy as np
import cv2

def detect_features(images, detector_type='SIFT'):

    if detector_type == 'SIFT':
        detector = cv2.SIFT_create(
            nfeatures=2000,          
            nOctaveLayers=3,         
            contrastThreshold=0.04,  
            edgeThreshold=10,        
            sigma=1.6                
        )
    elif detector_type == 'ORB':
        detector = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31
        )
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")
    
    keypoints_list = []
    descriptors_list = []
    num_keypoints = 0
    
    # Convert images to list if it's a generator or other iterable
    images_list = list(images)
    
    for idx, img in tqdm(enumerate(images_list), total=len(images_list), desc=f"Detecting {detector_type} features"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        
        # Convert descriptors to float32 for FLANN matcher compatibility
        if descriptors is not None and detector_type == 'ORB':
            descriptors = descriptors.astype(np.float32)
            
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
        num_keypoints += len(keypoints)
    
    return keypoints_list, descriptors_list, num_keypoints



def match_features(des1, des2, ratio=0.8, cross_check=True):
    """
    Match features with bi-directional verification.
    """
    if des1 is None or des2 is None:
        return []
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Forward match
    matches_12 = flann.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches_12 = []
    for m, n in matches_12:
        if m.distance < ratio * n.distance:
            good_matches_12.append(m)
    
    if not cross_check:
        return good_matches_12
    
    # Backward match for verification
    matches_21 = flann.knnMatch(des2, des1, k=2)
    
    # Apply ratio test
    good_matches_21 = []
    for m, n in matches_21:
        if m.distance < ratio * n.distance:
            good_matches_21.append(m)
    
    # Cross-check
    verified_matches = []
    for match_12 in good_matches_12:
        for match_21 in good_matches_21:
            if match_12.queryIdx == match_21.trainIdx and match_12.trainIdx == match_21.queryIdx:
                verified_matches.append(match_12)
                break
    
    return verified_matches