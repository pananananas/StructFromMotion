from tqdm import tqdm
import cv2

def detect_features(images):
    """
    Detect SIFT features and compute descriptors with better parameters.
    """
    # Create SIFT with more selective parameters
    sift = cv2.SIFT_create(
        nfeatures=2000,          
        nOctaveLayers=3,         
        contrastThreshold=0.04,  
        edgeThreshold=10,        
        sigma=1.6                
    )
    
    keypoints_list = []
    descriptors_list = []
    num_keypoints = 0
    
    # Convert images to list if it's a generator or other iterable
    images_list = list(images)
    
    for idx, img in tqdm(enumerate(images_list), total=len(images_list), desc="Detecting features"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Add Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
        num_keypoints += len(keypoints)
    
    return keypoints_list, descriptors_list



def match_features(des1, des2, ratio=0.6):  # Stricter ratio test
    """
    Match features with stricter filtering.
    """
    if des1 is None or des2 is None:
        return []
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)  # More checks for better accuracy
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except cv2.error as e:
        print("Error during FLANN matching:", e)
        return []
    
    # Stricter ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance and m.distance < 200:  # Added absolute distance threshold
            good_matches.append(m)
    
    return good_matches