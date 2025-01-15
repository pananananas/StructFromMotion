# StructFromMotion


## 1. Image Acquisition and Feature Detection
- [ ] Collect overlapping images from different viewpoints
- [ ] Implement feature detection (SIFT/SURF/ORB)
- [ ] Generate feature descriptors for keypoints

## 2. Feature Matching
- [ ] Implement descriptor matching between image pairs
- [ ] Apply ratio test for match filtering
- [ ] Implement RANSAC for outlier removal
- [ ] Verify geometric consistency of matches

## 3. Initial Camera Pose Estimation
- [ ] Select initial image pair with sufficient parallax
- [ ] Compute fundamental/essential matrix
- [ ] Decompose essential matrix for initial camera pose
- [ ] Triangulate initial 3D points

## 4. Incremental Reconstruction
- [ ] Establish 2D-3D correspondences for new images
- [ ] Implement PnP solver for camera pose estimation
- [ ] Set up incremental bundle adjustment
- [ ] Triangulate additional 3D points
- [ ] Add new images to reconstruction iteratively

## 5. Bundle Adjustment
- [ ] Implement sparse bundle adjustment
- [ ] Optimize 3D point positions
- [ ] Refine camera poses
- [ ] Optimize camera intrinsics (if needed)
- [ ] Implement Levenberg-Marquardt optimization

## 6. Dense Reconstruction
- [ ] Implement Multi-View Stereo algorithm
- [ ] Perform depth map estimation
- [ ] Implement depth map fusion
- [ ] Generate dense point cloud

## 7. Post-processing
- [ ] Clean up outlier points
- [ ] Implement scale recovery (if needed)
- [ ] Apply mesh reconstruction
- [ ] Implement texture mapping
- [ ] Export final 3D model

## Dependencies and Requirements
- [ ] Install required libraries (OpenCV, Ceres Solver, etc.)
- [ ] Set up build system
- [ ] Create test dataset
- [ ] Implement visualization tools
- [ ] Set up evaluation metrics




## Potential improvements
- Remove the fixed frame interval and dynamically get the least blurry or shaky frames.
- Checking paralax between initial frames.
- Loop closure detection.