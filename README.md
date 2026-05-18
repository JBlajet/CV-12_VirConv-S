# CV-12 VirConv_DMD3C
Computer vision project attempting to improve the VirConv-S model from the paper: [Virtual Sparse Convolution for Multimodal 3D Object Detection](https://arxiv.org/abs/2303.02314)

A fork of: [VirConv](https://github.com/hailanyi/VirConv)

Changes to repo can be found in Changes.md

# Computer Vision Project 12 - Image+LiDAR-based 3D Object Detection

Goal: Fuse complementary information from RGB cameras (appearance/texture) and LiDAR point clouds (accurate geometry/depth) to improve 3D detection accuracy and robustness, especially for distant, small, or partially occluded objects.

Input: Set of RGB image and LiDAR point cloud

Output: A set of 3D bounding boxes (class, x, y, z, w, h, l, θ)
