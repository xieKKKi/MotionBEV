# MotionBEV: Online LiDAR Moving Object segmentation with Birds' eye view based Appearance and Motion Features
[[**Paper | ArXiv**]](https://arxiv.org)
[[**Video | YouTube**]](https://arxiv.org)
[[**Video | Bilibili**]](https://arxiv.org)

PyTorch implementation for online LiDAR moving object segmentation framework **MotionBEV**.

<pre>
MotionBEV: Online LiDAR Moving Object segmentation with Birds' eye view based Appearance and Motion Features
</pre>

## Overview
MotionBEV is a fast and accurate framework for LiDAR moving object segmentation. We extract spatio-temporal information from consecutive LiDAR scans in bird's eye view domain, and perform multi-modal features fusion with the multi-modality co-attention modules.
<p align="center">
        <img src="imgs/overview.png" width="90%"> 
</p>

We achieve leading performance on SemanticKITTI-MOS benchmark, with 75.8% IoU for moving class and an average inference time of 23ms (on an RTX 3090 GPU).
<p align="center">
        <img src="imgs/kitti08.gif" width="90%"> 
</p>

MotionBEV is able to perform LiDAR-MOS with both mechanical LIDAR such as Velody HDL-64, and solid-state LiDAR with small Fov and non-repetitive scanning mode, such as Livox Avia.
<p align="center">
        <img src="imgs/livox06.gif" width="90%"> 
</p>
