# MotionBEV: Attention-Aware Online LiDAR Moving Object segmentation with Birds' Eye View based Appearance and Motion Features
[[**Paper | ArXiv**]](https://arxiv.org/abs/2305.07336)
[[**Video | YouTube**]](https://youtu.be/kOc7gJ72J-g)
[[**Video | Bilibili**]](https://www.bilibili.com/video/BV1Fs4y1G7V2)

PyTorch implementation for online LiDAR moving object segmentation framework **MotionBEV**.

<pre>
MotionBEV: Attention-Aware Online LiDAR Moving Object segmentation with Birds' Eye View based Appearance and Motion Features.
Bo Zhou* ,Jiapeng Xie* , Yan Pan, Jiajie Wu, and Chuanzhao Lu.
</pre>

**The code and pretrained models are coming soon.**
## Overview
MotionBEV is a fast and accurate framework for LiDAR moving object segmentation. We extract spatio-temporal information from consecutive LiDAR scans in bird's eye view domain, and perform multi-modal features fusion with the multi-modality co-attention modules.
<p align="center">
        <img src="imgs/overview.png" width="90%"> 
</p>
<p align="center" style="margin-top: -15px;">
    <span style="color:orange; border-bottom: 1px solid #d9d9d9;
        display: inline-block;
        color: #999;
        text-align: center;">Overview of MotionBEV.</span>
</p>

We achieve leading performance on [SemanticKITTI-MOS benchmark](http://semantic-kitti.org/tasks.html#mos), with 75.8% IoU (combined semantics) for moving class and an average inference time of 23ms (on an RTX 3090 GPU). See the [competition website](https://codalab.lisn.upsaclay.fr/competitions/7088).
<p align="center">
        <img src="imgs/leaderboard.png" width="60%"> 
</p>
<p align="center" style="margin-top: -15px;">
    <span style="color:orange; border-bottom: 1px solid #d9d9d9;
        display: inline-block;
        color: #999;
        text-align: center;">Screenshots of the rankings on the SemanticKITTI-MOS website.</span>
</p>


<p align="center">
        <img src="imgs/kitti08.gif" width="90%"> 
</p>
<p align="center" style="margin-top: -15px;">
    <span style="color:orange; border-bottom: 1px solid #d9d9d9;
        display: inline-block;
        color: #999;
        text-align: center;">Visualization of MOS results on SemanticKITTI validation set.</span>
</p>

MotionBEV is able to perform LiDAR-MOS with both mechanical LIDAR such as Velody HDL-64, and solid-state LiDAR with small Fov and non-repetitive scanning mode, such as Livox Avia.
<p align="center">
        <img src="imgs/livox06.gif" width="90%"> 
</p>
<p align="center" style="margin-top: -15px;">
    <span style="color:orange; border-bottom: 1px solid #d9d9d9;
        display: inline-block;
        color: #999;
        text-align: center;">Visualization of MOS results on SipailouCampus validation set.</span>
</p>
